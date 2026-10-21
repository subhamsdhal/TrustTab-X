"""Local-only task-specific FastAPI inference and monitoring service."""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .m5_artifact import FEATURES, SCHEMA_VERSION, ModelBundle, load_bundle
from .m5_monitor import ServiceMonitor, delayed_label_report, drift_report

LOG = logging.getLogger("trusttabx.service")
STATIC_DIR = Path(__file__).with_name("m5_web")


class ACSRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    AGEP: int = Field(ge=16, le=120)
    WKHP: int = Field(ge=0, le=100)
    COW: str = Field(min_length=1, max_length=32)
    SCHL: str = Field(min_length=1, max_length=32)
    MAR: str = Field(min_length=1, max_length=32)
    OCCP: str = Field(min_length=1, max_length=32)
    POBP: str = Field(min_length=1, max_length=32)
    RELP: str = Field(min_length=1, max_length=32)
    SEX: str = Field(min_length=1, max_length=32)
    RAC1P: str = Field(min_length=1, max_length=32)

    @field_validator("COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "SEX", "RAC1P")
    @classmethod
    def no_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Blank category")
        return value


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str
    record: ACSRecord


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str
    records: list[ACSRecord] = Field(min_length=1, max_length=100)


class MonitorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str
    records: list[ACSRecord] = Field(min_length=1, max_length=1000)


class DelayedLabels(BaseModel):
    labels: list[int]
    probabilities: list[float]


def _check_schema(version: str) -> None:
    if version != SCHEMA_VERSION:
        raise HTTPException(status_code=422, detail={"code": "schema_mismatch",
                                                    "expected": SCHEMA_VERSION})


def _frame(records: list[ACSRecord]) -> pd.DataFrame:
    return pd.DataFrame([record.model_dump() for record in records], columns=FEATURES)


def create_app(artifact_dir: str | Path | None = None) -> FastAPI:
    location = Path(artifact_dir or os.environ.get("TRUSTTABX_MODEL_DIR", "outputs/m5_acs_model"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.bundle = load_bundle(location)
        app.state.monitor = ServiceMonitor(app.state.bundle.manifest["model_version"])
        yield

    app = FastAPI(title="TrustTab-X ACSIncome Decision Support", version="0.1.0",
                  description="Local research demonstration. Fixed expert; no automatic routing.",
                  lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        if hasattr(request.app.state, "monitor"):
            request.app.state.monitor.error()
        # Omit raw field values from response and logs.
        return JSONResponse(status_code=422, content={"detail": {
            "code": "invalid_request", "fields": [
                {"path": ".".join(map(str, issue["loc"])), "type": issue["type"]}
                for issue in exc.errors()]}})

    @app.exception_handler(HTTPException)
    async def expected_error(request: Request, exc: HTTPException):
        if hasattr(request.app.state, "monitor"):
            request.app.state.monitor.error()
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    def bundle(request: Request) -> ModelBundle:
        return request.app.state.bundle

    @app.get("/health")
    def health(request: Request):
        return {"status": "ready", "model_version": bundle(request).manifest["model_version"]}

    @app.get("/model")
    def model(request: Request):
        manifest = bundle(request).manifest
        return {key: manifest[key] for key in ("model_version", "schema_version", "task",
                "selected_expert", "features", "target_mapping", "research_decision")}

    @app.get("/metrics")
    def metrics(request: Request):
        current = bundle(request)
        return request.app.state.monitor.snapshot(current.manifest["reference"])

    @app.get("/")
    def dashboard():
        return FileResponse(STATIC_DIR / "index.html")

    def predict_many(request: Request, records: list[ACSRecord]) -> list[dict]:
        current = bundle(request)
        frame = _frame(records)
        source_categories = current.manifest["reference"]["categorical"]
        unknown = sorted({name for name, allowed in source_categories.items()
                          if any(value not in allowed for value in frame[name].astype(str))})
        if unknown:
            raise HTTPException(status_code=422, detail={"code": "unknown_category",
                                                        "features": unknown})
        start = perf_counter()
        try:
            probability = current.pipeline.predict_proba(frame)[:, 1]
        except (ValueError, TypeError):
            raise HTTPException(status_code=422, detail={"code": "invalid_features"}) from None
        elapsed = (perf_counter() - start) * 1000
        results = []
        for p in probability:
            decision_id = str(uuid4())
            output = {"decision_id": decision_id,
                      "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                      "model_version": current.manifest["model_version"],
                      "schema_version": SCHEMA_VERSION,
                      "selected_expert": current.manifest["selected_expert"],
                      "probability_positive": float(p), "predicted_class": int(p >= 0.5),
                      "decision_threshold": 0.5,
                      "abstention": False,
                      "reliability_warning": "OOD reliability unverified; monitor drift"}
            LOG.info(json.dumps({"event": "prediction", "decision_id": decision_id,
                                 "model_version": output["model_version"],
                                 "probability_positive": output["probability_positive"],
                                 "latency_ms": elapsed / len(records)}))
            request.app.state.monitor.record(float(p), elapsed / len(records))
            results.append(output)
        return results

    @app.post("/predict")
    def predict(payload: PredictRequest, request: Request):
        _check_schema(payload.schema_version)
        return predict_many(request, [payload.record])[0]

    @app.post("/predict-batch")
    def predict_batch(payload: BatchRequest, request: Request):
        _check_schema(payload.schema_version)
        return {"mode": "fixed_online_policy", "predictions": predict_many(request,
                                                                         payload.records)}

    @app.post("/monitor/drift")
    def monitor_drift(payload: MonitorRequest, request: Request):
        _check_schema(payload.schema_version)
        current = bundle(request)
        frame = _frame(payload.records)
        probabilities = current.pipeline.predict_proba(frame)[:, 1]
        return drift_report(current.manifest["reference"],
                            [record.model_dump() for record in payload.records],
                            probabilities.tolist())

    @app.post("/monitor/delayed-labels")
    def monitor_delayed(payload: DelayedLabels):
        try:
            return delayed_label_report(payload.labels, payload.probabilities)
        except ValueError:
            raise HTTPException(status_code=422, detail={"code": "invalid_delayed_labels"}) from None

    return app


app = create_app()
