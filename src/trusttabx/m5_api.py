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
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .m5_artifact import FEATURES, SCHEMA_VERSION, ModelBundle, load_bundle
from .m5_monitor import ServiceMonitor, delayed_label_report, drift_report
from .m5_request_guard import RequestGuard
from .m5_security import RateLimiter, SecurityConfig, require_admin, require_operator

LOG = logging.getLogger("trusttabx.service")
LOG.setLevel(logging.INFO)
if not LOG.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOG.addHandler(handler)
LOG.propagate = False
STATIC_DIR = Path(__file__).with_name("m5_web")
MANUSCRIPT_PATH = Path.cwd() / "reports" / "paper" / "TrustTab-X-manuscript.pdf"


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
        app.state.security = SecurityConfig.from_env()
        app.state.bundle = load_bundle(location)
        app.state.monitor = ServiceMonitor(app.state.bundle.manifest["model_version"])
        app.state.rate_limiter = RateLimiter(app.state.security.rate_limit_per_minute)
        yield

    app = FastAPI(title="TrustTab-X ACSIncome Decision Support", version="0.1.0",
                  description="Local research demonstration. Fixed expert; no automatic routing.",
                  lifespan=lifespan)
    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")
    app.add_middleware(RequestGuard,
                       max_bytes=int(os.getenv("TRUSTTABX_MAX_REQUEST_BYTES", "262144")),
                       timeout_seconds=float(os.getenv("TRUSTTABX_REQUEST_TIMEOUT_SECONDS", "15")))
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=[host.strip() for host in os.getenv(
        "TRUSTTABX_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")])

    @app.middleware("http")
    async def audit(request: Request, call_next):
        request_id = str(uuid4())
        started = perf_counter()
        try:
            response = await call_next(request)
        except Exception:  # noqa: BLE001 - HTTP boundary hides internal failures from clients.
            response = JSONResponse(status_code=500, content={"detail": {"code": "internal_error"}})
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path == "/" or request.url.path.startswith("/assets/"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'none'; script-src 'self'; style-src 'self'; "
                "connect-src 'self'; img-src 'self'; object-src 'none'; "
                "base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
        current = getattr(request.app.state, "bundle", None)
        matched_route = request.scope.get("route")
        record = {
            "event": "http_request", "request_id": request_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "operation": matched_route.path if matched_route else "unmatched",
            "method": request.method,
            "status": response.status_code, "identity": getattr(
                request.state, "operation_identity", "anonymous"),
            "latency_ms": round((perf_counter() - started) * 1000, 3),
            "model_version": current.manifest["model_version"] if current else None,
            "artifact_sha256_prefix": current.manifest["pipeline_sha256"][:16]
            if current else None,
            "decision_id": getattr(request.state, "decision_id", None),
            "drift_alert": getattr(request.state, "drift_alert", None),
            "validation_error": response.status_code == 422,
        }
        LOG.info(json.dumps(record))
        return response

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
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail},
                            headers=exc.headers)

    def bundle(request: Request) -> ModelBundle:
        return request.app.state.bundle

    @app.get("/health")
    def health(request: Request):
        return {"status": "ready", "model_version": bundle(request).manifest["model_version"]}

    @app.get("/model", dependencies=[Depends(require_operator)])
    def model(request: Request):
        manifest = bundle(request).manifest
        return {key: manifest[key] for key in ("model_version", "schema_version", "task",
                "selected_expert", "features", "target_mapping", "research_decision")}

    @app.get("/metrics", dependencies=[Depends(require_operator)])
    def metrics(request: Request):
        current = bundle(request)
        return request.app.state.monitor.snapshot(current.manifest["reference"])

    @app.get("/")
    def dashboard():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/research/manuscript", include_in_schema=False)
    def research_manuscript():
        if not MANUSCRIPT_PATH.is_file():
            raise HTTPException(status_code=404, detail={"code": "manuscript_unavailable"})
        return FileResponse(MANUSCRIPT_PATH, media_type="application/pdf")

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
            if not results:
                request.state.decision_id = decision_id
            request.app.state.monitor.record(float(p), elapsed / len(records))
            results.append(output)
        return results

    @app.post("/predict", dependencies=[Depends(require_operator)])
    def predict(payload: PredictRequest, request: Request):
        _check_schema(payload.schema_version)
        return predict_many(request, [payload.record])[0]

    @app.post("/predict-batch", dependencies=[Depends(require_operator)])
    def predict_batch(payload: BatchRequest, request: Request):
        _check_schema(payload.schema_version)
        return {"mode": "fixed_online_policy", "predictions": predict_many(request,
                                                                         payload.records)}

    @app.post("/monitor/drift", dependencies=[Depends(require_operator)])
    def monitor_drift(payload: MonitorRequest, request: Request):
        _check_schema(payload.schema_version)
        current = bundle(request)
        frame = _frame(payload.records)
        probabilities = current.pipeline.predict_proba(frame)[:, 1]
        report = drift_report(current.manifest["reference"],
                              [record.model_dump() for record in payload.records],
                              probabilities.tolist())
        request.state.drift_alert = report["any_feature_alert"] or report["probability_alert"]
        return report

    @app.post("/monitor/delayed-labels", dependencies=[Depends(require_admin)])
    def monitor_delayed(payload: DelayedLabels):
        try:
            return delayed_label_report(payload.labels, payload.probabilities)
        except ValueError:
            raise HTTPException(status_code=422, detail={"code": "invalid_delayed_labels"}) from None

    return app


app = create_app()
