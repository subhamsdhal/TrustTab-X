"""Run the approved local ACS service against cached held-out public ACS rows.

Aggregate-only portfolio record; never writes raw features, labels, keys, or model binaries.
Requires the previously trained and independently reviewed local M5 artifact and ACS cache.
"""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path

from fastapi.testclient import TestClient

from trusttabx.acs_domain import load_acs_sample, split_acs
from trusttabx.m5_api import create_app
from trusttabx.m5_artifact import SCHEMA_VERSION, digest_file

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "outputs/m5_acs_model"
TRACKED = ROOT / "reports/results/m5_acs_artifact_summary.json"
OUTPUT = ROOT / "reports/results/m7_local_demo.json"


def checked(response, expected: int = 200) -> dict:
    assert response.status_code == expected, (response.status_code, response.text[:250])
    return response.json()


def main() -> None:
    if not (ARTIFACT / "manifest.json").is_file():
        raise FileNotFoundError("Train or provide the locally approved M5 ACS bundle first")
    manifest = json.loads((ARTIFACT / "manifest.json").read_text(encoding="utf-8"))
    tracked = json.loads(TRACKED.read_text(encoding="utf-8"))
    assert manifest["pipeline_sha256"] == tracked["pipeline_sha256"]
    assert manifest["split_hashes"] == tracked["split_hashes"]
    assert manifest["selected_expert"] == "hist_gradient_boosting"

    # Set process-local secrets only after checking the bundle against tracked provenance.
    operator = secrets.token_urlsafe(48)
    admin = secrets.token_urlsafe(48)
    os.environ.update({
        "TRUSTTABX_TRUST_LOCAL_ARTIFACT": "1",
        "TRUSTTABX_APPROVED_MANIFEST_SHA256": digest_file(ARTIFACT / "manifest.json"),
        "TRUSTTABX_API_KEY": operator,
        "TRUSTTABX_ADMIN_KEY": admin,
        "TRUSTTABX_ALLOWED_HOSTS": "localhost,127.0.0.1,testserver",
    })

    data = load_acs_sample(ROOT / "data/raw/acs2018", seed=manifest["seed"])
    for state, expected in manifest["source_file_sha256"].items():
        assert data.sample_manifest[state]["source_file_sha256"] == expected, state
    parts = split_acs(data, manifest["seed"])
    allowed = manifest["reference"]["categorical"]
    def supported(index: int) -> bool:
        row = data.X.iloc[index]
        return all(str(row[name]) in categories for name, categories in allowed.items())

    eligible_id = [int(index) for index in parts["id_test"] if supported(int(index))]
    assert len(eligible_id) >= 100
    id_rows = eligible_id[:100]
    target_rows = parts["ood_test"][:100]
    target_states = sorted(set(data.state[target_rows]))

    def records(indices) -> list[dict]:
        frame = data.X.iloc[indices][manifest["features"]]
        return [
            {key: int(value) if key in ("AGEP", "WKHP") else str(value)
             for key, value in row.items()}
            for row in frame.to_dict(orient="records")
        ]

    id_records = records(id_rows)
    target_records = records(target_rows)
    with TestClient(create_app(ARTIFACT)) as client:
        health = checked(client.get("/health"))
        dashboard = client.get("/")
        assert dashboard.status_code == 200 and "script-src 'self'" in dashboard.headers["Content-Security-Policy"]
        checked(client.get("/model"), 401)
        operator_headers = {"X-TrustTabX-Key": operator}
        model = checked(client.get("/model", headers=operator_headers))
        prediction = checked(client.post("/predict", headers=operator_headers, json={
            "schema_version": SCHEMA_VERSION, "record": id_records[0]}))
        batch = checked(client.post("/predict-batch", headers=operator_headers, json={
            "schema_version": SCHEMA_VERSION, "records": id_records}))
        assert batch["mode"] == "fixed_online_policy"
        assert len(batch["predictions"]) == len(id_rows)
        drift = checked(client.post("/monitor/drift", headers=operator_headers, json={
            "schema_version": SCHEMA_VERSION, "records": target_records}))
        checked(client.post("/monitor/delayed-labels", headers=operator_headers, json={
            "labels": [0, 1], "probabilities": [0.2, 0.8]}), 403)
        labels = checked(client.post("/monitor/delayed-labels", headers={
            "X-TrustTabX-Key": admin}, json={
            "labels": [int(v) for v in data.y[id_rows]],
            "probabilities": [p["probability_positive"] for p in batch["predictions"]],
        }))
        metrics = checked(client.get("/metrics", headers=operator_headers))

    report = {
        "scope": "local in-process FastAPI portfolio demonstration on cached public ACS rows; descriptive subset only",
        "research_decision": "M4.2 C - no adaptive advantage; fixed expert",
        "release_decision": "M5.1 B - local demo only",
        "artifact": {
            "model_version": manifest["model_version"],
            "selected_expert": manifest["selected_expert"],
            "pipeline_sha256": manifest["pipeline_sha256"],
            "manifest_sha256": digest_file(ARTIFACT / "manifest.json"),
            "tracked_pipeline_and_split_hashes_match": True,
            "source_archive_hashes_match": True,
        },
        "health": health,
        "dashboard_status": dashboard.status_code,
        "unauthorized_model_status": 401,
        "model_task": model["task"],
        "schema_version": model["schema_version"],
        "prediction": {k: prediction[k] for k in ("model_version", "selected_expert",
            "probability_positive", "predicted_class", "decision_threshold", "abstention",
            "reliability_warning")},
        "drift": {
            "cohort": "first 100 deterministic New England target sample rows (CT only under state ordering); no target labels used",
            "n": len(target_rows),
            "states": target_states,
            "any_feature_alert": drift["any_feature_alert"],
            "probability_alert": drift["probability_alert"],
            "numeric_alerts": {k: bool(v["alert"]) for k, v in drift["numeric"].items()},
            "performance_status": drift["performance_status"],
        },
        "delayed_labels": {
            "cohort": "first 100 deterministic household-disjoint source ID-test rows accepted by the artifact's categorical schema; feature-only filter",
            "n": len(id_rows),
            "observed_positive_rate": float(data.y[id_rows].mean()),
            "metrics": labels,
            "limitation": "Feature-only service-eligible subset; descriptive demo, not a preregistered estimate or deployment validation",
        },
        "telemetry": {k: metrics[k] for k in ("requests", "errors")},
        "admin_only_delayed_labels_operator_status": 403,
        "dashboard_csp_present": True,
        "raw_records_written": False,
        "keys_written": False,
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "model": report["artifact"]["model_version"],
                      "drift_alert": report["drift"]["any_feature_alert"],
                      "id_demo_brier": labels["brier"]}))


if __name__ == "__main__":
    main()
