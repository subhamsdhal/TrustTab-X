"""Offline M5 contract tests; no ACS download or hosted service."""

import json

import joblib
import pandas as pd
import pytest
import sklearn
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from trusttabx.benchmark import make_preprocessor
from trusttabx.m5_api import create_app
from trusttabx.m5_artifact import FEATURES, MODEL_VERSION, SCHEMA_VERSION, digest_file, load_bundle
from trusttabx.m5_monitor import delayed_label_report, drift_report


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("TRUSTTABX_TRUST_LOCAL_ARTIFACT", "1")
    record = {"AGEP": 39, "WKHP": 40, "COW": "1.0", "SCHL": "21.0",
              "MAR": "1", "OCCP": "2310.0", "POBP": "42", "RELP": "0",
              "SEX": "1", "RAC1P": "1"}
    rows = []
    for age, work, label in [(22, 20, 0), (28, 30, 0), (35, 40, 0),
                              (43, 42, 1), (55, 50, 1), (62, 44, 1)]:
        rows.append({**record, "AGEP": age, "WKHP": work})
    frame = pd.DataFrame(rows, columns=FEATURES)
    pipeline = Pipeline([
        ("preprocess", make_preprocessor(frame, [name not in ("AGEP", "WKHP")
                                                for name in FEATURES])),
        ("model", LogisticRegression(random_state=17)),
    ])
    pipeline.fit(frame, [0, 0, 0, 1, 1, 1])
    model_file = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, model_file)
    reference = {"numeric": {"AGEP": {"mean": 40.0, "sd": 10.0},
                             "WKHP": {"mean": 40.0, "sd": 10.0}},
                 "categorical": {key: {record[key]: 1.0} for key in FEATURES[2:]},
                 "probability_mean": 0.5}
    manifest = {"artifact_format": 1, "schema_version": SCHEMA_VERSION,
                "model_version": MODEL_VERSION, "task": "ACS fixture",
                "selected_expert": "logistic_regression", "features": list(FEATURES),
                "feature_types": {name: ("category_code_string" if name not in
                                         ("AGEP", "WKHP") else "integer") for name in FEATURES},
                "target_mapping": {"0": "low", "1": "high"},
                "research_decision": "M4.2 C", "reference": reference,
                "sklearn_version": sklearn.__version__, "pipeline_file": model_file.name,
                "pipeline_sha256": digest_file(model_file)}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path, record


def test_prediction_contract_and_validation(artifact):
    path, record = artifact
    with TestClient(create_app(path)) as client:
        assert client.get("/health").json()["status"] == "ready"
        assert client.get("/model").json()["model_version"] == MODEL_VERSION
        request = {"schema_version": SCHEMA_VERSION, "record": record}
        first = client.post("/predict", json=request)
        second = client.post("/predict", json=request)
        assert first.status_code == second.status_code == 200
        a, b = first.json(), second.json()
        assert a["probability_positive"] == b["probability_positive"]
        assert a["decision_id"] != b["decision_id"]
        assert a["model_version"] == MODEL_VERSION and a["abstention"] is False
        batch = client.post("/predict-batch", json={"schema_version": SCHEMA_VERSION,
                                                    "records": [record, record]})
        assert batch.status_code == 200 and len(batch.json()["predictions"]) == 2
        assert client.get("/metrics").json()["requests"] == 4
        assert client.post("/predict", json={**request, "schema_version": "bad"}).status_code == 422
        assert client.post("/predict", json={"schema_version": SCHEMA_VERSION,
                                            "record": {k: v for k, v in record.items()
                                                       if k != "AGEP"}}).status_code == 422
        assert client.post("/predict", json={**request,
                                            "record": {**record, "COW": "unseen"}}).status_code == 422
        assert client.post("/predict", json={**request,
                                            "record": {**record, "AGEP": 180}}).status_code == 422
        assert client.post("/predict-batch", json={"schema_version": SCHEMA_VERSION,
                                                  "records": [record] * 101}).status_code == 422
        assert client.get("/metrics").json()["errors"] == 5
        assert client.get("/").status_code == 200


def test_artifact_rejects_missing_corrupt_untrusted_and_incompatible(artifact, monkeypatch):
    path, _ = artifact
    assert load_bundle(path).manifest["model_version"] == MODEL_VERSION
    monkeypatch.delenv("TRUSTTABX_TRUST_LOCAL_ARTIFACT")
    with pytest.raises(PermissionError):
        load_bundle(path)
    monkeypatch.setenv("TRUSTTABX_TRUST_LOCAL_ARTIFACT", "1")
    manifest_file = path / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    manifest["schema_version"] = "wrong"
    manifest_file.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incompatible"):
        load_bundle(path)
    manifest["schema_version"] = SCHEMA_VERSION
    manifest_file.write_text(json.dumps(manifest))
    (path / "pipeline.joblib").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupted"):
        load_bundle(path)
    manifest_file.unlink()
    with pytest.raises(FileNotFoundError):
        load_bundle(path)


def test_monitor_detects_controlled_shift_and_delayed_labels(artifact):
    path, record = artifact
    reference = json.loads((path / "manifest.json").read_text())["reference"]
    report = drift_report(reference, [{**record, "AGEP": 80}], [0.9])
    assert report["numeric"]["AGEP"]["alert"]
    assert report["probability_alert"]
    assert report["performance_status"].startswith("labels unavailable")
    labels = delayed_label_report([0, 1], [0.2, 0.8])
    assert labels["brier"] == pytest.approx(0.04)
    with pytest.raises(ValueError):
        delayed_label_report([0, 2], [0.2, 0.8])
    with pytest.raises(ValueError):
        delayed_label_report([0, 1], [0.2, 1.2])
