"""Train and validate a task-specific, source-selected ACSIncome model bundle."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import secrets
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import psutil
import sklearn
from sklearn.metrics import brier_score_loss

from .acs_domain import (
    CATEGORICAL,
    FEATURES,
    SOURCE_STATES,
    YEAR,
    bounded_acs_train,
    load_acs_sample,
    split_acs,
)
from .benchmark import index_hash, make_experts, make_preprocessor
from .reliability import probability_metrics

SCHEMA_VERSION = "acs-income-2018-v1"
MODEL_VERSION = "m5-acs-fixed-1"
SEED = 264738
NUMERIC_RANGES = {"AGEP": (16, 120), "WKHP": (0, 100)}


def digest_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class ModelBundle:
    pipeline: object
    manifest: dict
    root: Path


def train_acs_bundle(output: str | Path = "outputs/m5_acs_model",
                     *, track_mlflow: bool = False) -> dict:
    """Fit on 2018 source households; selection chooses one fixed expert."""
    data = load_acs_sample(seed=SEED)
    parts = split_acs(data, SEED)
    fit = bounded_acs_train(data, parts["train"], 14000, SEED)
    pre = make_preprocessor(data.X, [name in CATEGORICAL for name in FEATURES])
    candidates = make_experts(pre, SEED, advanced=True)
    scores, models, fit_seconds = {}, {}, {}
    for name, model in candidates.items():
        tic = perf_counter()
        model.fit(data.X.iloc[fit], data.y[fit])
        fit_seconds[name] = perf_counter() - tic
        scores[name] = float(brier_score_loss(
            data.y[parts["selection"]],
            model.predict_proba(data.X.iloc[parts["selection"]])[:, 1],
        ))
        models[name] = model
    selected = min(scores, key=scores.get)
    chosen = models[selected]
    calibration_probability = chosen.predict_proba(data.X.iloc[parts["calibration"]])[:, 1]
    # Calibration is an audit set only: M4.2 did not support automatic abstention.
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    model_file = root / "pipeline.joblib"
    joblib.dump(chosen, model_file, compress=3)
    ref = build_reference(data.X.iloc[fit],
                          chosen.predict_proba(data.X.iloc[fit])[:, 1])
    manifest = {
        "artifact_format": 1, "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION, "task": "ACSIncome 2018 > $56,000",
        "prediction_semantics": "research decision support, not a high-stakes decision",
        "selected_expert": selected, "seed": SEED, "survey_year": YEAR,
        "source_states": list(SOURCE_STATES),
        "features": list(FEATURES), "categorical": list(CATEGORICAL),
        "feature_types": {name: ("category_code_string" if name in CATEGORICAL
                                 else "integer") for name in FEATURES},
        "numeric_ranges": {key: list(value) for key, value in NUMERIC_RANGES.items()},
        "target_mapping": {"0": "PINCP <= 56000", "1": "PINCP > 56000"},
        "selection_brier_by_expert": scores,
        "calibration_metrics": probability_metrics(data.y[parts["calibration"]],
                                                   calibration_probability),
        "split_hashes": {"fit": index_hash(fit),
                         "selection": index_hash(parts["selection"]),
                         "calibration": index_hash(parts["calibration"])},
        "split_sizes": {"fit": len(fit), "selection": len(parts["selection"]),
                        "calibration": len(parts["calibration"])},
        "source_file_sha256": {state: data.sample_manifest[state]["source_file_sha256"]
                               for state in SOURCE_STATES},
        "dataset_provenance": "US Census 2018 ACS PUMS via Folktables 0.0.12",
        "data_rights_note": "Verify Census and Folktables terms before redistribution or real use",
        "research_decision": "M4.2 C: no adaptive advantage demonstrated",
        "fit_seconds_by_expert": fit_seconds,
        "peak_working_set_kib": getattr(psutil.Process().memory_info(), "peak_wset",
                                        psutil.Process().memory_info().rss) // 1024,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "reference": ref,
        "pipeline_file": model_file.name, "pipeline_sha256": digest_file(model_file),
        "trusted_local_artifact_only": True,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n",
                                        encoding="utf-8")
    if track_mlflow:
        log_mlflow(manifest, root)
    return manifest


def log_mlflow(manifest: dict, root: Path) -> None:
    """Optional local-only experiment record; no hosted tracking URI."""
    import mlflow

    tracking = root / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{tracking.as_posix()}")
    mlflow.set_experiment("trusttabx-m5-acs-fixed")
    with mlflow.start_run(run_name=manifest["model_version"]):
        mlflow.log_params({"selected_expert": manifest["selected_expert"],
                           "seed": manifest["seed"],
                           "schema_version": manifest["schema_version"],
                           "pipeline_sha256": manifest["pipeline_sha256"]})
        mlflow.log_metrics({"source_selection_brier": manifest[
            "selection_brier_by_expert"][manifest["selected_expert"]],
                            "source_calibration_brier": manifest[
                                "calibration_metrics"]["brier"]})
        mlflow.log_artifact(str(root / "manifest.json"))


def load_bundle(root: str | Path) -> ModelBundle:
    """Load an explicitly trusted local artifact after schema/hash/version checks."""
    path = Path(root).resolve()
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Model manifest is missing")
    # The operator pins this digest through deployment configuration. A hash
    # shipped inside the same bundle cannot establish who approved the bundle.
    approved = os.environ.get("TRUSTTABX_APPROVED_MANIFEST_SHA256", "")
    actual = digest_file(manifest_path)
    if len(approved) != 64 or not secrets.compare_digest(approved, actual):
        raise PermissionError("Manifest digest differs from operator-approved digest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("artifact_format") != 1
            or manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("features") != list(FEATURES)
            or manifest.get("feature_types") != {
                name: ("category_code_string" if name in CATEGORICAL else "integer")
                for name in FEATURES}
            or manifest.get("model_version") != MODEL_VERSION
            or manifest.get("sklearn_version") != sklearn.__version__):
        raise ValueError("Model manifest is incompatible with this service")
    filename = manifest.get("pipeline_file")
    if filename != "pipeline.joblib":
        raise ValueError("Invalid pipeline path in model manifest")
    model_file = path / filename
    if not model_file.is_file() or digest_file(model_file) != manifest.get("pipeline_sha256"):
        raise ValueError("Model pipeline is missing or corrupted")
    # joblib uses pickle. Load only an operator-approved local directory.
    if os.environ.get("TRUSTTABX_TRUST_LOCAL_ARTIFACT", "") != "1":
        raise PermissionError("Set TRUSTTABX_TRUST_LOCAL_ARTIFACT=1 for an approved local bundle")
    model = joblib.load(model_file)
    if list(model.feature_names_in_) != list(FEATURES):
        raise ValueError("Loaded pipeline feature schema differs from manifest")
    return ModelBundle(model, manifest, path)


def build_reference(frame: pd.DataFrame, probability: np.ndarray) -> dict:
    """Source-fit reference summaries, not target-derived thresholds."""
    numeric = {}
    for feature in NUMERIC_RANGES:
        values = pd.to_numeric(frame[feature], errors="coerce").dropna()
        numeric[feature] = {"mean": float(values.mean()),
                            "sd": float(values.std()),
                            "missing_rate": float(frame[feature].isna().mean())}
    categorical = {}
    for feature in CATEGORICAL:
        categorical[feature] = {
            str(key): float(value) for key, value in
            frame[feature].astype(str).value_counts(normalize=True).items()
        }
    return {"numeric": numeric, "categorical": categorical,
            "probability_mean": float(np.mean(probability)),
            "probability_sd": float(np.std(probability)),
            "source_rows": len(frame)}
