"""Pre-call UCI Bank Marketing benchmark with locked chronological partitions."""

from __future__ import annotations

import json
import platform
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import pandas as pd
import psutil
import sklearn
from sklearn.metrics import brier_score_loss

from .benchmark import make_experts, make_preprocessor
from .conformal import SplitConformalClassifier
from .evaluation import evaluate_binary

SOURCE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
ARCHIVE_SHA256 = "a607b5edab6c6c75ce09c39142a77702c38123bd5aa7ae89a63503bbe17d65cd"
NUMERIC = ("age", "pdays", "previous")
CATEGORICAL = (
    "job", "marital", "education", "default", "housing", "loan", "contact", "month",
    "day_of_week", "poutcome",
)
FEATURES = NUMERIC + CATEGORICAL
EXCLUDED = (
    "duration", "campaign", "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed",
)


def chronological_partitions(n: int) -> dict[str, np.ndarray]:
    """Contiguous 50/15/10/10/15 percent windows in original row order."""
    if n < 100:
        raise ValueError("Too few rows for five chronological partitions")
    bounds = [0, round(0.5 * n), round(0.65 * n), round(0.75 * n), round(0.85 * n), n]
    names = ("fit", "selection", "calibration", "near_future", "final_future")
    result = {name: np.arange(start, end) for name, start, end in
              zip(names, bounds[:-1], bounds[1:])}
    if any(len(idx) == 0 for idx in result.values()):
        raise ValueError("Empty chronological partition")
    return result


def load_bank_archive(path: str | Path) -> tuple[pd.DataFrame, np.ndarray, str]:
    path = Path(path)
    digest = sha256(path.read_bytes()).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError("Bank archive differs from the pinned UCI SHA-256")
    with ZipFile(path) as archive, archive.open(
        "bank-additional/bank-additional-full.csv"
    ) as stream:
        raw = pd.read_csv(stream, sep=";")
    if len(raw) != 41188 or not set(FEATURES + EXCLUDED + ("y",)) <= set(raw):
        raise ValueError("Unexpected bank-additional-full schema or row count")
    if not raw["y"].isin(["yes", "no"]).all():
        raise ValueError("Unexpected Bank Marketing target values")
    X = raw.loc[:, FEATURES].copy()
    for column in NUMERIC:
        X[column] = pd.to_numeric(X[column], errors="coerce")
    for column in CATEGORICAL:
        X[column] = X[column].astype("string").fillna("MISSING").astype(str)
    return X, raw["y"].eq("yes").to_numpy(dtype=int), digest


def reliability_bins(y: np.ndarray, probability: np.ndarray) -> list[dict]:
    bins = np.minimum(np.floor(probability * 10).astype(int), 9)
    return [{
        "bin": i, "count": int((bins == i).sum()),
        "mean_predicted": float(probability[bins == i].mean()) if (bins == i).any() else None,
        "observed_rate": float(y[bins == i].mean()) if (bins == i).any() else None,
    } for i in range(10)]


def run_bank_benchmark(
    archive: str | Path = "data/raw/bank/bank-additional.zip", *, seed: int = 264738,
) -> dict:
    X, y, digest = load_bank_archive(archive)
    parts = chronological_partitions(len(y))
    preprocessing = make_preprocessor(X, [name in CATEGORICAL for name in FEATURES])
    models = make_experts(preprocessing, seed, advanced=True)
    fit_time, fitted, selection_brier = {}, {}, {}
    for name, model in models.items():
        start = perf_counter()
        model.fit(X.iloc[parts["fit"]], y[parts["fit"]])
        fit_time[name] = perf_counter() - start
        fitted[name] = model
        selection_brier[name] = float(brier_score_loss(
            y[parts["selection"]], model.predict_proba(X.iloc[parts["selection"]])[:, 1]
        ))
    separate_three_time, separate_models = {}, {}
    for name, model in make_experts(preprocessing, seed, advanced=False).items():
        start = perf_counter()
        model.fit(X.iloc[parts["fit"]], y[parts["fit"]])
        separate_three_time[name] = perf_counter() - start
        separate_models[name] = model

    def score_chosen(model) -> dict:
        calibrator = SplitConformalClassifier.fit(
            model.predict_proba(X.iloc[parts["calibration"]]),
            y[parts["calibration"]], alpha=0.1,
        )
        windows = {}
        for name in ("near_future", "final_future"):
            start = perf_counter()
            proba = model.predict_proba(X.iloc[parts[name]])
            elapsed = perf_counter() - start
            windows[name] = {
                "metrics": evaluate_binary(y[parts[name]], proba, calibrator),
                "reliability_bins": reliability_bins(y[parts[name]], proba[:, 1]),
                "inference_seconds": elapsed,
                "positive_rate_final_audit": float(y[parts[name]].mean()),
            }
        return windows

    results = {}
    for label, choices in (("three_experts", list(models)[:3]),
                           ("five_experts", list(models))):
        winner = min(choices, key=selection_brier.get)
        results[label] = {
            "selected_model": winner, "selection_brier": selection_brier[winner],
            **score_chosen(fitted[winner]),
        }
    separate_winner = results["three_experts"]["selected_model"]
    separate_result = {"selected_model": separate_winner,
                       **score_chosen(separate_models[separate_winner])}
    process = psutil.Process()
    return {
        "benchmark": "UCI Bank Marketing pre-call chronological holdout",
        "source_url": SOURCE_URL, "archive_sha256": digest,
        "target": "y == yes", "prediction_point": "before current telephone call",
        "features": list(FEATURES), "excluded_features": list(EXCLUDED),
        "seed": seed, "splits": {name: {"start": int(idx[0]), "end_exclusive": int(idx[-1]) + 1,
                                  "size": len(idx), "positive_rate_final_audit": float(y[idx].mean())}
                                for name, idx in parts.items()},
        "selection_brier_by_model": selection_brier,
        "fit_seconds_by_model": fit_time,
        "shared_five_fit_seconds": sum(fit_time.values()),
        "separate_three_fit_seconds_by_model": separate_three_time,
        "separate_three_fit_seconds": sum(separate_three_time.values()),
        "peak_process_working_set_kib": process.memory_info().peak_wset // 1024,
        "package_versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "results": results, "separately_realized_three_experts": separate_result,
        "limitations": [
            "Original order is chronological but exact per-record timestamps are unavailable.",
            "Near-future is a later time window, not IID; no customer IDs to enforce entity isolation.",
            "One final future window cannot establish arbitrary temporal generalisation.",
        ],
    }


def save_bank(result: dict, output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path
