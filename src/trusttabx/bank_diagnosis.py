"""Descriptive temporal drift audit for the locked bank benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

from .bank_temporal import (
    CATEGORICAL,
    FEATURES,
    NUMERIC,
    chronological_partitions,
    load_bank_archive,
)
from .benchmark import make_experts, make_preprocessor


def run_bank_diagnosis(archive: str | Path = "data/raw/bank/bank-additional.zip") -> dict:
    X, y, digest = load_bank_archive(archive)
    parts = chronological_partitions(len(y))
    pre = make_preprocessor(X, [name in CATEGORICAL for name in FEATURES])
    model = make_experts(pre, 264738, advanced=False)["hist_gradient_boosting"]
    model.fit(X.iloc[parts["fit"]], y[parts["fit"]])
    source = X.iloc[parts["fit"]]
    result = {"archive_sha256": digest, "model": "hist_gradient_boosting",
              "windows": {}, "feature_drift_from_fit": {}}
    for name, idx in parts.items():
        window = X.iloc[idx]
        p = model.predict_proba(window)[:, 1]
        result["windows"][name] = {
            "n": len(idx), "observed_rate": float(y[idx].mean()),
            "mean_predicted_probability": float(p.mean()),
            "predicted_positive_rate_at_half": float((p >= 0.5).mean()),
            "brier": float(brier_score_loss(y[idx], p)),
            "log_loss": float(log_loss(y[idx], p, labels=[0, 1])),
        }
        numeric = {}
        for column in NUMERIC:
            a, b = source[column], window[column]
            scale = float(a.std())
            numeric[column] = {
                "source_mean": float(a.mean()), "window_mean": float(b.mean()),
                "source_sd": scale,
                "standardized_mean_change": (
                    float((b.mean() - a.mean()) / scale) if scale > 1e-8 else None
                ),
            }
        categorical = {}
        for column in CATEGORICAL:
            a = source[column].value_counts(normalize=True)
            b = window[column].value_counts(normalize=True)
            categories = a.index.union(b.index)
            categorical[column] = float(0.5 * np.abs(
                a.reindex(categories, fill_value=0) - b.reindex(categories, fill_value=0)
            ).sum())
        result["feature_drift_from_fit"][name] = {
            "numeric_mean_change_fit_sd": numeric,
            "categorical_total_variation": categorical,
        }
    return result


def save_bank_diagnosis(result: dict, output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path
