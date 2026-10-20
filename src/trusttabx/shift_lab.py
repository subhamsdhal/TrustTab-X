"""Pre-specified stress tests on a fixed, already selected predictor.

Feature interventions are measurement corruption, not necessarily pure
covariate shift. The prior-shift cohort uses labels for *evaluation design*
only; neither the model nor thresholds are fitted with those labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .conformal import SplitConformalClassifier
from .evaluation import evaluate_binary


def mcar_missingness(X: pd.DataFrame, rate: float, seed: int) -> tuple[pd.DataFrame, float]:
    if not 0 <= rate <= 1:
        raise ValueError("Missingness rate must be in [0, 1]")
    mask = np.random.default_rng(seed).random(X.shape) < rate
    return X.mask(mask), float(mask.mean())


def numeric_measurement_noise(
    X_test: pd.DataFrame, X_train: pd.DataFrame,
    categorical: list[bool], strength: float, seed: int,
) -> pd.DataFrame:
    if strength < 0 or not np.isfinite(strength):
        raise ValueError("Noise strength must be finite and nonnegative")
    changed = X_test.copy()
    rng = np.random.default_rng(seed)
    for name, is_categorical in zip(X_test.columns, categorical):
        if not is_categorical:
            scale = pd.to_numeric(X_train[name], errors="coerce").std()
            if pd.notna(scale) and scale > 0:
                changed[name] = pd.to_numeric(X_test[name], errors="coerce") + rng.normal(
                    0, strength * scale, len(X_test)
                )
    return changed


def prior_shift_indices(y: np.ndarray, positive_fraction: float, seed: int) -> np.ndarray:
    """Downsample a labelled test cohort to a pre-specified class prevalence."""
    if not 0 < positive_fraction < 1:
        raise ValueError("positive_fraction must lie strictly between 0 and 1")
    y = np.asarray(y)
    negative, positive = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    if not len(negative) or not len(positive):
        raise ValueError("Both classes are needed for a prior-shift cohort")
    size = min(len(positive) / positive_fraction, len(negative) / (1 - positive_fraction))
    n_positive = max(1, min(len(positive), round(size * positive_fraction)))
    n_negative = max(1, min(len(negative), round(size * (1 - positive_fraction))))
    rng = np.random.default_rng(seed)
    selected = np.concatenate([
        rng.choice(positive, n_positive, replace=False),
        rng.choice(negative, n_negative, replace=False),
    ])
    return rng.permutation(selected)


def run_shift_lab(
    model, calibrator: SplitConformalClassifier,
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_test: np.ndarray,
    categorical: list[bool], seed: int,
) -> dict:
    missing, observed_rate = mcar_missingness(X_test, 0.2, seed + 20)
    reports = {
        "mcar_20pct": {
            "mechanism": "independent feature-cell missingness; labels unchanged",
            "realized_missing_fraction": observed_rate,
            "metrics": evaluate_binary(y_test, model.predict_proba(missing), calibrator),
        }
    }
    if not all(categorical):
        noisy = numeric_measurement_noise(X_test, X_train, categorical, 0.5, seed + 30)
        reports["numeric_noise_0_5sd"] = {
            "mechanism": "numeric measurement noise at 0.5 train-standard-deviation; labels unchanged",
            "metrics": evaluate_binary(y_test, model.predict_proba(noisy), calibrator),
        }
    chosen = prior_shift_indices(y_test, 0.7, seed + 40)
    reports["prior_70pct_positive"] = {
        "mechanism": "test-label-stratified downsampling for evaluation only; no adaptation",
        "cohort_size": len(chosen),
        "realized_positive_fraction": float(np.mean(y_test[chosen])),
        "metrics": evaluate_binary(
            y_test[chosen], model.predict_proba(X_test.iloc[chosen]), calibrator
        ),
    }
    return reports
