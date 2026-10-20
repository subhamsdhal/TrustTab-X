"""Proper scoring rules plus conformal coverage and abstention statistics."""

from itertools import pairwise

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from .conformal import SplitConformalClassifier


def selective_risk_curve(y_true: ArrayLike, proba: ArrayLike) -> dict:
    """Rank by maximum probability; report error risk at fixed retained fractions."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    if p.shape != (len(y), 2) or len(y) == 0:
        raise ValueError("Expected nonempty binary labels and two probability columns")
    confidence = np.max(p, axis=1)
    errors = (np.argmax(p, axis=1) != y).astype(float)
    order = np.argsort(-confidence, kind="stable")
    cumulative_risk = np.cumsum(errors[order]) / np.arange(1, len(y) + 1)
    points = []
    for target in np.linspace(0.1, 1.0, 10):
        kept = max(1, int(np.ceil(target * len(y))))
        points.append({"coverage": kept / len(y), "error_risk": float(cumulative_risk[kept - 1])})
    return {"aurc": float(np.mean(cumulative_risk)), "points": points}


def expected_calibration_error(y_true: ArrayLike, proba: ArrayLike, bins: int = 10) -> float:
    """Equal-width confidence-bin ECE; descriptive, not a calibrated guarantee."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    confidence = np.max(p, axis=1)
    correct = (np.argmax(p, axis=1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lower, upper in pairwise(edges):
        mask = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        if mask.any():
            ece += mask.mean() * abs(float(correct[mask].mean() - confidence[mask].mean()))
    return float(ece)


def evaluate_binary(
    y_true: ArrayLike,
    proba: ArrayLike,
    calibrator: SplitConformalClassifier,
) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    if p.shape != (len(y), 2):
        raise ValueError("Only binary classification is supported in the starter")
    sets = calibrator.predict_sets(p)
    sizes = sets.sum(axis=1)
    single = sizes == 1
    chosen = np.argmax(sets[single], axis=1) if single.any() else np.array([])
    auc = float(roc_auc_score(y, p[:, 1])) if len(np.unique(y)) == 2 else None
    average_precision = (
        float(average_precision_score(y, p[:, 1])) if len(np.unique(y)) == 2 else None
    )
    selective = selective_risk_curve(y, p)
    return {
        "accuracy": float(accuracy_score(y, np.argmax(p, axis=1))),
        "brier": float(brier_score_loss(y, p[:, 1])),
        "log_loss": float(log_loss(y, p[:, 1], labels=[0, 1])),
        "roc_auc": auc,
        "average_precision": average_precision,
        "ece_10_bins": expected_calibration_error(y, p),
        "aurc_confidence": selective["aurc"],
        "risk_coverage_points": selective["points"],
        "marginal_coverage": float(np.mean(sets[np.arange(len(y)), y])),
        "mean_set_size": float(np.mean(sizes)),
        "abstention_rate": float(np.mean(~single)),
        "singleton_accuracy": (
            float(np.mean(chosen == y[single])) if single.any() else None
        ),
    }
