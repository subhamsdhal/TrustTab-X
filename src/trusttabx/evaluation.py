"""Proper scoring rules plus conformal coverage and abstention statistics."""

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from .conformal import SplitConformalClassifier


def evaluate_binary(
    y_true: ArrayLike,
    proba: ArrayLike,
    calibrator: SplitConformalClassifier,
) -> dict[str, float | None]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    if p.shape != (len(y), 2):
        raise ValueError("Only binary classification is supported in the starter")
    sets = calibrator.predict_sets(p)
    sizes = sets.sum(axis=1)
    single = sizes == 1
    chosen = np.argmax(sets[single], axis=1) if single.any() else np.array([])
    auc = float(roc_auc_score(y, p[:, 1])) if len(np.unique(y)) == 2 else None
    return {
        "accuracy": float(accuracy_score(y, np.argmax(p, axis=1))),
        "brier": float(brier_score_loss(y, p[:, 1])),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "roc_auc": auc,
        "marginal_coverage": float(np.mean(sets[np.arange(len(y)), y])),
        "mean_set_size": float(np.mean(sizes)),
        "abstention_rate": float(np.mean(~single)),
        "singleton_accuracy": (
            float(np.mean(chosen == y[single])) if single.any() else None
        ),
    }
