"""Source-only reliability baselines and descriptive selective-risk diagnostics."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def fit_sigmoid_calibrator(probability: np.ndarray, labels: np.ndarray):
    """Learn a one-dimensional sigmoid map on disjoint source calibration rows."""
    p = np.asarray(probability, dtype=float)
    y = np.asarray(labels, dtype=int)
    if len(p) != len(y) or len(np.unique(y)) != 2:
        raise ValueError("Calibration needs aligned probabilities and both classes")
    model = LogisticRegression(C=1.0, solver="lbfgs")
    model.fit(np.log(np.clip(p, 1e-6, 1 - 1e-6) /
                     np.clip(1 - p, 1e-6, 1 - 1e-6)).reshape(-1, 1), y)
    return model


def calibrated_probability(model, probability: np.ndarray) -> np.ndarray:
    p = np.asarray(probability, dtype=float)
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) /
                   np.clip(1 - p, 1e-6, 1 - 1e-6))
    return model.predict_proba(logit.reshape(-1, 1))[:, 1]


def probability_metrics(labels: np.ndarray, probability: np.ndarray) -> dict:
    y = np.asarray(labels, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-15, 1 - 1e-15)
    if len(y) != len(p) or not len(y):
        raise ValueError("Expected aligned, nonempty labels and probabilities")
    return {
        "n": len(y), "positive_rate": float(y.mean()),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
    }


def selective_metrics(labels: np.ndarray, probability: np.ndarray,
                      calibration_score: np.ndarray, target_score: np.ndarray,
                      *, coverage: float = 0.8) -> dict:
    """Fixed source threshold plus target-ranked equal-coverage diagnostic."""
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probability, dtype=float)
    cal = np.asarray(calibration_score, dtype=float)
    score = np.asarray(target_score, dtype=float)
    if not 0 < coverage <= 1 or not len(cal) or len(y) != len(p) or len(y) != len(score):
        raise ValueError("Invalid coverage or score arrays")
    threshold = float(np.quantile(cal, coverage))
    accepted = score <= threshold
    error = (p >= 0.5) != y
    count = max(1, int(np.ceil(coverage * len(y))))
    ranked = np.argsort(score, kind="stable")[:count]
    full_order = np.argsort(score, kind="stable")
    prefix_risk = np.cumsum(error[full_order]) / np.arange(1, len(y) + 1)
    curve = []
    for fraction in np.linspace(0.1, 1.0, 10):
        retained = max(1, int(np.ceil(fraction * len(y))))
        curve.append({"coverage": retained / len(y),
                      "error_risk": float(prefix_risk[retained - 1])})
    coverage_at_fixed_risk = {}
    for risk_limit in (0.1, 0.2):
        valid = np.flatnonzero(prefix_risk <= risk_limit)
        coverage_at_fixed_risk[str(risk_limit)] = (
            float((valid[-1] + 1) / len(y)) if len(valid) else 0.0
        )
    return {
        "source_threshold": threshold,
        "target_coverage": float(accepted.mean()),
        "target_error_risk": float(error[accepted].mean()) if accepted.any() else None,
        "matched_target_coverage": count / len(y),
        "matched_target_error_risk_descriptive": float(error[ranked].mean()),
        "target_ranked_aurc_descriptive": float(prefix_risk.mean()),
        "target_risk_coverage_curve_descriptive": curve,
        "target_coverage_at_fixed_error_risk_descriptive": coverage_at_fixed_risk,
    }


def confidence_uncertainty(probability: np.ndarray) -> np.ndarray:
    return 0.5 - np.abs(np.asarray(probability) - 0.5)


def disagreement_uncertainty(matrix: np.ndarray) -> np.ndarray:
    p = np.asarray(matrix, dtype=float)
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("At least two experts required")
    return p.std(axis=1)
