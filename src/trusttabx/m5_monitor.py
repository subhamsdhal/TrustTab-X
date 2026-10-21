"""Reference-based drift and delayed-label monitoring for the ACS demo."""

from __future__ import annotations

from collections import deque
from threading import Lock
from time import perf_counter

import numpy as np
import pandas as pd

from .m5_artifact import CATEGORICAL, NUMERIC_RANGES
from .reliability import probability_metrics


def drift_report(reference: dict, records: list[dict], probabilities: list[float]) -> dict:
    """Diagnostic thresholds: TV>0.2, numeric mean>1 source SD, p-mean>0.1."""
    if not records or len(records) != len(probabilities):
        raise ValueError("Aligned, nonempty records and probabilities required")
    frame = pd.DataFrame(records)
    numeric = {}
    for feature in NUMERIC_RANGES:
        values = pd.to_numeric(frame[feature], errors="coerce")
        ref = reference["numeric"][feature]
        shift = abs(float(values.mean()) - ref["mean"]) / max(ref["sd"], 1e-9)
        numeric[feature] = {"standardized_mean_shift": shift,
                            "missing_rate": float(values.isna().mean()),
                            "alert": shift > 1.0}
    categorical = {}
    for feature in CATEGORICAL:
        source = reference["categorical"][feature]
        observed = frame[feature].astype(str).value_counts(normalize=True).to_dict()
        keys = set(source) | set(observed)
        tv = 0.5 * sum(abs(source.get(key, 0) - observed.get(key, 0)) for key in keys)
        unseen = sorted(set(observed) - set(source))
        categorical[feature] = {"total_variation": tv, "unseen_categories": unseen,
                                "alert": tv > 0.2 or bool(unseen)}
    probability_shift = abs(float(np.mean(probabilities)) - reference["probability_mean"])
    return {"n": len(records), "numeric": numeric, "categorical": categorical,
            "mean_probability": float(np.mean(probabilities)),
            "probability_mean_shift": probability_shift,
            "probability_alert": probability_shift > 0.1,
            "any_feature_alert": any(x["alert"] for x in numeric.values()) or
                                 any(x["alert"] for x in categorical.values()),
            "performance_status": "labels unavailable; Brier, calibration and accuracy unknown",
            "threshold_note": "heuristic demo thresholds; drift does not prove performance loss"}


def delayed_label_report(labels: list[int], probabilities: list[float]) -> dict:
    if len(labels) != len(probabilities) or not labels or not set(labels) <= {0, 1}:
        raise ValueError("Aligned binary delayed labels required")
    values = np.asarray(probabilities, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Probabilities must be finite and within [0, 1]")
    return probability_metrics(np.asarray(labels), np.asarray(probabilities))


class ServiceMonitor:
    """Bounded in-memory operational counters; no raw request retention."""

    def __init__(self, version: str, capacity: int = 1000):
        self.version = version
        self.started = perf_counter()
        self.latencies_ms: deque[float] = deque(maxlen=capacity)
        self.probabilities: deque[float] = deque(maxlen=capacity)
        self.requests = 0
        self.errors = 0
        self.lock = Lock()

    def record(self, probability: float, latency_ms: float) -> None:
        with self.lock:
            self.requests += 1
            self.probabilities.append(probability)
            self.latencies_ms.append(latency_ms)

    def error(self) -> None:
        with self.lock:
            self.errors += 1

    def snapshot(self, reference: dict) -> dict:
        with self.lock:
            latency = np.asarray(self.latencies_ms)
            probabilities = np.asarray(self.probabilities)
            return {"model_version": self.version, "requests": self.requests,
                    "errors": self.errors, "uptime_seconds": perf_counter() - self.started,
                    "latency_p50_ms": float(np.quantile(latency, 0.5)) if len(latency) else None,
                    "latency_p95_ms": float(np.quantile(latency, 0.95)) if len(latency) else None,
                    "mean_probability": float(probabilities.mean()) if len(probabilities) else None,
                    "reference_mean_probability": reference["probability_mean"],
                    "performance_status": "labels unavailable; Brier and calibration unknown"}
