"""Split conformal prediction sets with a finite-sample corrected threshold.

Coverage is marginal under exchangeability and a held-out calibration split.
Do not claim finite-sample coverage under arbitrary distribution shift.
"""

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_probabilities(probabilities: ArrayLike) -> NDArray[np.float64]:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("Expected [n_rows, n_classes] probabilities")
    if len(probabilities) < 1 or not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities must be finite and nonempty")
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Probabilities must be in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Each row must sum to 1")
    return probabilities


@dataclass(frozen=True)
class SplitConformalClassifier:
    """Classification prediction sets based on nonconformity 1 - p(y|x)."""

    threshold: float
    alpha: float
    n_calibration: int
    n_classes: int

    @classmethod
    def fit(cls, probabilities: ArrayLike, labels: ArrayLike, alpha: float = 0.1):
        p = _validate_probabilities(probabilities)
        y = np.asarray(labels)
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if y.shape != (len(p),) or not np.issubdtype(y.dtype, np.integer):
            raise ValueError("labels must be a one-dimensional array of class indices")
        if np.any(y < 0) or np.any(y >= p.shape[1]):
            raise ValueError("labels must be valid class indices")
        scores = 1.0 - p[np.arange(len(y)), y]
        k = math.ceil((len(y) + 1) * (1.0 - alpha))
        # For very small calibration sets, an infinite threshold yields all classes.
        threshold = float(np.sort(scores)[k - 1]) if k <= len(y) else float("inf")
        return cls(threshold, alpha, len(y), p.shape[1])

    def predict_sets(self, probabilities: ArrayLike) -> NDArray[np.bool_]:
        p = _validate_probabilities(probabilities)
        if p.shape[1] != self.n_classes:
            raise ValueError("Prediction class count does not match calibration")
        return (1.0 - p) <= (self.threshold + 1e-12)
