"""Feature-wise two-sample KS shift monitor with a Bonferroni correction.

The p-values do not license arbitrary repeated monitoring without adjustment.
Do not use this detector as evidence that OOD predictions are reliable.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import ks_2samp


@dataclass(frozen=True)
class ShiftReport:
    shifted: bool
    min_p_value: float
    significant_feature_count: int
    number_of_features: int
    familywise_alpha: float


class FeatureShiftMonitor:
    def __init__(self, reference: ArrayLike, alpha: float = 0.05):
        reference = np.asarray(reference, dtype=float)
        if reference.ndim != 2 or reference.shape[0] < 2 or reference.shape[1] < 1:
            raise ValueError("Expected at least two reference rows and one feature")
        if not np.all(np.isfinite(reference)):
            raise ValueError("Reference features must be finite")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.reference = reference.copy()
        self.alpha = float(alpha)

    def evaluate(self, candidate: ArrayLike) -> ShiftReport:
        candidate = np.asarray(candidate, dtype=float)
        if candidate.ndim != 2 or candidate.shape[0] < 2:
            raise ValueError("Need at least two candidate rows")
        if candidate.shape[1] != self.reference.shape[1] or not np.all(np.isfinite(candidate)):
            raise ValueError("Invalid candidate feature shape or values")
        p = np.array([
            ks_2samp(self.reference[:, j], candidate[:, j], method="auto").pvalue
            for j in range(candidate.shape[1])
        ])
        count = int(np.sum(p < self.alpha / len(p)))
        return ShiftReport(
            shifted=count > 0,
            min_p_value=float(np.min(p)),
            significant_feature_count=count,
            number_of_features=int(len(p)),
            familywise_alpha=self.alpha,
        )
