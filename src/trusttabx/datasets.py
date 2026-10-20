"""Controlled data generation for the pipeline smoke test.

Synthetic samples are NOT evidence of performance on real enterprise datasets.
The shift changes p(X) while preserving p(Y|X), so it is covariate shift.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Split:
    X: NDArray[np.float64]
    y: NDArray[np.int64]


@dataclass(frozen=True)
class DatasetBundle:
    train: Split
    selection: Split
    calibration: Split
    iid_test: Split
    shifted_test: Split


def _sample(
    rng: np.random.Generator,
    n: int,
    *,
    n_features: int,
    shift: float = 0.0,
) -> Split:
    if n <= 0 or n_features < 5:
        raise ValueError("Require positive rows and at least five features")
    X = rng.normal(size=(n, n_features))
    X[:, 0] += shift
    X[:, 2] += 0.5 * shift
    logits = (
        0.9 * X[:, 0]
        - 1.1 * X[:, 1]
        + 0.8 * np.sin(X[:, 2])
        - 0.65 * X[:, 3] * X[:, 4]
        - 0.1
    )
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    y = rng.binomial(1, probability).astype(np.int64)
    return Split(X=X, y=y)


def generate_synthetic_bundle(
    *,
    seed: int = 42,
    n_train: int = 1400,
    n_selection: int = 350,
    n_calibration: int = 350,
    n_test: int = 600,
    n_features: int = 8,
    shift_strength: float = 1.5,
) -> DatasetBundle:
    """Generate independent training/selection/calibration/test samples.

    The selected model never sees calibration labels; the conformal calibrator
    never sees test labels. Shifted samples share the same outcome mechanism.
    """
    if not np.isfinite(shift_strength):
        raise ValueError("shift_strength must be finite")
    rng = np.random.default_rng(seed)
    return DatasetBundle(
        train=_sample(rng, n_train, n_features=n_features),
        selection=_sample(rng, n_selection, n_features=n_features),
        calibration=_sample(rng, n_calibration, n_features=n_features),
        iid_test=_sample(rng, n_test, n_features=n_features),
        shifted_test=_sample(rng, n_test, n_features=n_features, shift=shift_strength),
    )
