import numpy as np
import pytest

from trusttabx.conformal import SplitConformalClassifier


def test_finite_sample_rank_threshold():
    p_true = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    p = np.column_stack([p_true, 1 - p_true])
    cal = SplitConformalClassifier.fit(p, np.zeros(9, dtype=int), alpha=0.2)
    # ceil((9 + 1) * 0.8) = 8; 8th sorted score = 0.8
    assert cal.threshold == pytest.approx(0.8)


def test_tiny_calibration_set_returns_full_prediction_sets():
    cal = SplitConformalClassifier.fit([[0.6, 0.4], [0.4, 0.6]], [0, 1], alpha=0.1)
    assert np.isinf(cal.threshold)
    assert cal.predict_sets([[0.99, 0.01]]).tolist() == [[True, True]]


def test_invalid_probability_shape_or_labels_rejected():
    with pytest.raises(ValueError):
        SplitConformalClassifier.fit([[0.8, 0.8]], [0])
    with pytest.raises(ValueError):
        SplitConformalClassifier.fit([[0.8, 0.2]], [2])
