import numpy as np
import pandas as pd
import pytest

from trusttabx.shift_lab import mcar_missingness, numeric_measurement_noise, prior_shift_indices


def test_mcar_deterministic_and_labels_untouched():
    X = pd.DataFrame({"n": range(30), "c": ["a", "b"] * 15})
    first, rate = mcar_missingness(X, 0.2, 5)
    second, _ = mcar_missingness(X, 0.2, 5)
    assert first.equals(second)
    assert 0 < rate < 1
    assert not X.isna().any().any()


def test_numeric_noise_uses_train_scale_and_preserves_categories():
    train = pd.DataFrame({"n": [1.0, 2.0, 3.0], "c": ["a", "b", "a"]})
    test = pd.DataFrame({"n": [1.0, 2.0], "c": ["z", "z"]})
    altered = numeric_measurement_noise(test, train, [False, True], 0.5, 9)
    assert not altered["n"].equals(test["n"])
    assert altered["c"].equals(test["c"])
    assert test["n"].tolist() == [1.0, 2.0]


def test_prior_cohort_is_disjoint_sample_with_target_prevalence():
    y = np.array([0] * 70 + [1] * 30)
    chosen = prior_shift_indices(y, 0.7, 42)
    assert len(chosen) == len(set(chosen))
    assert np.mean(y[chosen]) == pytest.approx(0.7, abs=0.02)
    assert np.array_equal(chosen, prior_shift_indices(y, 0.7, 42))
