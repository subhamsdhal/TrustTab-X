import numpy as np
import pytest

from trusttabx.datasets import generate_synthetic_bundle


def test_bundle_has_four_separate_purposes_and_two_tests():
    b = generate_synthetic_bundle(n_train=100, n_selection=50, n_calibration=50, n_test=80)
    assert b.train.X.shape == (100, 8)
    assert b.selection.X.shape == (50, 8)
    assert b.calibration.y.shape == (50,)
    assert b.iid_test.X.shape == b.shifted_test.X.shape == (80, 8)
    assert not np.array_equal(b.train.X[:50], b.selection.X)
    assert b.shifted_test.X[:, 0].mean() > b.iid_test.X[:, 0].mean()


def test_dataset_is_reproducible():
    a = generate_synthetic_bundle(seed=2)
    b = generate_synthetic_bundle(seed=2)
    assert np.array_equal(a.train.X, b.train.X)
    assert np.array_equal(a.shifted_test.y, b.shifted_test.y)


def test_invalid_generator_arguments():
    with pytest.raises(ValueError):
        generate_synthetic_bundle(n_train=0)
