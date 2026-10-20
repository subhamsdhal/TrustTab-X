"""Offline fixtures: no OpenML network request is made in CI."""

import numpy as np
import pandas as pd
import pytest

from trusttabx.benchmark import evaluate_arrays, make_preprocessor, partition_outer_train


@pytest.fixture
def small_task():
    rng = np.random.default_rng(17)
    X = pd.DataFrame({
        "amount": rng.normal(size=120),
        "kind": np.where(np.arange(120) % 3 == 0, "A", "B"),
    })
    X.loc[100:, "kind"] = "test_only"
    X.loc[100:, "amount"] += 1000
    y = np.array([0, 1] * 60)
    return X, y, np.arange(100), np.arange(100, 120)


def test_inner_partitions_disjoint_and_reproducible(small_task):
    _, y, outer_train, outer_test = small_task
    first = partition_outer_train(outer_train, y, 42)
    second = partition_outer_train(outer_train, y, 42)
    assert all(np.array_equal(first[k], second[k]) for k in first)
    assert set(np.concatenate(list(first.values()))) == set(outer_train)
    assert not set(np.concatenate(list(first.values()))) & set(outer_test)
    assert {k: len(v) for k, v in first.items()} == {
        "train": 60, "selection": 20, "calibration": 20,
    }


def test_preprocessing_fits_only_training_rows(small_task):
    X, _, outer_train, outer_test = small_task
    preprocessor = make_preprocessor(X, [False, True])
    preprocessor.fit(X.iloc[outer_train])
    numeric = preprocessor.named_transformers_["numeric"]
    categories = preprocessor.named_transformers_["categorical"]["encode"].categories_[0]
    assert numeric["scale"].mean_[0] == pytest.approx(X.iloc[outer_train]["amount"].mean())
    assert "test_only" not in categories
    transformed = preprocessor.transform(X.iloc[outer_test])
    assert np.isfinite(transformed).all()


def test_offline_full_task_uses_official_test_only_for_final_metrics(small_task):
    X, y, outer_train, outer_test = small_task
    result = evaluate_arrays(X, y, [False, True], outer_train, outer_test, seed=9)
    assert result["split_sizes"] == {
        "train": 60, "selection": 20, "calibration": 20, "test": 20,
    }
    assert result["selected_model"] == min(result["selection_brier"],
                                           key=result["selection_brier"].get)
    assert 0 <= result["test_metrics_selected_model_only"]["brier"] <= 1
    with pytest.raises(ValueError, match="partition"):
        evaluate_arrays(X, y, [False, True], outer_train, outer_train[:20])
