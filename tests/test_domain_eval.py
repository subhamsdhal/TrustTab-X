import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from trusttabx.domain_eval import (
    crossfit_experts,
    domain_holdout_plan,
    router_probability_metadata,
)


def test_geographic_plan_never_uses_final_domain_and_groups_do_not_cross():
    domains = np.array(["a"] * 20 + ["b"] * 20 + ["c"] * 20 + ["final"] * 20)
    rows = np.arange(60)
    folds = domain_holdout_plan(domains, rows, np.arange(60, 80), groups=np.arange(80))
    assert len(folds) == 3
    assert all(not np.intersect1d(f.train, f.validation).size for f in folds)
    assert all(not np.intersect1d(f.train, np.arange(60, 80)).size for f in folds)
    with pytest.raises(ValueError, match="overlap"):
        domain_holdout_plan(domains, np.arange(65), np.arange(60, 80))


def test_temporal_plan_trains_only_on_earlier_domains():
    domains = np.array(["t0"] * 10 + ["t1"] * 10 + ["t2"] * 10 + ["future"] * 10)
    folds = domain_holdout_plan(domains, np.arange(30), np.arange(30, 40), ordered=True)
    assert [(len(f.train), len(f.validation)) for f in folds] == [(10, 10), (20, 10)]
    assert all(f.train.max() < f.validation.min() for f in folds)
    shuffled = domains.copy()
    shuffled[[0, 15]] = shuffled[[15, 0]]
    with pytest.raises(ValueError, match="contiguous"):
        domain_holdout_plan(shuffled, np.arange(30), np.arange(30, 40), ordered=True)


def test_crossfit_predictions_only_cover_heldout_source_rows():
    domains = np.array(["a"] * 20 + ["b"] * 20 + ["c"] * 20 + ["final"] * 20)
    folds = domain_holdout_plan(domains, np.arange(60), np.arange(60, 80))
    X = pd.DataFrame({"x": np.arange(80) % 2})
    y = np.arange(80) % 2
    output, metrics = crossfit_experts(X, y, folds, {"dummy": DummyClassifier()})
    assert set(output["row_index"]) == set(range(60))
    assert set(metrics) == {"a", "b", "c"}
    metadata = router_probability_metadata(output, ["dummy"])
    assert metadata.shape == (60, 4)
