import numpy as np
import pandas as pd
import pytest

from trusttabx.nested_source import SourceData, inner_partitions
from trusttabx.reliability import batch_selective_metrics, fit_error_router, routed_probability


def test_nested_entity_partitions_exclude_outer_domain_and_keep_groups_disjoint():
    rows = np.arange(2400)
    entity = np.repeat(np.arange(1200), 2)
    domains = np.repeat(["a", "b", "c", "d"], 600)
    data = SourceData(pd.DataFrame({"x": rows}), rows % 2, domains, entity, rows,
                      frozenset(), "fixture")
    outer = rows[domains == "d"]
    train = rows[domains != "d"]
    parts = inner_partitions(data, train, outer)
    assert sum(len(value) for value in parts.values()) == len(train)
    group_sets = [set(entity[idx]) for idx in [*parts.values(), outer]]
    assert all(not left & right for i, left in enumerate(group_sets)
               for right in group_sets[i + 1:])
    with pytest.raises(ValueError, match="Outer validation"):
        inner_partitions(data, rows, outer)


def test_bank_inner_partitions_respect_order():
    rows = np.arange(3000)
    domains = np.repeat(["fit_0", "fit_1", "fit_2"], 1000)
    data = SourceData(pd.DataFrame({"x": rows}), rows % 2, domains, None, rows,
                      frozenset(), "fixture")
    parts = inner_partitions(data, rows[:2000], rows[2000:])
    assert parts["fit"].max() < parts["selection"].min()
    assert parts["selection"].max() < parts["probability_calibration"].min()
    assert parts["probability_calibration"].max() < parts["threshold_calibration"].min()
    assert parts["threshold_calibration"].max() < 2000


def test_routers_and_transductive_cutoff_are_shape_safe():
    p = np.column_stack([np.linspace(0.1, 0.9, 40), np.linspace(0.2, 0.8, 40)])
    y = (np.arange(40) > 20).astype(int)
    for simple in (False, True):
        fitted = fit_error_router(p, y, ["a", "b"], simplified=simple)
        predicted, risk = routed_probability(p, ["a", "b"], fitted, simplified=simple)
        assert predicted.shape == risk.shape == (40,)
    summary = batch_selective_metrics(y, p[:, 0], np.arange(40), coverage=0.8)
    assert summary["acceptance"] == 0.8
