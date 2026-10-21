import numpy as np
import pytest

from trusttabx.shopper_confirmation import month_partitions


def test_shopper_month_holdout_and_october_calibration_are_disjoint():
    month = np.repeat(["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep",
                       "Oct", "Nov", "Dec"], 220)
    y = np.tile([0, 1], len(month) // 2)
    parts = month_partitions(month, y)
    assert set(month[parts["fit"]]) == {"Feb", "Mar", "May", "June", "Jul", "Aug"}
    assert set(month[parts["nov"]]) == {"Nov"}
    assert set(month[parts["dec"]]) == {"Dec"}
    assert parts["probability_calibration"].max() < parts["threshold_calibration"].min()
    assert len(np.unique(np.concatenate(list(parts.values())))) == len(month)


def test_shopper_confirmation_stops_if_target_month_is_too_small():
    month = np.repeat(["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep",
                       "Oct", "Nov", "Dec"], 200)
    y = np.tile([0, 1], len(month) // 2)
    month[-150:] = "Nov"
    with pytest.raises(ValueError, match="fewer than 100"):
        month_partitions(month, y)
