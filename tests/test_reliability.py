from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import pytest

from trusttabx.college_scorecard import (
    FEATURES,
    institution_partitions,
    load_college_scorecard,
    stable_bucket,
)
from trusttabx.reliability import (
    calibrated_probability,
    disagreement_uncertainty,
    fit_sigmoid_calibrator,
    selective_metrics,
)


def test_college_partitions_are_entity_disjoint_and_deterministic():
    data = pd.DataFrame({"UNITID": np.arange(10000),
                         "CONTROL": np.where(np.arange(10000) < 8000, 1, 3),
                         "label": np.arange(10000) % 2})
    first = institution_partitions(data)
    second = institution_partitions(data)
    assert all(np.array_equal(first[name], second[name]) for name in first)
    assert len(np.unique(np.concatenate(list(first.values())))) == len(data)
    assert set(data.iloc[first["final"]]["CONTROL"]) == {3}
    assert stable_bucket(42) == stable_bucket(42)


def test_calibration_and_selective_risk_use_disjoint_inputs():
    cal_p = np.array([0.1, 0.2, 0.75, 0.9])
    model = fit_sigmoid_calibrator(cal_p, np.array([0, 0, 1, 1]))
    adjusted = calibrated_probability(model, np.array([0.2, 0.8]))
    assert adjusted[0] < adjusted[1]
    result = selective_metrics(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.8, 0.2]),
                               np.array([0.1, 0.2, 0.3, 0.4]),
                               np.array([0.1, 0.2, 0.3, 0.4]), coverage=0.5)
    assert result["target_coverage"] == 0.5
    assert result["target_error_risk"] == 0.0
    assert result["matched_target_error_risk_descriptive"] == 0.0
    with pytest.raises(ValueError):
        disagreement_uncertainty(np.ones((3, 1)))


def test_college_loader_removes_duplicate_entities_and_unobserved_outcomes(tmp_path):
    rows = pd.DataFrame([{**dict.fromkeys(FEATURES, 1), "UNITID": unit,
                          "CONTROL": control, "C150_4": outcome}
                         for unit, control, outcome in [(1, 1, 0.7), (1, 1, 0.7),
                                                        (2, 3, 0.2), (3, 2, "PrivacySuppressed")]])
    archive = tmp_path / "fixture.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as package:
        package.writestr("institution.csv", rows.to_csv(index=False))
    loaded = load_college_scorecard(archive)
    assert loaded["UNITID"].tolist() == [2]
    assert loaded["label"].tolist() == [0]
