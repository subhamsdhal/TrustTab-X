import numpy as np
import pytest

from trusttabx.evaluation import expected_calibration_error, selective_risk_curve


def test_selective_risk_curve_perfect_and_all_wrong():
    y = np.array([0, 1] * 10)
    perfect = np.column_stack([1 - y, y]).astype(float)
    wrong = perfect[:, ::-1]
    assert selective_risk_curve(y, perfect)["aurc"] == 0
    assert selective_risk_curve(y, wrong)["aurc"] == 1
    assert selective_risk_curve(y, perfect)["points"][-1]["coverage"] == 1


def test_ece_is_zero_for_confident_correct_predictions():
    y = np.array([0, 1, 0, 1])
    p = np.column_stack([1 - y, y]).astype(float)
    assert expected_calibration_error(y, p) == pytest.approx(0)
