import numpy as np

from trusttabx.shift import FeatureShiftMonitor


def test_shift_detector_finds_large_feature_change():
    rng = np.random.default_rng(77)
    train = rng.standard_normal((350, 4))
    test = rng.standard_normal((350, 4))
    test[:, 2] += 3.0
    report = FeatureShiftMonitor(train).evaluate(test)
    assert report.shifted is True
    assert report.significant_feature_count >= 1
    assert report.min_p_value < 0.05 / 4
