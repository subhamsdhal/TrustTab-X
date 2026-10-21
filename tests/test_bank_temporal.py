import numpy as np
import pytest

from trusttabx.bank_temporal import EXCLUDED, FEATURES, chronological_partitions


def test_locked_chronological_boundaries_and_disjoint_order():
    parts = chronological_partitions(41188)
    expected = [(0, 20594), (20594, 26772), (26772, 30891),
                (30891, 35010), (35010, 41188)]
    for idx, (start, end) in zip(parts.values(), expected):
        assert idx[0] == start and idx[-1] + 1 == end
        assert np.array_equal(idx, np.arange(start, end))
    assert np.array_equal(np.concatenate(list(parts.values())), np.arange(41188))


def test_pre_call_schema_excludes_post_contact_information():
    assert "duration" in EXCLUDED and "duration" not in FEATURES
    assert "campaign" in EXCLUDED and "campaign" not in FEATURES
    assert "y" not in FEATURES
    with pytest.raises(ValueError):
        chronological_partitions(20)
