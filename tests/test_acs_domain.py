import numpy as np
import pandas as pd

from trusttabx.acs_domain import SOURCE_STATES, TARGET_STATES, ACSData, split_acs


def test_acs_splits_are_household_disjoint_and_ood_untouched():
    states = np.array([state for state in SOURCE_STATES + TARGET_STATES for _ in range(120)])
    households = np.array([f"{state}:{i // 2}" for state in SOURCE_STATES + TARGET_STATES
                           for i in range(120)])
    data = ACSData(pd.DataFrame(index=np.arange(len(states))), np.zeros(len(states)),
                   states, households, {})
    parts = split_acs(data)
    assert set(states[parts["ood_test"]]) == set(TARGET_STATES)
    assert set(states[parts["train"]]) == set(SOURCE_STATES)
    groups = [set(households[idx]) for idx in parts.values()]
    assert all(not a & b for i, a in enumerate(groups) for b in groups[i + 1:])
    assert sum(map(len, parts.values())) == len(states)
