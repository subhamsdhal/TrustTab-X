import numpy as np
import pandas as pd

from trusttabx.natural_domain import DomainData, patient_disjoint_splits


def test_natural_splits_are_patient_disjoint_and_keep_ood_for_test():
    patient = np.arange(3000)
    ood = patient >= 1800
    data = DomainData(pd.DataFrame(index=patient), np.zeros(3000), patient, ood, "fixture")
    parts = patient_disjoint_splits(data)
    assert all(len(indices) > 0 for indices in parts.values())
    for name, indices in parts.items():
        assert np.all(ood[indices] == name.startswith("ood_"))
    sets = [set(patient[indices]) for indices in parts.values()]
    assert all(not left & right for i, left in enumerate(sets) for right in sets[i + 1:])


def test_cross_domain_patients_are_purged_from_source():
    patient = np.concatenate([np.arange(2000), np.arange(1900, 3200)])
    ood = np.concatenate([np.zeros(2000, dtype=bool), np.ones(1300, dtype=bool)])
    data = DomainData(pd.DataFrame(index=np.arange(len(patient))), np.zeros(len(patient)),
                      patient, ood, "fixture")
    parts = patient_disjoint_splits(data)
    source = np.concatenate([indices for name, indices in parts.items() if not name.startswith("ood_")])
    assert not set(patient[source]) & set(patient[ood])
