import numpy as np
import pytest

from trusttabx.router_prototype import (
    apply_policies,
    fit_reliability_models,
    policy_probabilities,
)


class ConstantRisk:
    def __init__(self, risk):
        self.risk = risk

    def predict(self, X):
        return np.full(len(X), self.risk)


def test_locked_policies_use_source_selection_and_deployment_probabilities():
    names = ["a", "b"]
    selection = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
    y = np.array([0, 1, 0, 1])
    models = {"a": ConstantRisk(0.2), "b": ConstantRisk(0.1)}
    config = policy_probabilities(selection, names, models, y, selection)
    assert config["static_index"] == 0
    test = np.array([[0.2, 0.7], [0.6, 0.4]])
    policies, risk = apply_policies(test, names, models, config)
    assert np.allclose(policies["static"], [0.2, 0.6])
    assert np.allclose(policies["reliability_router"], [0.7, 0.4])
    assert np.allclose(risk, 0.1)
    with pytest.raises(ValueError):
        policy_probabilities(test, names, models)


def test_family_holdout_does_not_load_heldout_labels(tmp_path):
    # The third family is intentionally absent: fitting must read only the named families.
    for family in ("acs", "bank"):
        p = np.linspace(0.1, 0.9, 20)
        np.savez_compressed(tmp_path / f"{family}_source_crossfit_predictions.npz",
                            row_index=np.arange(20), domain=np.repeat(["a", "b"], 10),
                            label=(p > 0.5).astype(int),
                            router_metadata=np.column_stack([p, p]),
                            expert_a=p, expert_b=1 - p)
    models, metadata = fit_reliability_models(tmp_path, ("acs", "bank"))
    assert metadata["family_count"] == 2
    assert metadata["source_crossfit_rows"] == 40
    assert set(models) == {"expert_a", "expert_b"}
