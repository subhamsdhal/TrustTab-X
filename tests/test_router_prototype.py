import numpy as np
import pytest

from trusttabx.router_prototype import apply_policies, policy_probabilities


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
