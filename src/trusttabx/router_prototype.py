"""Locked exploratory reliability router and fixed source-selected controls."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .benchmark import make_experts, make_preprocessor
from .conformal import SplitConformalClassifier
from .domain_eval import router_probability_metadata
from .evaluation import evaluate_binary

FAMILIES = ("acs", "diabetes", "bank")


def fit_reliability_models(prediction_dir: str | Path = "data/processed",
                           families: tuple[str, ...] = FAMILIES) -> tuple[dict, dict]:
    """Fit one squared-error regressor per expert on domain-held-out source rows."""
    arrays = {}
    names = None
    if not families or not set(families).issubset(FAMILIES):
        raise ValueError("Expected nonempty subset of known families")
    for family in families:
        with np.load(Path(prediction_dir) / f"{family}_source_crossfit_predictions.npz") as saved:
            arrays[family] = {key: saved[key] for key in saved.files}
        current = [key for key in arrays[family] if key not in
                   {"row_index", "domain", "label", "router_metadata"}]
        if names is None:
            names = current
        elif current != names:
            raise ValueError("Cross-fit expert schemas differ across families")
    assert names is not None
    features, labels, weights = [], [], []
    for family in families:
        record = arrays[family]
        domains, counts = np.unique(record["domain"], return_counts=True)
        domain_weight = {name: 1.0 / (len(families) * len(domains) * count)
                         for name, count in zip(domains, counts)}
        features.append(record["router_metadata"])
        labels.append(np.column_stack([record[name] for name in names]))
        weights.append(np.array([domain_weight[name] for name in record["domain"]]))
    X = np.vstack(features)
    probabilities = np.vstack(labels)
    y = np.concatenate([arrays[family]["label"] for family in families])
    weight = np.concatenate(weights)
    models = {}
    start = perf_counter()
    for column, name in enumerate(names):
        squared_error = (probabilities[:, column] - y) ** 2
        model = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
        model.fit(X, squared_error, ridge__sample_weight=weight * len(weight))
        models[name] = model
    return models, {"expert_names": names, "fit_seconds": perf_counter() - start,
                    "source_crossfit_rows": len(y), "family_count": len(families),
                    "domain_count": {family: len(np.unique(arrays[family]["domain"]))
                                     for family in families}}


def policy_probabilities(probabilities: np.ndarray, names: list[str], models: dict,
                         selection_y: np.ndarray | None = None,
                         selection_probability: np.ndarray | None = None) -> dict:
    """Return deployable policy probabilities and router estimated risk."""
    if selection_y is None or selection_probability is None:
        raise ValueError("Source selection labels and probabilities are required")
    scores = np.array([brier_score_loss(selection_y, selection_probability[:, i])
                       for i in range(len(names))])
    static_index = int(np.argmin(scores))
    weights = 1.0 / np.maximum(scores, 1e-8)
    weights /= weights.sum()
    return {"static_index": static_index, "selection_brier": scores.tolist(),
            "ensemble_weights": weights.tolist()}


def apply_policies(probabilities: np.ndarray, names: list[str], models: dict,
                   config: dict) -> tuple[dict[str, np.ndarray], np.ndarray]:
    metadata = router_probability_metadata(
        {name: probabilities[:, i] for i, name in enumerate(names)}, names
    )
    predicted_error = np.column_stack([
        np.clip(models[name].predict(metadata), 0.0, 1.0) for name in names
    ])
    routed = np.argmin(predicted_error, axis=1)
    confidence = np.argmax(np.abs(probabilities - 0.5), axis=1)
    rows = np.arange(len(probabilities))
    output = {
        "static": probabilities[:, config["static_index"]],
        "uniform_ensemble": probabilities.mean(axis=1),
        "weighted_ensemble": probabilities @ np.asarray(config["ensemble_weights"]),
        "confidence_expert": probabilities[rows, confidence],
        "reliability_router": probabilities[rows, routed],
    }
    return output, predicted_error[rows, routed]


def _family_data(family: str, seed: int):
    if family == "acs":
        from .acs_domain import (
            CATEGORICAL,
            FEATURES,
            bounded_acs_train,
            load_acs_sample,
            split_acs,
        )

        data = load_acs_sample(seed=seed)
        parts = split_acs(data, seed)
        parts["fit"] = bounded_acs_train(data, parts["train"], 14000, seed)
        parts["final"] = parts["ood_test"]
        return data.X, data.y, parts, FEATURES, CATEGORICAL
    if family == "diabetes":
        from .natural_domain import (
            CATEGORICAL,
            FEATURES,
            load_uci_diabetes,
            patient_disjoint_splits,
        )

        data = load_uci_diabetes("data/raw/tableshift/dataset_diabetes.zip")
        parts = patient_disjoint_splits(data, seed)
        train = parts["train"]
        if len(train) > 12000:
            rng = np.random.default_rng(seed)
            patients, counts = np.unique(data.patient[train], return_counts=True)
            chosen, size = [], 0
            for pos in rng.permutation(len(patients)):
                if size + counts[pos] <= 12000:
                    chosen.append(patients[pos])
                    size += counts[pos]
                if size == 12000:
                    break
            train = train[np.isin(data.patient[train], chosen)]
        parts["fit"] = train
        parts["final"] = parts["ood_test"]
        return data.X, data.y, parts, FEATURES, CATEGORICAL
    if family == "bank":
        from .bank_temporal import (
            CATEGORICAL,
            FEATURES,
            chronological_partitions,
            load_bank_archive,
        )

        X, y, _ = load_bank_archive("data/raw/bank/bank-additional.zip")
        parts = chronological_partitions(len(y))
        parts["final"] = parts["final_future"]
        return X, y, parts, FEATURES, CATEGORICAL
    raise ValueError("Unknown family")


def expert_probability_matrix(X, indices: np.ndarray, fitted: dict,
                              names: list[str]) -> np.ndarray:
    return np.column_stack([
        fitted[name].predict_proba(X.iloc[indices])[:, 1] for name in names
    ])


def run_router_prototype(seed: int = 264738) -> dict:
    models, meta = fit_reliability_models()
    names = meta["expert_names"]
    results = {}
    for family in FAMILIES:
        X, y, parts, feature_names, categorical = _family_data(family, seed)
        pre = make_preprocessor(X, [name in categorical for name in feature_names])
        experts = make_experts(pre, seed, advanced=True)
        fitted, fit_time = {}, {}
        for name in names:
            start = perf_counter()
            experts[name].fit(X.iloc[parts["fit"]], y[parts["fit"]])
            fit_time[name] = perf_counter() - start
            fitted[name] = experts[name]
        selection = expert_probability_matrix(X, parts["selection"], fitted, names)
        config = policy_probabilities(selection, names, models, y[parts["selection"]],
                                      selection)
        calibration = expert_probability_matrix(X, parts["calibration"], fitted, names)
        final = expert_probability_matrix(X, parts["final"], fitted, names)
        cal_outputs, cal_risk = apply_policies(calibration, names, models, config)
        final_outputs, final_risk = apply_policies(final, names, models, config)
        family_result = {"fit_seconds_by_model": fit_time, "selection": config,
                         "calibration_rows": len(calibration), "final_rows": len(final),
                         "policies": {}}
        for policy, cal_p in cal_outputs.items():
            test_p = final_outputs[policy]
            cal_two = np.column_stack((1 - cal_p, cal_p))
            test_two = np.column_stack((1 - test_p, test_p))
            conformal = SplitConformalClassifier.fit(
                cal_two, y[parts["calibration"]], alpha=0.1
            )
            if policy == "reliability_router":
                cal_score, test_score = cal_risk, final_risk
            else:
                cal_score, test_score = 0.5 - np.abs(cal_p - 0.5), 0.5 - np.abs(test_p - 0.5)
            threshold = float(np.quantile(cal_score, 0.8))
            kept = test_score <= threshold
            predicted = (test_p >= 0.5).astype(int)
            family_result["policies"][policy] = {
                "metrics": evaluate_binary(y[parts["final"]], test_two, conformal),
                "selection_threshold": threshold,
                "calibration_decision_coverage": float(np.mean(cal_score <= threshold)),
                "final_decision_coverage": float(kept.mean()),
                "final_selected_error_risk": (
                    float(np.mean(predicted[kept] != y[parts["final"]][kept]))
                    if kept.any() else None
                ),
            }
        results[family] = family_result
    return {"protocol": "locked exploratory cross-family metadata router",
            "router_fit": meta, "results": results,
            "limitations": [
                "Only three dataset families; no independent family-level test of router learning.",
                "Final domains are evaluated once; no target-label tuning or shift adaptation.",
                "Router uses probability metadata only; explicit feature-shift input remains future work.",
            ]}


def save_router(result: dict, output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path
