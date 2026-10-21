"""One-shot, preregistered UCI shopper month-shift confirmation."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import psutil

from .benchmark import make_experts, make_preprocessor
from .reliability import (
    batch_selective_metrics,
    calibrated_probability,
    confidence_uncertainty,
    disagreement_uncertainty,
    fit_error_router,
    fit_sigmoid_calibrator,
    probability_metrics,
    routed_probability,
    selective_metrics,
)
from .router_diagnosis import load_source_predictions
from .router_prototype import FAMILIES, fit_reliability_models, policy_probabilities

ARCHIVE = Path("data/raw/uci_shoppers/online_shoppers_468.zip")
ARCHIVE_SHA256 = "2972e6184d3ad7beaaa831d9fc2b059dc3ee29df69d1ec593c466a5cd8485d14"
SEED = 264738
FEATURES = (
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "SpecialDay", "OperatingSystems", "Browser", "Region", "TrafficType",
    "VisitorType", "Weekend",
)
CATEGORICAL = frozenset({"OperatingSystems", "Browser", "Region", "TrafficType",
                         "VisitorType"})
FIT_MONTHS = frozenset({"Feb", "Mar", "May", "June", "Jul", "Aug"})


def load_shopper(archive: str | Path = ARCHIVE) -> tuple[pd.DataFrame, str]:
    path = Path(archive)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError("UCI shopper archive differs from preregistered SHA256")
    with zipfile.ZipFile(path) as package:
        names = [name for name in package.namelist() if name.endswith(".csv")]
        if names != ["online_shoppers_intention.csv"]:
            raise ValueError("Unexpected UCI shopper archive schema")
        with package.open(names[0]) as stream:
            data = pd.read_csv(stream)
    if len(data) != 12330 or not {*FEATURES, "Month", "Revenue"} <= set(data):
        raise ValueError("Unexpected UCI shopper rows or columns")
    if not data["Revenue"].isin([True, False]).all():
        raise ValueError("Unexpected Revenue labels")
    for name in FEATURES:
        if name in CATEGORICAL:
            data[name] = data[name].astype("string").fillna("missing").astype(str)
        elif name == "Weekend":
            data[name] = data[name].astype(int)
        else:
            data[name] = pd.to_numeric(data[name], errors="coerce")
    return data, digest


def month_partitions(month: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    m = np.asarray(month).astype(str)
    oct_rows = np.flatnonzero(m == "Oct")
    midpoint = len(oct_rows) // 2
    parts = {
        "fit": np.flatnonzero(np.isin(m, list(FIT_MONTHS))),
        "selection": np.flatnonzero(m == "Sep"),
        "probability_calibration": oct_rows[:midpoint],
        "threshold_calibration": oct_rows[midpoint:],
        "nov": np.flatnonzero(m == "Nov"),
        "dec": np.flatnonzero(m == "Dec"),
    }
    if min(map(len, parts.values())) < 100:
        raise ValueError("A preregistered month partition has fewer than 100 sessions")
    if any(len(np.unique(labels[idx])) < 2 for idx in parts.values()):
        raise ValueError("A preregistered month partition contains one class")
    if len(np.unique(np.concatenate(list(parts.values())))) != len(m):
        raise ValueError("Months overlap or omit rows")
    return parts


def fit_simple_crossfamily_router(directory: str | Path = "data/processed") -> tuple[dict, dict]:
    records = {family: load_source_predictions(family, directory) for family in FAMILIES}
    names = [name for name in records[FAMILIES[0]]
             if name not in {"row_index", "domain", "label", "router_metadata"}]
    matrices, labels, weights = [], [], []
    for record in records.values():
        matrix = np.column_stack([record[name] for name in names])
        domains, counts = np.unique(record["domain"], return_counts=True)
        domain_weights = {domain: 1 / (len(FAMILIES) * len(domains) * count)
                          for domain, count in zip(domains, counts)}
        matrices.append(matrix)
        labels.append(record["label"])
        weights.append(np.array([domain_weights[d] for d in record["domain"]]))
    models = fit_error_router(np.vstack(matrices), np.concatenate(labels), names,
                              simplified=True, sample_weight=np.concatenate(weights))
    return models, {"expert_names": names, "source_rows": sum(map(len, labels)),
                    "equal_family_and_domain_weighting": True}


def _matrix(models: dict, names: list[str], X: pd.DataFrame) -> np.ndarray:
    return np.column_stack([models[name].predict_proba(X)[:, 1] for name in names])


def run_shopper_confirmation(archive: str | Path = ARCHIVE) -> dict:
    data, digest = load_shopper(archive)
    y = data["Revenue"].astype(int).to_numpy()
    parts = month_partitions(data["Month"].to_numpy(), y)
    X = data.loc[:, FEATURES]
    pre = make_preprocessor(X, [name in CATEGORICAL for name in FEATURES])
    experts = make_experts(pre, SEED, advanced=True)
    names = list(experts)
    fit_seconds = {}
    for name, model in experts.items():
        tic = perf_counter()
        model.fit(X.iloc[parts["fit"]], y[parts["fit"]])
        fit_seconds[name] = perf_counter() - tic
    matrix = {}
    inference_seconds = {}
    for split, indices in parts.items():
        if split == "fit":
            continue
        tic = perf_counter()
        matrix[split] = _matrix(experts, names, X.iloc[indices])
        inference_seconds[split] = perf_counter() - tic
    original, original_meta = fit_reliability_models()
    simple, simple_meta = fit_simple_crossfamily_router()
    config = policy_probabilities(matrix["selection"], names, original,
                                  y[parts["selection"]], matrix["selection"])
    outputs = {}
    for split, p in matrix.items():
        metadata_p, _ = routed_probability(p, names, original)
        simple_p, _ = routed_probability(p, names, simple, simplified=True)
        outputs[split] = {
            "static": p[:, config["static_index"]],
            "uniform": p.mean(axis=1),
            "weighted": p @ np.asarray(config["ensemble_weights"]),
            "metadata_router": metadata_p,
            "simple_router": simple_p,
        }
    calibrator = fit_sigmoid_calibrator(
        outputs["probability_calibration"]["static"],
        y[parts["probability_calibration"]]
    )
    for split_outputs in outputs.values():
        split_outputs["calibrated_static"] = calibrated_probability(
            calibrator, split_outputs["static"]
        )
    result = {
        "protocol": "M4.2 preregistered UCI shopper month-shift confirmation",
        "archive_sha256": digest, "seed": SEED, "rows": len(data),
        "partition": {name: {"n": len(idx), "positive_rate": float(y[idx].mean()),
                             "row_index_sha256": hashlib.sha256(np.asarray(idx, dtype="<i8")
                                                            .tobytes()).hexdigest()}
                      for name, idx in parts.items()},
        "selected_expert": names[config["static_index"]],
        "selection_brier": dict(zip(names, config["selection_brier"])),
        "fit_seconds_by_expert": fit_seconds,
        "inference_seconds_by_split": inference_seconds,
        "original_router_source": original_meta,
        "simple_router_source": simple_meta,
        "peak_process_working_set_kib": psutil.Process().memory_info().peak_wset // 1024,
        "environments": {},
    }
    for split in ("threshold_calibration", "nov", "dec"):
        environment = {"n": len(parts[split]), "policies": {}}
        for name, p in outputs[split].items():
            item = {"probability": probability_metrics(y[parts[split]], p)}
            if split != "threshold_calibration":
                item["selective"] = {}
                for criterion, (cal_score, test_score) in {
                    "confidence": (
                        confidence_uncertainty(outputs["threshold_calibration"][name]),
                        confidence_uncertainty(p),
                    ),
                    "disagreement": (
                        disagreement_uncertainty(matrix["threshold_calibration"]),
                        disagreement_uncertainty(matrix[split]),
                    ),
                }.items():
                    item["selective"][criterion] = {
                        "source_threshold": selective_metrics(y[parts[split]], p,
                                                               cal_score, test_score),
                        "unlabeled_batch": batch_selective_metrics(y[parts[split]], p,
                                                                    test_score),
                    }
            environment["policies"][name] = item
        result["environments"][split] = environment
    return result


def save_shopper_result(result: dict, output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path
