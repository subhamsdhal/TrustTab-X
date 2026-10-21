"""Execute source-only geographic or forward-temporal expert cross-fitting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from .benchmark import make_experts, make_preprocessor
from .domain_eval import crossfit_experts, domain_holdout_plan, router_probability_metadata


def run_source_crossfit(dataset: str, *, seed: int = 264738) -> tuple[dict, dict]:
    if dataset == "acs":
        from .acs_domain import (
            CATEGORICAL,
            FEATURES,
            bounded_acs_train,
            load_acs_sample,
            split_acs,
        )

        data = load_acs_sample(seed=seed)
        parts = split_acs(data, seed)
        source = bounded_acs_train(data, parts["train"], 14000, seed)
        excluded = np.setdiff1d(np.arange(len(data.y)), source)
        domains = data.state
        folds = domain_holdout_plan(domains, source, excluded, groups=data.household)
        X, y = data.X, data.y
        categorical = CATEGORICAL
        feature_names = FEATURES
        protocol = "leave-one-source-state-out; all other partitions excluded"
    elif dataset == "diabetes":
        from .natural_domain import (
            CATEGORICAL,
            FEATURES,
            load_uci_diabetes,
            patient_disjoint_splits,
        )

        data = load_uci_diabetes("data/raw/tableshift/dataset_diabetes.zip")
        parts = patient_disjoint_splits(data, seed)
        codes = (1, 17, 4, 6)  # fixed from source counts before outcome inspection
        candidate = parts["train"][np.isin(data.admission_source[parts["train"]], codes)]
        counts = pd.DataFrame({"patient": data.patient[candidate],
                               "code": data.admission_source[candidate]}).groupby(
                                   "patient"
                               )["code"].nunique()
        single_code_patients = counts.index[counts == 1]
        source = candidate[np.isin(data.patient[candidate], single_code_patients)]
        excluded = np.setdiff1d(np.arange(len(data.y)), source)
        domains = data.admission_source.astype(str)
        folds = domain_holdout_plan(domains, source, excluded, groups=data.patient)
        X, y = data.X, data.y
        categorical = CATEGORICAL
        feature_names = FEATURES
        protocol = ("leave-one-source-admission-code-out (1, 17, 4, 6), "
                    "purging patients who cross these codes; all other rows excluded")
    elif dataset == "bank":
        from .bank_temporal import (
            CATEGORICAL,
            FEATURES,
            chronological_partitions,
            load_bank_archive,
        )

        X, y, _ = load_bank_archive("data/raw/bank/bank-additional.zip")
        parts = chronological_partitions(len(y))
        source = parts["fit"]
        excluded = np.setdiff1d(np.arange(len(y)), source)
        domains = np.full(len(y), "excluded", dtype="<U8")
        for i, block in enumerate(np.array_split(source, 4)):
            domains[block] = f"fit_{i}"
        folds = domain_holdout_plan(domains, source, excluded, ordered=True)
        categorical = CATEGORICAL
        feature_names = FEATURES
        protocol = "forward-only four source-fit blocks; later windows excluded"
    else:
        raise ValueError("dataset must be 'acs', 'diabetes' or 'bank'")
    preprocessing = make_preprocessor(X, [name in categorical for name in feature_names])
    experts = make_experts(preprocessing, seed, advanced=True)
    predictions, fold_metrics = crossfit_experts(X, y, folds, experts)
    names = list(experts)
    metadata = router_probability_metadata(predictions, names)
    predictions["router_metadata"] = metadata
    process = psutil.Process()
    summary = {
        "dataset": dataset, "seed": seed, "protocol": protocol,
        "source_rows": len(source), "excluded_rows": len(excluded),
        "fold_count": len(folds), "expert_names": names,
        "trained_model_count": len(folds) * len(names),
        "router_metadata_columns": (
            [f"probability_{name}" for name in names]
            + [f"binary_entropy_{name}" for name in names]
            + ["mean_probability", "std_probability"]
        ),
        "fold_metrics": fold_metrics,
        "fit_seconds_total": sum(
            item["fit_seconds"] for fold in fold_metrics.values()
            for item in fold["experts"].values()
        ),
        "peak_process_working_set_kib": process.memory_info().peak_wset // 1024,
        "limitations": [
            "Predictions and labels are source-fold only; no final test rows are present.",
            ("These cross-fitted rows support router development but do not themselves "
             "provide nested unbiased router evaluation."),
        ],
    }
    if np.intersect1d(predictions["row_index"], excluded).size:
        raise ValueError("Final or reserved rows entered cross-fitted predictions")
    return predictions, summary


def save_source_crossfit(predictions: dict, summary: dict, output: str | Path,
                         prediction_output: str | Path) -> None:
    path, pred_path = Path(output), Path(prediction_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred_path, **predictions)
    summary["prediction_artifact"] = str(pred_path)
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
