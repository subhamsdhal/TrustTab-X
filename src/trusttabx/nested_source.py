"""Nested, entity-safe source-domain validation for the M4.2 method decision."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import psutil
from sklearn.base import clone

from .benchmark import make_experts, make_preprocessor
from .domain_eval import domain_holdout_plan
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

SEED = 264738


@dataclass(frozen=True)
class SourceData:
    X: pd.DataFrame
    y: np.ndarray
    domain: np.ndarray
    entity: np.ndarray | None
    source: np.ndarray
    categorical: frozenset[str]
    protocol: str


def _hash_bucket(value: object) -> int:
    return int.from_bytes(hashlib.sha256(f"{SEED}:{value}".encode()).digest()[:8],
                          "big") % 10000


def load_source(family: str) -> SourceData:
    if family == "acs":
        from .acs_domain import (
            CATEGORICAL,
            bounded_acs_train,
            load_acs_sample,
            split_acs,
        )

        data = load_acs_sample(seed=SEED)
        parts = split_acs(data, SEED)
        source = bounded_acs_train(data, parts["train"], 14000, SEED)
        return SourceData(data.X, data.y, data.state.astype(str), data.household,
                          source, frozenset(CATEGORICAL), "household-disjoint states")
    if family == "diabetes":
        from .natural_domain import (
            CATEGORICAL,
            load_uci_diabetes,
            patient_disjoint_splits,
        )

        data = load_uci_diabetes("data/raw/tableshift/dataset_diabetes.zip")
        parts = patient_disjoint_splits(data, SEED)
        codes = (1, 17, 4, 6)
        candidate = parts["train"][np.isin(data.admission_source[parts["train"]], codes)]
        patients = pd.DataFrame({"entity": data.patient[candidate],
                                 "domain": data.admission_source[candidate]}).groupby(
                                     "entity"
                                 )["domain"].nunique()
        source = candidate[np.isin(data.patient[candidate], patients.index[patients == 1])]
        return SourceData(data.X, data.y, data.admission_source.astype(str), data.patient,
                          source, frozenset(CATEGORICAL), "patient-disjoint admission codes")
    if family == "bank":
        from .bank_temporal import CATEGORICAL, chronological_partitions, load_bank_archive

        X, y, _ = load_bank_archive("data/raw/bank/bank-additional.zip")
        source = chronological_partitions(len(y))["fit"]
        domain = np.full(len(y), "excluded", dtype="<U8")
        for position, block in enumerate(np.array_split(source, 4)):
            domain[block] = f"fit_{position}"
        return SourceData(X, y, domain, None, source, frozenset(CATEGORICAL),
                          "forward-only bank blocks; no customer ID")
    raise ValueError("Unknown source family")


def inner_partitions(data: SourceData, outer_train: np.ndarray,
                     outer_validation: np.ndarray) -> dict[str, np.ndarray]:
    """Return four disjoint inner sets; bank uses chronological boundaries."""
    train = np.asarray(outer_train, dtype=int)
    outer = np.asarray(outer_validation, dtype=int)
    if np.intersect1d(train, outer).size:
        raise ValueError("Outer validation entered inner training")
    if data.entity is None:
        order = np.sort(train)
        domain_order = list(dict.fromkeys(data.domain[order]))
        if not np.array_equal(order, np.arange(order[0], order[-1] + 1)):
            raise ValueError("Bank inner training must be chronologically contiguous")
        first = order[data.domain[order] == domain_order[0]]
        if len(domain_order) == 2:
            second = order[data.domain[order] == domain_order[1]]
            # Use explicit contiguous boundaries to avoid touching future blocks.
            n = len(second)
            result = {"fit": first, "selection": second[:n // 2],
                      "probability_calibration": second[n // 2:3 * n // 4],
                      "threshold_calibration": second[3 * n // 4:]}
        elif len(domain_order) == 3:
            second = order[data.domain[order] == domain_order[1]]
            third = order[data.domain[order] == domain_order[2]]
            result = {"fit": first, "selection": second,
                      "probability_calibration": third[:len(third) // 2],
                      "threshold_calibration": third[len(third) // 2:]}
        else:
            raise ValueError("Bank outer fold needs two or three prior blocks")
    else:
        bucket = np.array([_hash_bucket(value) for value in data.entity[train]])
        result = {
            "fit": train[bucket < 7000],
            "selection": train[(bucket >= 7000) & (bucket < 8500)],
            "probability_calibration": train[(bucket >= 8500) & (bucket < 9250)],
            "threshold_calibration": train[bucket >= 9250],
        }
    if any(len(idx) < 100 for idx in result.values()):
        raise ValueError("Inner partition too small")
    if len(np.unique(np.concatenate([*result.values(), outer]))) != len(train) + len(outer):
        raise ValueError("Nested partitions overlap or omit rows")
    if data.entity is not None:
        all_sets = [set(data.entity[idx]) for idx in [*result.values(), outer]]
        if any(left & right for i, left in enumerate(all_sets)
               for right in all_sets[i + 1:]):
            raise ValueError("Entities cross nested partitions")
    return result


def _inner_folds(data: SourceData, fit: np.ndarray, outer: np.ndarray):
    if data.entity is None:
        ordered = np.sort(fit)
        domain = np.full(len(data.y), "excluded", dtype="<U16")
        for i, block in enumerate(np.array_split(ordered, 4)):
            domain[block] = f"inner_{i}"
        return domain_holdout_plan(domain, fit, outer, ordered=True)
    return domain_holdout_plan(data.domain, fit, outer, groups=data.entity)


def _probability_matrix(models: dict, names: list[str], X: pd.DataFrame) -> np.ndarray:
    return np.column_stack([models[name].predict_proba(X)[:, 1] for name in names])


def _policy_probabilities(matrix: np.ndarray, static_index: int, weights: np.ndarray,
                          routers: dict, names: list[str]) -> dict[str, np.ndarray]:
    full, _ = routed_probability(matrix, names, routers["metadata_router"])
    simple, _ = routed_probability(matrix, names, routers["simple_router"],
                                   simplified=True)
    return {"static": matrix[:, static_index], "uniform": matrix.mean(axis=1),
            "weighted": matrix @ weights, "metadata_router": full,
            "simple_router": simple}


def run_nested_family(family: str) -> dict:
    data = load_source(family)
    excluded = np.setdiff1d(np.arange(len(data.y)), data.source)
    outer_folds = domain_holdout_plan(data.domain, data.source, excluded,
                                      ordered=data.entity is None, groups=data.entity)
    if family == "bank":
        outer_folds = [fold for fold in outer_folds if fold.domain in {"fit_2", "fit_3"}]
    pre = make_preprocessor(data.X, [name in data.categorical for name in data.X.columns])
    experts = make_experts(pre, SEED, advanced=True)
    names = list(experts)
    result = {"family": family, "protocol": data.protocol,
              "source_rows": len(data.source), "outer_domains": {}}
    for outer in outer_folds:
        start = perf_counter()
        parts = inner_partitions(data, outer.train, outer.validation)
        folds = _inner_folds(data, parts["fit"], outer.validation)
        inner_p, inner_y, inner_domains = [], [], []
        inner_fit_seconds = 0.0
        for fold in folds:
            fold_models = {name: clone(model) for name, model in experts.items()}
            for model in fold_models.values():
                tic = perf_counter()
                model.fit(data.X.iloc[fold.train], data.y[fold.train])
                inner_fit_seconds += perf_counter() - tic
            inner_p.append(_probability_matrix(fold_models, names,
                                               data.X.iloc[fold.validation]))
            inner_y.append(data.y[fold.validation])
            inner_domains.append(fold.domain)
        router_p = np.vstack(inner_p)
        router_y = np.concatenate(inner_y)
        routers = {
            "metadata_router": fit_error_router(router_p, router_y, names),
            "simple_router": fit_error_router(router_p, router_y, names, simplified=True),
        }
        fitted = {name: clone(model) for name, model in experts.items()}
        fit_seconds = {}
        for name, model in fitted.items():
            tic = perf_counter()
            model.fit(data.X.iloc[parts["fit"]], data.y[parts["fit"]])
            fit_seconds[name] = perf_counter() - tic
        matrices = {}
        inference_seconds = {}
        for split, idx in {**parts, "outer": outer.validation}.items():
            if split == "fit":
                continue
            tic = perf_counter()
            matrices[split] = _probability_matrix(fitted, names, data.X.iloc[idx])
            inference_seconds[split] = perf_counter() - tic
        selection = matrices["selection"]
        select_brier = np.mean((selection - data.y[parts["selection"], None]) ** 2,
                               axis=0)
        static_index = int(np.argmin(select_brier))
        weights = 1 / np.maximum(select_brier, 1e-8)
        weights /= weights.sum()
        policies = {split: _policy_probabilities(matrix, static_index, weights,
                                                 routers, names)
                    for split, matrix in matrices.items()}
        calibrator = fit_sigmoid_calibrator(
            policies["probability_calibration"]["static"],
            data.y[parts["probability_calibration"]]
        )
        for split_outputs in policies.values():
            split_outputs["calibrated_static"] = calibrated_probability(
                calibrator, split_outputs["static"]
            )
        outer_result = {
            "partition_sizes": {**{name: len(idx) for name, idx in parts.items()},
                                "outer": len(outer.validation)},
            "partition_index_sha256": {name: hashlib.sha256(np.sort(idx).astype(
                "<i8").tobytes()).hexdigest() for name, idx in
                                       {**parts, "outer": outer.validation}.items()},
            "inner_domains": inner_domains, "inner_oof_rows": len(router_y),
            "selected_expert": names[static_index],
            "selection_brier": dict(zip(names, select_brier.tolist())),
            "inner_fit_seconds": inner_fit_seconds,
            "full_fit_seconds": fit_seconds,
            "inference_seconds": inference_seconds,
            "policies": {},
        }
        yy = data.y[outer.validation]
        for name, p in policies["outer"].items():
            cal_p = policies["threshold_calibration"][name]
            item = {"probability": probability_metrics(yy, p), "selective": {}}
            for criterion, (cal_score, test_score) in {
                "confidence": (confidence_uncertainty(cal_p), confidence_uncertainty(p)),
                "disagreement": (disagreement_uncertainty(matrices["threshold_calibration"]),
                                 disagreement_uncertainty(matrices["outer"])),
            }.items():
                item["selective"][criterion] = {
                    "source_threshold": selective_metrics(yy, p, cal_score, test_score),
                    "unlabeled_batch": batch_selective_metrics(yy, p, test_score),
                }
            outer_result["policies"][name] = item
        outer_result["elapsed_seconds"] = perf_counter() - start
        result["outer_domains"][outer.domain] = outer_result
    result["peak_process_working_set_kib"] = psutil.Process().memory_info().peak_wset // 1024
    return result


def save_nested_family(result: dict, output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path
