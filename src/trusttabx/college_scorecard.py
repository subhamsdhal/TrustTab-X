"""Locked single-cohort College Scorecard institution-type confirmation run."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import brier_score_loss

from .benchmark import make_experts, make_preprocessor
from .reliability import (
    calibrated_probability,
    confidence_uncertainty,
    disagreement_uncertainty,
    fit_sigmoid_calibrator,
    probability_metrics,
    selective_metrics,
)
from .router_prototype import apply_policies, fit_reliability_models, policy_probabilities

ARCHIVE = Path("data/raw/college_scorecard/Most-Recent-Cohorts-Institution_05192025.zip")
SEED = 264738
FEATURES = (
    "STABBR", "HIGHDEG", "LOCALE", "UGDS", "ADM_RATE", "SATVRMID",
    "SATMTMID", "PCTPELL", "TUITIONFEE_IN", "AVGFACSAL", "PFTFAC", "PPTUG_EF",
)
CATEGORICAL = frozenset({"STABBR", "HIGHDEG", "LOCALE"})


def stable_bucket(unitid: object, seed: int = SEED) -> int:
    payload = f"{seed}:{unitid}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 10000


def load_college_scorecard(archive: str | Path = ARCHIVE) -> pd.DataFrame:
    columns = ["UNITID", "C150_4", "CONTROL", *FEATURES]
    with zipfile.ZipFile(archive) as package:
        csvs = [name for name in package.namelist() if name.lower().endswith(".csv")]
        if len(csvs) != 1:
            raise ValueError("Expected one institution-level CSV")
        with package.open(csvs[0]) as stream:
            data = pd.read_csv(stream, usecols=columns, low_memory=False,
                               na_values=["PrivacySuppressed", "NULL", "NA"])
    for name in ["UNITID", "C150_4", "CONTROL"]:
        data[name] = pd.to_numeric(data[name], errors="coerce")
    data = data.dropna(subset=["UNITID", "C150_4", "CONTROL"])
    data = data[data["CONTROL"].isin([1, 2, 3])]
    data = data.drop_duplicates(subset="UNITID", keep=False).copy()
    data["UNITID"] = data["UNITID"].astype(np.int64)
    data["label"] = (data["C150_4"] > 0.5).astype(int)
    for name in FEATURES:
        if name in CATEGORICAL:
            data[name] = data[name].astype("string").fillna("missing").astype(str)
        else:
            data[name] = pd.to_numeric(data[name], errors="coerce")
    return data.reset_index(drop=True)


def institution_partitions(data: pd.DataFrame) -> dict[str, np.ndarray]:
    source = data["CONTROL"].isin([1, 2]).to_numpy()
    target = np.flatnonzero(data["CONTROL"].eq(3).to_numpy())
    bucket = np.array([stable_bucket(value) for value in data["UNITID"]])
    parts = {
        "fit": np.flatnonzero(source & (bucket < 6000)),
        "selection": np.flatnonzero(source & (bucket >= 6000) & (bucket < 8000)),
        "probability_calibration": np.flatnonzero(source & (bucket >= 8000) & (bucket < 9000)),
        "threshold_calibration": np.flatnonzero(source & (bucket >= 9000)),
        "final": target,
    }
    if min(map(len, parts.values())) < 100:
        raise ValueError("A preregistered partition has fewer than 100 observations")
    if len(np.unique(np.concatenate(list(parts.values())))) != len(data):
        raise ValueError("Partitions overlap or omit rows")
    if any(data.iloc[idx]["label"].nunique() < 2 for idx in parts.values()):
        raise ValueError("A preregistered partition has only one target class")
    return parts


def _matrix(models: dict, names: list[str], X: pd.DataFrame) -> np.ndarray:
    return np.column_stack([models[name].predict_proba(X)[:, 1] for name in names])


def run_college_confirmation(archive: str | Path = ARCHIVE) -> dict:
    data = load_college_scorecard(archive)
    parts = institution_partitions(data)
    X = data.loc[:, FEATURES]
    y = data["label"].to_numpy()
    pre = make_preprocessor(X, [name in CATEGORICAL for name in FEATURES])
    experts = make_experts(pre, SEED, advanced=True)
    names = list(experts)
    fit_seconds = {}
    for name, expert in experts.items():
        start = perf_counter()
        expert.fit(X.iloc[parts["fit"]], y[parts["fit"]])
        fit_seconds[name] = perf_counter() - start
    matrices = {}
    inference_seconds = {}
    for split in ("selection", "probability_calibration", "threshold_calibration", "final"):
        start = perf_counter()
        matrices[split] = _matrix(experts, names, X.iloc[parts[split]])
        inference_seconds[split] = perf_counter() - start
    selection = matrices["selection"]
    router_models, router_meta = fit_reliability_models()
    config = policy_probabilities(selection, names, router_models, y[parts["selection"]],
                                  selection)
    outputs = {}
    risks = {}
    for split in ("probability_calibration", "threshold_calibration", "final"):
        outputs[split], risks[split] = apply_policies(matrices[split], names,
                                                       router_models, config)
    raw_cal = outputs["probability_calibration"]["static"]
    calibrator = fit_sigmoid_calibrator(raw_cal, y[parts["probability_calibration"]])
    for split_outputs in outputs.values():
        split_outputs["calibrated_static"] = calibrated_probability(
            calibrator, split_outputs["static"]
        )
    result = {
        "protocol": "M4.1 preregistered single-cohort College Scorecard confirmation",
        "archive_sha256": hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
        "seed": SEED, "rows": len(data),
        "partitions": {name: {"n": len(idx), "positive_rate": float(y[idx].mean()),
                              "control_counts": {str(int(k)): int(v) for k, v in
                                                 data.iloc[idx]["CONTROL"].value_counts().items()},
                              "unitid_sha256": hashlib.sha256(np.sort(
                                  data.iloc[idx]["UNITID"].to_numpy(dtype="<i8")
                              ).tobytes()).hexdigest()}
                       for name, idx in parts.items()},
        "selection": config, "fit_seconds_by_expert": fit_seconds,
        "inference_seconds_by_split": inference_seconds,
        "router_meta_fit_seconds": router_meta["fit_seconds"],
        "peak_process_working_set_kib": psutil.Process().memory_info().peak_wset // 1024,
        "policies": {},
    }
    cal_matrix = matrices["threshold_calibration"]
    target_matrix = matrices["final"]
    for name, p in outputs["final"].items():
        cal_p = outputs["threshold_calibration"][name]
        item = {"probability": probability_metrics(y[parts["final"]], p),
                "selective": {}}
        uncertainty = {
            "confidence": (confidence_uncertainty(cal_p), confidence_uncertainty(p)),
            "expert_disagreement": (disagreement_uncertainty(cal_matrix),
                                    disagreement_uncertainty(target_matrix)),
        }
        if name == "reliability_router":
            uncertainty["router_estimated_error"] = (
                risks["threshold_calibration"], risks["final"]
            )
        for criterion, (cal_score, test_score) in uncertainty.items():
            item["selective"][criterion] = selective_metrics(
                y[parts["final"]], p, cal_score, test_score
            )
        result["policies"][name] = item
    result["source_selection_brier"] = {
        name: float(brier_score_loss(y[parts["selection"]], selection[:, i]))
        for i, name in enumerate(names)
    }
    return result


def save_college_result(result: dict, output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path
