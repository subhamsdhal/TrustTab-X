"""Bounded 2018 ACSIncome geographic-domain pilot using public Folktables data."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from hashlib import file_digest, sha256
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import psutil
import sklearn
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GroupShuffleSplit

from .benchmark import make_experts, make_preprocessor
from .conformal import SplitConformalClassifier
from .evaluation import evaluate_binary

YEAR = 2018
SOURCE_STATES = ("PA", "NJ", "NY", "OR", "NM", "KS", "AL")
TARGET_STATES = ("CT", "MA", "ME", "NH", "RI", "VT")  # Census division 01
NUMERIC = ("AGEP", "WKHP")
CATEGORICAL = ("COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "SEX", "RAC1P")
FEATURES = NUMERIC + CATEGORICAL


@dataclass(frozen=True)
class ACSData:
    X: pd.DataFrame
    y: np.ndarray
    state: np.ndarray
    household: np.ndarray
    sample_manifest: dict


def load_acs_sample(
    cache_dir: str | Path = "data/raw/acs2018", *, seed: int = 264738,
    source_per_state: int = 6000, target_per_state: int = 3000,
) -> ACSData:
    """Filter the official adult population, then sample without looking at labels."""
    try:
        from folktables import ACSDataSource, adult_filter
    except ImportError as exc:
        raise RuntimeError("Install the optional ACS dependency: pip install -e '.[acs]'") from exc

    source = ACSDataSource(
        survey_year=str(YEAR), horizon="1-Year", survey="person", root_dir=str(cache_dir)
    )
    frames = []
    manifest = {}
    for state in SOURCE_STATES + TARGET_STATES:
        raw = source.get_data(states=[state], download=True)
        if not set(FEATURES + ("PINCP", "SERIALNO", "DIVISION", "ST")) <= set(raw):
            raise ValueError(f"ACS schema missing required columns in {state}")
        if state in TARGET_STATES and not raw["DIVISION"].eq(1).all():
            raise ValueError(f"Expected New England division 01 for {state}")
        if state in SOURCE_STATES and raw["DIVISION"].eq(1).any():
            raise ValueError(f"Unexpected source state in division 01: {state}")
        state_code = int(raw["ST"].iloc[0])
        source_file = Path(cache_dir) / str(YEAR) / "1-Year" / f"psam_p{state_code:02d}.csv"
        with source_file.open("rb") as stream:
            source_hash = file_digest(stream, "sha256").hexdigest()
        eligible = adult_filter(raw)
        cap = source_per_state if state in SOURCE_STATES else target_per_state
        state_seed = seed + int.from_bytes(state.encode("ascii"), "big")
        state_rng = np.random.default_rng(state_seed)
        chosen = np.sort(state_rng.choice(
            len(eligible), size=min(cap, len(eligible)), replace=False
        ))
        frame = eligible.iloc[chosen][list(FEATURES) + ["PINCP", "SERIALNO"]].copy()
        frame["state_name"] = state
        frames.append(frame)
        manifest[state] = {
            "raw_rows": len(raw), "eligible_rows": len(eligible),
            "sampled_rows": len(frame), "sampling_seed": state_seed,
            "source_file": source_file.name, "source_file_sha256": source_hash,
            "sample_index_sha256": sha256(chosen.astype("<i8").tobytes()).hexdigest(),
        }
        del raw, eligible
    sampled = pd.concat(frames, ignore_index=True)
    X = sampled[list(FEATURES)].copy()
    for column in NUMERIC:
        X[column] = pd.to_numeric(X[column], errors="coerce")
    for column in CATEGORICAL:
        X[column] = X[column].astype("string").fillna("MISSING").astype(str)
    return ACSData(
        X=X, y=(sampled["PINCP"] > 56000).to_numpy(dtype=int),
        state=sampled["state_name"].to_numpy(),
        household=(sampled["state_name"] + ":" + sampled["SERIALNO"].astype(str)).to_numpy(),
        sample_manifest=manifest,
    )


def split_acs(data: ACSData, seed: int = 264738) -> dict[str, np.ndarray]:
    """Source-only household-disjoint fit/selection/calibration/ID partitions."""
    result: dict[str, list[np.ndarray]] = {key: [] for key in
                                          ("train", "selection", "calibration", "id_test")}
    for state in SOURCE_STATES:
        idx = np.flatnonzero(data.state == state)
        first = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
        train_pos, held_pos = next(first.split(idx, groups=data.household[idx]))
        result["train"].append(idx[train_pos])
        held = idx[held_pos]
        second = GroupShuffleSplit(n_splits=1, test_size=2 / 3, random_state=seed + 1)
        selection_pos, rest_pos = next(second.split(held, groups=data.household[held]))
        result["selection"].append(held[selection_pos])
        rest = held[rest_pos]
        third = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + 2)
        cal_pos, test_pos = next(third.split(rest, groups=data.household[rest]))
        result["calibration"].append(rest[cal_pos])
        result["id_test"].append(rest[test_pos])
    parts = {name: np.concatenate(chunks) for name, chunks in result.items()}
    parts["ood_test"] = np.flatnonzero(np.isin(data.state, TARGET_STATES))
    household_sets = [set(data.household[idx]) for idx in parts.values()]
    if any(a & b for i, a in enumerate(household_sets) for b in household_sets[i + 1:]):
        raise ValueError("ACS household overlap between partitions")
    if set(data.state[parts["ood_test"]]) != set(TARGET_STATES):
        raise ValueError("Missing target state")
    return parts


def run_acs_benchmark(
    cache_dir: str | Path = "data/raw/acs2018", *, seed: int = 264738,
    max_fit_rows: int = 12000, source_per_state: int = 6000,
    target_per_state: int = 3000,
) -> dict:
    data = load_acs_sample(cache_dir, seed=seed, source_per_state=source_per_state,
                           target_per_state=target_per_state)
    parts = split_acs(data, seed)
    train = parts["train"]
    if len(train) > max_fit_rows:
        rng = np.random.default_rng(seed + 3)
        chosen = []
        # Group-safe bounded sampling in each source state.
        for state in SOURCE_STATES:
            state_idx = train[data.state[train] == state]
            groups, counts = np.unique(data.household[state_idx], return_counts=True)
            budget = max_fit_rows // len(SOURCE_STATES)
            selected = []
            size = 0
            for pos in rng.permutation(len(groups)):
                if size + counts[pos] <= budget:
                    selected.append(groups[pos])
                    size += counts[pos]
                if size == budget:
                    break
            chosen.append(state_idx[np.isin(data.household[state_idx], selected)])
        train = np.concatenate(chosen)
    parts["train"] = train
    preprocessing = make_preprocessor(data.X, [c in CATEGORICAL for c in FEATURES])
    model_pool = make_experts(preprocessing, seed, advanced=True)
    fitted, fit_seconds, select_brier, select_probability = {}, {}, {}, {}
    for name, model in model_pool.items():
        start = perf_counter()
        model.fit(data.X.iloc[train], data.y[train])
        fit_seconds[name] = perf_counter() - start
        fitted[name] = model
        select_probability[name] = model.predict_proba(data.X.iloc[parts["selection"]])[:, 1]
        select_brier[name] = float(brier_score_loss(
            data.y[parts["selection"]], select_probability[name]
        ))
    separate_three_fit_seconds = {}
    separately_fitted = {}
    for name, model in make_experts(preprocessing, seed, advanced=False).items():
        start = perf_counter()
        model.fit(data.X.iloc[train], data.y[train])
        separate_three_fit_seconds[name] = perf_counter() - start
        separately_fitted[name] = model
    results = {}
    for label, names in (("three_experts", list(model_pool)[:3]),
                         ("five_experts", list(model_pool))):
        winner = min(names, key=select_brier.get)
        model = fitted[winner]
        cal = SplitConformalClassifier.fit(
            model.predict_proba(data.X.iloc[parts["calibration"]]),
            data.y[parts["calibration"]], alpha=0.1,
        )
        outcome = {"selected_model": winner, "selection_brier": select_brier[winner]}
        for split in ("id_test", "ood_test"):
            start = perf_counter()
            proba = model.predict_proba(data.X.iloc[parts[split]])
            elapsed = perf_counter() - start
            outcome[split] = {
                "metrics": evaluate_binary(data.y[parts[split]], proba, cal),
                "inference_seconds": elapsed,
                "positive_rate": float(data.y[parts[split]].mean()),
            }
        outcome["ood_by_state"] = {}
        for state in TARGET_STATES:
            idx = parts["ood_test"][data.state[parts["ood_test"]] == state]
            outcome["ood_by_state"][state] = evaluate_binary(
                data.y[idx], model.predict_proba(data.X.iloc[idx]), cal
            )
        results[label] = outcome
    separate_winner = results["three_experts"]["selected_model"]
    separate_model = separately_fitted[separate_winner]
    separate_cal = SplitConformalClassifier.fit(
        separate_model.predict_proba(data.X.iloc[parts["calibration"]]),
        data.y[parts["calibration"]], alpha=0.1,
    )
    separate_results = {"selected_model": separate_winner}
    for split in ("id_test", "ood_test"):
        start = perf_counter()
        proba = separate_model.predict_proba(data.X.iloc[parts[split]])
        separate_results[split] = {
            "metrics": evaluate_binary(data.y[parts[split]], proba, separate_cal),
            "inference_seconds": perf_counter() - start,
        }
    process = psutil.Process()
    for state in SOURCE_STATES + TARGET_STATES:
        indices = np.flatnonzero(data.state == state)
        data.sample_manifest[state]["positive_rate_final_audit"] = float(data.y[indices].mean())
    diversity = {}
    names = list(model_pool)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a, b = select_probability[left], select_probability[right]
            diversity[f"{left}__{right}"] = {
                "mean_absolute_difference": float(np.mean(np.abs(a - b))),
                "pearson_correlation": float(np.corrcoef(a, b)[0, 1]),
            }
    return {
        "benchmark": "TableShift-inspired ACSIncome geographic pilot",
        "survey": "2018 ACS 1-year PUMS person survey", "target": "PINCP > 56000 USD",
        "source_states": list(SOURCE_STATES), "ood_states": list(TARGET_STATES),
        "features": list(FEATURES), "seed": seed, "source_per_state": source_per_state,
        "target_per_state": target_per_state, "max_fit_rows": max_fit_rows,
        "sample_manifest": data.sample_manifest,
        "split_sizes": {key: len(idx) for key, idx in parts.items()},
        "split_sha256": {key: sha256(idx.astype("<i8").tobytes()).hexdigest()
                         for key, idx in parts.items()},
        "fit_seconds_by_model": fit_seconds,
        "shared_five_model_fit_seconds": sum(fit_seconds.values()),
        "separately_realized_three_model_fit_seconds": sum(separate_three_fit_seconds.values()),
        "separate_three_fit_seconds_by_model": separate_three_fit_seconds,
        "selection_trials": {"three_experts": 3, "five_experts": 5},
        "selection_brier_by_model": select_brier,
        "selection_prediction_diversity": diversity,
        "peak_process_working_set_kib": process.memory_info().peak_wset // 1024,
        "package_versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "sklearn": sklearn.__version__,
                             "folktables": version("folktables")},
        "results": results,
        "separately_realized_three_experts": separate_results,
        "limitations": [
            "Sampled nine states; not the full official TableShift dataset or row splits.",
            "Three-model configuration was separately refit for timing; metrics use shared fits.",
            "US Census person records may share household; partitions are household-disjoint.",
            ("Seven source states span five census divisions, but only one ACS OOD division "
             "and one hospital domain pair are available for final evaluation."),
        ],
    }


def save_acs(result: dict, output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path
