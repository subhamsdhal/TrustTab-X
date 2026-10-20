"""Patient-disjoint TableShift-derived diabetes readmission benchmark.

The domain is TableShift's admission_source_id == 7. Splits are deliberately
not claimed to be exact TableShift splits: patient grouping prevents encounter
leakage across source and target cohorts.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import pandas as pd
import psutil
import sklearn
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GroupShuffleSplit

from .benchmark import make_experts, make_preprocessor
from .conformal import SplitConformalClassifier
from .evaluation import evaluate_binary

SOURCE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"
)
EXPECTED_SHA256 = "66d9820c3bd5fd2abf7957bda2d9714b2de1e7bba1ac1f1ff8fa85e83c22cb33"

# Fixed before model fitting. These encounter-level fields avoid identifiers,
# discharge disposition and post-outcome variables. Other TableShift fields are
# omitted for a bounded CPU pilot, so this is not its original feature set.
NUMERIC = (
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
)
CATEGORICAL = (
    "gender", "age", "admission_type_id", "payer_code", "medical_specialty",
    "max_glu_serum", "A1Cresult", "change", "diabetesMed",
)
FEATURES = NUMERIC + CATEGORICAL


@dataclass(frozen=True)
class DomainData:
    X: pd.DataFrame
    y: np.ndarray
    patient: np.ndarray
    ood: np.ndarray
    source_sha256: str


def load_uci_diabetes(path: str | Path) -> DomainData:
    path = Path(path)
    digest = sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("UCI diabetes archive SHA-256 does not match the pinned file")
    with ZipFile(path) as archive, archive.open("dataset_diabetes/diabetic_data.csv") as stream:
        raw = pd.read_csv(stream, na_values="?", low_memory=False)
    raw = raw.dropna(subset=["race"]).reset_index(drop=True)
    if len(raw) != 99493:
        raise ValueError("Unexpected row count after TableShift's race filter")
    if not set(FEATURES + ("patient_nbr", "admission_source_id", "readmitted")) <= set(raw):
        raise ValueError("Expected UCI schema is missing fields")
    X = raw.loc[:, FEATURES].copy()
    for column in NUMERIC:
        X[column] = pd.to_numeric(X[column], errors="coerce")
    for column in CATEGORICAL:
        X[column] = X[column].astype("string").fillna("MISSING").astype(str)
    return DomainData(
        X=X, y=(raw["readmitted"] != "NO").to_numpy(dtype=int),
        patient=raw["patient_nbr"].to_numpy(),
        ood=(raw["admission_source_id"] == 7).to_numpy(), source_sha256=digest,
    )


def patient_disjoint_splits(data: DomainData, seed: int = 264738) -> dict[str, np.ndarray]:
    """Retain OOD rows; remove source patients appearing in OOD, then group split."""
    ood_idx = np.flatnonzero(data.ood)
    ood_patients = set(data.patient[ood_idx])
    source_idx = np.array([
        i for i in np.flatnonzero(~data.ood) if data.patient[i] not in ood_patients
    ])
    if len(source_idx) < 1000 or len(ood_idx) < 1000:
        raise ValueError("Insufficient patient-disjoint source and target domains")
    # GroupShuffleSplit does not use labels, and all inner splits remain source-only.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    fit_pos, hold_pos = next(splitter.split(source_idx, groups=data.patient[source_idx]))
    train = source_idx[fit_pos]
    hold = source_idx[hold_pos]
    half = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + 1)
    select_pos, calibrate_pos = next(half.split(hold, groups=data.patient[hold]))
    ood_split = GroupShuffleSplit(n_splits=1, test_size=0.9, random_state=seed)
    ood_val_pos, ood_test_pos = next(
        ood_split.split(ood_idx, groups=data.patient[ood_idx])
    )
    result = {
        "train": train, "selection": hold[select_pos], "calibration": hold[calibrate_pos],
        "ood_validation_unused": ood_idx[ood_val_pos], "ood_test": ood_idx[ood_test_pos],
    }
    id_split = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed + 2)
    source_fit, id_test_pos = next(
        id_split.split(result["train"], groups=data.patient[result["train"]])
    )
    result["id_test"] = result["train"][id_test_pos]
    result["train"] = result["train"][source_fit]
    groups = [set(data.patient[indices]) for indices in result.values()]
    if any(groups[i] & groups[j] for i in range(len(groups)) for j in range(i + 1, len(groups))):
        raise ValueError("Patient groups overlap between partitions")
    return result


def _metrics(y: np.ndarray, p: np.ndarray, calibrator: SplitConformalClassifier) -> dict:
    return evaluate_binary(y, p, calibrator)


def run_natural_benchmark(
    archive: str | Path = "data/raw/tableshift/dataset_diabetes.zip",
    *, seed: int = 264738, max_fit_rows: int = 12000,
) -> dict:
    """Run fixed classical experts; OOD labels are read only by final metrics."""
    data = load_uci_diabetes(archive)
    splits = patient_disjoint_splits(data, seed)
    train = splits["train"]
    if len(train) > max_fit_rows:
        rng = np.random.default_rng(seed)
        # Choose whole patients without repeated scans of all encounter rows.
        patients, counts = np.unique(data.patient[train], return_counts=True)
        order = rng.permutation(len(patients))
        admitted = []
        size = 0
        for idx in order:
            if size + counts[idx] <= max_fit_rows:
                admitted.append(patients[idx])
                size += counts[idx]
            if size == max_fit_rows:
                break
        train = train[np.isin(data.patient[train], admitted)]
    parts = {**splits, "train": train}
    preprocessor = make_preprocessor(data.X, [column in CATEGORICAL for column in FEATURES])
    experts = make_experts(preprocessor, seed, advanced=True)
    fitted = {}
    fit_seconds = {}
    select_scores = {}
    selection_probabilities = {}
    for name, model in experts.items():
        start = perf_counter()
        model.fit(data.X.iloc[train], data.y[train])
        fit_seconds[name] = perf_counter() - start
        fitted[name] = model
        selection_probabilities[name] = model.predict_proba(
            data.X.iloc[parts["selection"]]
        )[:, 1]
        select_scores[name] = float(brier_score_loss(
            data.y[parts["selection"]], selection_probabilities[name]
        ))
    diversity = {}
    names = list(experts)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a, b = selection_probabilities[left], selection_probabilities[right]
            diversity[f"{left}__{right}"] = {
                "mean_absolute_probability_difference": float(np.mean(np.abs(a - b))),
                "pearson_probability_correlation": float(np.corrcoef(a, b)[0, 1]),
            }
    results = {}
    for candidate_set, names in {
        "three_experts": list(experts)[:3], "five_experts": list(experts),
    }.items():
        winner = min(names, key=select_scores.get)
        model = fitted[winner]
        calibration = model.predict_proba(data.X.iloc[parts["calibration"]])
        calibrator = SplitConformalClassifier.fit(
            calibration, data.y[parts["calibration"]], alpha=0.1
        )
        result = {"selected_model": winner, "selection_brier": select_scores[winner]}
        for split in ("id_test", "ood_test"):
            start = perf_counter()
            predictions = model.predict_proba(data.X.iloc[parts[split]])
            elapsed = perf_counter() - start
            result[split] = {
                "metrics": _metrics(data.y[parts[split]], predictions, calibrator),
                "inference_seconds": elapsed,
            }
        results[candidate_set] = result
    process = psutil.Process()
    return {
        "benchmark": "TableShift-derived diabetes admission-source domain holdout",
        "upstream_commit": "fca9429814703a07e3902d005d46563a207b7f0a",
        "source_url": SOURCE_URL, "source_sha256": data.source_sha256,
        "seed": seed, "feature_columns": list(FEATURES),
        "target_definition": "1 if readmitted != NO (any recorded readmission)",
        "domain_definition": "admission_source_id == 7 (emergency room) is OOD",
        "split_policy": "patient-disjoint TableShift-derived; not exact TableShift row splits",
        "split_sizes": {key: len(value) for key, value in parts.items()},
        "split_sha256": {
            key: sha256(np.asarray(value, dtype="<i8").tobytes()).hexdigest()
            for key, value in parts.items()
        },
        "fit_seconds_by_model": fit_seconds,
        "selection_brier_by_model": select_scores,
        "selection_prediction_diversity": diversity,
        "model_count": {"three_experts": 3, "five_experts": 5},
        "shared_fit_budget_model_count": 5,
        "selection_trials": {"three_experts": 3, "five_experts": 5},
        "fit_seconds_by_configuration": {
            "three_experts": sum(fit_seconds.values()),
            "five_experts": sum(fit_seconds.values()),
        },
        "package_versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        },
        "process_rss_kib_at_completion": process.memory_info().rss // 1024,
        "peak_process_working_set_kib": process.memory_info().peak_wset // 1024,
        "results": results,
        "limitations": [
            "The original TableShift package and exact row splits were not run.",
            "Patient grouping and a bounded fixed feature subset change the benchmark protocol.",
            "OOD labels are evaluation-only; OOD patient IDs were used by the curator to prevent leakage.",
            ("Both selections share the same five fitted experts and thus equal realized fit cost; "
             "the three-expert decision ignores two already-trained models."),
            "One domain pair is insufficient to assess a trained router's domain generalization.",
        ],
    }


def save_natural(result: dict, output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path
