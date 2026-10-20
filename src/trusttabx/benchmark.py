"""Small, leakage-aware OpenML-CC18 binary benchmark.

Only the selected expert is evaluated on the official outer test fold. The
single-fold result is a pilot, not a full CC18 cross-validation estimate.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .conformal import SplitConformalClassifier
from .evaluation import evaluate_binary
from .shift_lab import run_shift_lab


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    dataset_id: int
    dataset_version: int
    dataset_name: str
    target: str
    md5_checksum: str
    licence: str


TASKS = {
    3: TaskSpec(3, 3, 1, "kr-vs-kp", "class", "ad6eb32b7492524d4382a40e23cdbb8e", "Public"),
    31: TaskSpec(31, 31, 1, "credit-g", "class", "9a475053fed0c26ee95cd4525e50074c", "Public"),
    10101: TaskSpec(10101, 1464, 1, "blood-transfusion-service-center", "Class", "c3242468edab8c2e7a907674122dc851", "Public"),
}


def index_hash(indices: np.ndarray) -> str:
    """Hash ordered integer row indices to identify the exact split."""
    return sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def partition_outer_train(
    outer_train: np.ndarray, y: np.ndarray, seed: int
) -> dict[str, np.ndarray]:
    """Use 60/20/20 train/selection/calibration within the official train fold."""
    outer_train = np.asarray(outer_train, dtype=int)
    fit, rest = train_test_split(
        outer_train, test_size=0.4, random_state=seed, stratify=y[outer_train]
    )
    selection, calibration = train_test_split(
        rest, test_size=0.5, random_state=seed + 1, stratify=y[rest]
    )
    parts = {"train": fit, "selection": selection, "calibration": calibration}
    if len(set(np.concatenate(list(parts.values())).tolist())) != len(outer_train):
        raise ValueError("Inner partitions overlap or omit rows")
    return parts


def make_preprocessor(X: pd.DataFrame, categorical: list[bool]) -> ColumnTransformer:
    if len(categorical) != X.shape[1]:
        raise ValueError("Categorical indicator does not match the feature schema")
    num = [name for name, is_cat in zip(X.columns, categorical) if not is_cat]
    cat = [name for name, is_cat in zip(X.columns, categorical) if is_cat]
    blocks = []
    if num:
        blocks.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), num))
    if cat:
        blocks.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", max_categories=20, sparse_output=False)),
        ]), cat))
    if not blocks:
        raise ValueError("Dataset has no usable features")
    return ColumnTransformer(blocks, remainder="drop", sparse_threshold=0)


def make_experts(
    preprocessor: ColumnTransformer, seed: int, *, advanced: bool = False
) -> dict[str, Pipeline]:
    experts = {
        "logistic_regression": Pipeline([
            ("preprocess", clone(preprocessor)),
            ("model", LogisticRegression(max_iter=1000, random_state=seed)),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("preprocess", clone(preprocessor)),
            ("model", HistGradientBoostingClassifier(
                max_iter=90, learning_rate=0.07, max_leaf_nodes=15, random_state=seed
            )),
        ]),
        "random_forest": Pipeline([
            ("preprocess", clone(preprocessor)),
            ("model", RandomForestClassifier(
                n_estimators=100, min_samples_leaf=4, n_jobs=1, random_state=seed
            )),
        ]),
    }
    if advanced:
        try:
            from catboost import CatBoostClassifier
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("Advanced experts require pip install -e '.[benchmarks]'") from exc
        experts["xgboost"] = Pipeline([
            ("preprocess", clone(preprocessor)),
            ("model", XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                n_jobs=1, random_state=seed,
            )),
        ])
        experts["catboost_ohe"] = Pipeline([
            ("preprocess", clone(preprocessor)),
            ("model", CatBoostClassifier(
                iterations=150, depth=5, learning_rate=0.05, loss_function="Logloss",
                random_seed=seed, thread_count=1, verbose=False, allow_writing_files=False,
            )),
        ])
    return experts


def evaluate_arrays(
    X: pd.DataFrame,
    labels: np.ndarray,
    categorical: list[bool],
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    *,
    seed: int = 42,
    alpha: float = 0.1,
    max_rows: int = 5000,
    advanced: bool = False,
    shift_lab: bool = False,
    fold: int = 0,
) -> dict:
    """Evaluate one official fold, with no fitting on outer test rows."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if not isinstance(X, pd.DataFrame) or len(X) != len(labels):
        raise ValueError("X must be a DataFrame aligned with labels")
    if len(X) > max_rows:
        raise ValueError(f"Dataset has {len(X)} rows, exceeding max_rows={max_rows}")
    outer_train = np.asarray(outer_train, dtype=int)
    outer_test = np.asarray(outer_test, dtype=int)
    if (len(outer_train) + len(outer_test) != len(X)
            or set(outer_train) & set(outer_test)
            or len(set(np.concatenate([outer_train, outer_test]).tolist())) != len(X)):
        raise ValueError("Official train and test indices must partition all rows")
    classes = np.unique(labels[outer_train])
    if len(classes) != 2 or not set(np.unique(labels)).issubset(set(classes)):
        raise ValueError("Only binary classification with both classes in outer train is supported")
    y = np.searchsorted(classes, labels)
    parts = partition_outer_train(outer_train, y, seed)
    preprocessor = make_preprocessor(X, categorical)
    experts = make_experts(preprocessor, seed, advanced=advanced)
    selection_scores: dict[str, float] = {}
    fit_seconds: dict[str, float] = {}
    for name, expert in experts.items():
        start = perf_counter()
        expert.fit(X.iloc[parts["train"]], y[parts["train"]])
        fit_seconds[name] = round(perf_counter() - start, 6)
        probabilities = expert.predict_proba(X.iloc[parts["selection"]])[:, 1]
        selection_scores[name] = float(brier_score_loss(y[parts["selection"]], probabilities))
    selected = min(selection_scores, key=selection_scores.get)
    chosen = experts[selected]
    calibration_probabilities = chosen.predict_proba(X.iloc[parts["calibration"]])
    calibrator = SplitConformalClassifier.fit(
        calibration_probabilities, y[parts["calibration"]], alpha=alpha
    )
    start = perf_counter()
    test_probabilities = chosen.predict_proba(X.iloc[outer_test])
    inference_seconds = round(perf_counter() - start, 6)
    metrics = evaluate_binary(y[outer_test], test_probabilities, calibrator)
    result = {
        "seed": seed,
        "outer_fold": fold,
        "advanced_experts": advanced,
        "shift_lab_enabled": shift_lab,
        "alpha": alpha,
        "selected_model": selected,
        "selection_brier": selection_scores,
        "test_metrics_selected_model_only": metrics,
        "fit_seconds_by_model": fit_seconds,
        "test_inference_seconds": inference_seconds,
        "calibration_threshold": calibrator.threshold,
        "class_labels": [str(value) for value in classes],
        "feature_count": X.shape[1],
        "categorical_feature_count": sum(categorical),
        "split_sizes": {**{name: len(rows) for name, rows in parts.items()}, "test": len(outer_test)},
        "split_index_sha256": {
            **{name: index_hash(rows) for name, rows in parts.items()},
            "outer_train": index_hash(outer_train),
            "test": index_hash(outer_test),
        },
        "protocol": f"CC18 official repeat=0 fold={fold} sample=0; 60/20/20 inner train/selection/calibration",
        "limitations": [
            "Interpret each record within its full run set; one fold alone is a pilot.",
            "CC18 tasks here are IID benchmarks; no natural domain shift is claimed.",
            "Conformal marginal coverage requires exchangeability and is not guaranteed under shift.",
        ],
    }
    if shift_lab:
        result["shift_lab"] = run_shift_lab(
            chosen, calibrator, X.iloc[parts["train"]], X.iloc[outer_test],
            y[outer_test], categorical, seed,
        )
    return result


def run_openml_task(
    task_id: int, *, seed: int = 42, alpha: float = 0.1,
    max_rows: int = 5000, cache_directory: str | Path = "data/raw/openml",
    advanced: bool = False,
    shift_lab: bool = False,
    fold: int = 0,
) -> dict:
    """Download a pinned public task and evaluate one specified official test fold."""
    if task_id not in TASKS:
        raise ValueError(f"Task {task_id} is not in the pinned M1 manifest")
    try:
        import openml
    except ImportError as exc:
        raise RuntimeError("Install the OpenML extra: pip install -e '.[openml]'") from exc
    openml.config.cache_directory = Path(cache_directory).resolve()
    spec = TASKS[task_id]
    task = openml.tasks.get_task(task_id)
    if task.dataset_id != spec.dataset_id or task.target_name != spec.target:
        raise ValueError(f"OpenML task {task_id} no longer matches the pinned metadata")
    dataset = task.get_dataset()
    if (dataset.id != spec.dataset_id or dataset.version != spec.dataset_version
            or dataset.name != spec.dataset_name or dataset.licence != spec.licence
            or (spec.md5_checksum and dataset.md5_checksum != spec.md5_checksum)):
        raise ValueError(f"OpenML dataset for task {task_id} does not match the pinned manifest")
    dimensions = task.get_split_dimensions()
    if fold < 0 or fold >= dimensions[1]:
        raise ValueError(f"Fold {fold} is outside OpenML task's {dimensions[1]} folds")
    X, y, categorical, names = dataset.get_data(target=task.target_name, dataset_format="dataframe")
    if X.columns.tolist() != names:
        raise ValueError("Feature names are misaligned with OpenML data")
    outer_train, outer_test = task.get_train_test_split_indices(fold=fold, repeat=0, sample=0)
    result = evaluate_arrays(
        X, np.asarray(y), categorical, outer_train, outer_test,
        seed=seed, alpha=alpha, max_rows=max_rows, advanced=advanced,
        shift_lab=shift_lab, fold=fold,
    )
    versions = {
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "openml": openml.__version__,
    }
    if advanced:
        import catboost
        import xgboost

        versions.update({"catboost": catboost.__version__, "xgboost": xgboost.__version__})
    return {
        "task": asdict(spec),
        "source_url": f"https://www.openml.org/t/{task_id}",
        "dataset_url": f"https://www.openml.org/d/{spec.dataset_id}",
        "dataset_md5_observed": dataset.md5_checksum,
        "row_count": len(X),
        "split_dimensions": list(dimensions),
        "package_versions": versions,
        "result": result,
    }


def save_benchmark(results: list[dict], destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"benchmark": "OpenML-CC18 M1 pilot", "tasks": results},
                               indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path
