"""Source-only domain holdout plans and cross-fitted expert predictions."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import brier_score_loss


@dataclass(frozen=True)
class DomainFold:
    domain: str
    train: np.ndarray
    validation: np.ndarray


def domain_holdout_plan(
    domains: np.ndarray, source_indices: np.ndarray, final_indices: np.ndarray,
    *, ordered: bool = False, groups: np.ndarray | None = None,
) -> list[DomainFold]:
    """Create source-only LODO or forward-domain folds, excluding final indices."""
    domains = np.asarray(domains)
    source = np.asarray(source_indices, dtype=int)
    final = np.asarray(final_indices, dtype=int)
    if domains.ndim != 1 or len(source) == 0 or len(final) == 0:
        raise ValueError("Expected nonempty source/final indices and 1D domain labels")
    if np.any(source < 0) or np.any(final < 0) or max(source.max(), final.max()) >= len(domains):
        raise ValueError("Out-of-range domain indices")
    if len(np.unique(source)) != len(source) or len(np.unique(final)) != len(final):
        raise ValueError("Duplicate row indices")
    if np.intersect1d(source, final).size:
        raise ValueError("Final test rows overlap source cross-fitting rows")
    domain_order = list(dict.fromkeys(domains[np.sort(source)].astype(str)))
    if len(domain_order) < 3:
        raise ValueError("At least three source environments are required")
    if ordered:
        observed = domains[np.sort(source)].astype(str)
        for name in domain_order:
            positions = np.flatnonzero(observed == name)
            if positions[-1] - positions[0] + 1 != len(positions):
                raise ValueError("Ordered source domains must be contiguous")
    folds = []
    for position, name in enumerate(domain_order):
        if ordered and position == 0:
            continue
        validation = source[domains[source].astype(str) == name]
        if ordered:
            previous = set(domain_order[:position])
            train = source[np.isin(domains[source].astype(str), list(previous))]
        else:
            train = source[domains[source].astype(str) != name]
        if len(train) == 0 or len(validation) == 0:
            raise ValueError("Empty domain fold")
        if groups is not None and set(groups[train]) & set(groups[validation]):
            raise ValueError("Entity groups cross domain fit/validation")
        folds.append(DomainFold(name, train, validation))
    return folds


def crossfit_experts(
    X: pd.DataFrame, y: np.ndarray, folds: list[DomainFold], experts: dict,
) -> tuple[dict, dict]:
    """Fit each expert on source folds; emit held-out source predictions only."""
    if not folds or not experts:
        raise ValueError("Cross-fitting needs folds and experts")
    rows, domain_names, labels = [], [], []
    predictions: dict[str, list[np.ndarray]] = {name: [] for name in experts}
    metrics = {}
    for fold in folds:
        rows.append(fold.validation)
        domain_names.extend([fold.domain] * len(fold.validation))
        labels.append(y[fold.validation])
        metrics[fold.domain] = {"n": len(fold.validation), "fit_n": len(fold.train),
                                "positive_rate": float(y[fold.validation].mean()),
                                "experts": {}}
        for name, prototype in experts.items():
            model = clone(prototype)
            start = perf_counter()
            model.fit(X.iloc[fold.train], y[fold.train])
            fit_seconds = perf_counter() - start
            probability = model.predict_proba(X.iloc[fold.validation])[:, 1]
            predictions[name].append(probability)
            metrics[fold.domain]["experts"][name] = {
                "brier": float(brier_score_loss(y[fold.validation], probability)),
                "fit_seconds": fit_seconds,
            }
    stacked = {name: np.concatenate(chunks) for name, chunks in predictions.items()}
    output = {"row_index": np.concatenate(rows), "domain": np.asarray(domain_names),
              "label": np.concatenate(labels), **stacked}
    if len(np.unique(output["row_index"])) != len(output["row_index"]):
        raise ValueError("Validation rows repeated across domain folds")
    return output, metrics


def router_probability_metadata(predictions: dict, expert_names: list[str]) -> np.ndarray:
    """Schema-independent, deployment-available probability summaries."""
    p = np.column_stack([predictions[name] for name in expert_names])
    p = np.clip(p, 1e-8, 1 - 1e-8)
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return np.column_stack([p, entropy, p.mean(axis=1), p.std(axis=1)])
