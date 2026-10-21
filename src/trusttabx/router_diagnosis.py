"""Exploratory, source-only diagnosis and leave-one-family-out meta-transfer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss

from .reliability import probability_metrics
from .router_prototype import FAMILIES, fit_reliability_models

RESERVED = {"row_index", "domain", "label", "router_metadata"}


def load_source_predictions(family: str, directory: str | Path = "data/processed") -> dict:
    with np.load(Path(directory) / f"{family}_source_crossfit_predictions.npz") as saved:
        return {name: saved[name] for name in saved.files}


def _matrix(record: dict, names: list[str]) -> np.ndarray:
    return np.column_stack([record[name] for name in names])


def run_router_diagnosis(directory: str | Path = "data/processed") -> dict:
    records = {family: load_source_predictions(family, directory) for family in FAMILIES}
    names = [key for key in records[FAMILIES[0]] if key not in RESERVED]
    if any([key for key in records[f] if key not in RESERVED] != names for f in FAMILIES):
        raise ValueError("Expert schemas differ")
    result = {"protocol": "exploratory source-only leave-one-family-out meta-transfer",
              "expert_names": names, "families": {}}
    for heldout in FAMILIES:
        training = tuple(f for f in FAMILIES if f != heldout)
        models, fit_meta = fit_reliability_models(directory, training)
        source_brier = []
        for family in training:
            record = records[family]
            y, p = record["label"], _matrix(record, names)
            domain_scores = []
            for domain in np.unique(record["domain"]):
                mask = record["domain"] == domain
                domain_scores.append(np.mean((p[mask] - y[mask, None]) ** 2, axis=0))
            source_brier.append(np.mean(domain_scores, axis=0))
        averaged = np.mean(source_brier, axis=0)
        static_index = int(np.argmin(averaged))
        weights = 1 / np.maximum(averaged, 1e-8)
        weights /= weights.sum()
        record = records[heldout]
        y, p = record["label"], _matrix(record, names)
        metadata = record["router_metadata"]
        predicted_error = np.column_stack([
            np.clip(models[name].predict(metadata), 0, 1) for name in names
        ])
        choice = predicted_error.argmin(axis=1)
        router = p[np.arange(len(y)), choice]
        policies = {"static_other_families": p[:, static_index],
                    "uniform_ensemble": p.mean(axis=1),
                    "weighted_other_families": p @ weights,
                    "router_other_families": router}
        family_result = {
            "training_families": list(training), "router_fit": fit_meta,
            "source_selected_expert": names[static_index],
            "source_brier_by_expert": dict(zip(names, averaged.tolist())),
            "router_choice_fraction": {name: float(np.mean(choice == i))
                                       for i, name in enumerate(names)},
            "domains": {},
        }
        for domain in np.unique(record["domain"]):
            mask = record["domain"] == domain
            yy, pp = y[mask], p[mask]
            err = (pp - yy[:, None]) ** 2
            disagreement = pp.std(axis=1)
            mean_offdiag = float(np.mean(np.corrcoef(err, rowvar=False)[
                np.triu_indices(len(names), 1)]))
            chosen_error = (router[mask] - yy) ** 2
            risk_rank = spearmanr(predicted_error[mask, choice[mask]], chosen_error).statistic
            family_result["domains"][str(domain)] = {
                "n": int(mask.sum()), "positive_rate": float(yy.mean()),
                "expert_brier": {name: float(brier_score_loss(yy, pp[:, i]))
                                 for i, name in enumerate(names)},
                "mean_expert_disagreement": float(disagreement.mean()),
                "mean_pairwise_squared_error_correlation": mean_offdiag,
                "router_predicted_vs_realized_error_spearman": (
                    float(risk_rank) if np.isfinite(risk_rank) else None
                ),
                "label_aware_per_row_oracle_brier_diagnostic": float(err.min(axis=1).mean()),
                "policies": {name: probability_metrics(yy, probability[mask])
                             for name, probability in policies.items()},
            }
        result["families"][heldout] = family_result
    return result


def save_diagnosis(result: dict, output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path
