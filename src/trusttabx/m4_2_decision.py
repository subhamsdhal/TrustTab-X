"""Apply the frozen M4.2 research decision rule to aggregate run files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FAMILIES = ("acs", "diabetes", "bank")
METHODS = ("static", "uniform", "weighted", "metadata_router", "simple_router",
           "calibrated_static")


def decide_m4_2(source: dict[str, dict], confirmation: dict) -> dict:
    family_summary = {}
    domain_count = 0
    for family in FAMILIES:
        domains = source[family]["outer_domains"]
        domain_count += len(domains)
        family_summary[family] = {
            "outer_domains": len(domains),
            "mean_brier_by_policy": {
                method: float(np.mean([domain["policies"][method]["probability"]["brier"]
                                       for domain in domains.values()]))
                for method in METHODS
            },
            "mean_log_loss_by_policy": {
                method: float(np.mean([domain["policies"][method]["probability"]["log_loss"]
                                       for domain in domains.values()]))
                for method in METHODS
            },
        }
    router_source_gate = all(
        family_summary[family]["mean_brier_by_policy"]["metadata_router"]
        < min(family_summary[family]["mean_brier_by_policy"]["static"],
              family_summary[family]["mean_brier_by_policy"]["weighted"])
        for family in FAMILIES
    )
    final = confirmation["environments"]
    confirmation_gate = all(
        final[month]["policies"]["metadata_router"]["probability"]["brier"]
        <= min(final[month]["policies"]["static"]["probability"]["brier"],
               final[month]["policies"]["weighted"]["probability"]["brier"]) - 0.002
        and final[month]["policies"]["metadata_router"]["probability"]["log_loss"]
        <= min(final[month]["policies"]["static"]["probability"]["log_loss"],
               final[month]["policies"]["weighted"]["probability"]["log_loss"])
        for month in ("nov", "dec")
    )
    selective_support = []
    for method in ("static", "uniform", "weighted", "calibrated_static"):
        for criterion in ("confidence", "disagreement"):
            qualifies = True
            for month in ("nov", "dec"):
                candidate = final[month]["policies"][method]
                policy = candidate["selective"][criterion]["source_threshold"]
                baseline = final[month]["policies"]["static"]["selective"][
                    "confidence"]["source_threshold"]
                if not (policy["target_coverage"] >= 0.8
                        and policy["target_error_risk"] is not None
                        and policy["target_error_risk"] < baseline["target_error_risk"]
                        and candidate["probability"]["brier"]
                        <= final[month]["policies"]["static"]["probability"]["brier"]
                        + 0.002):
                    qualifies = False
                    break
            if qualifies:
                selective_support.append(f"{method}:{criterion}")
    if router_source_gate and confirmation_gate:
        decision = "A. ADAPTIVE ROUTER SUPPORTED"
    elif selective_support:
        decision = "B. SELECTIVE RELIABILITY SUPPORTED"
    else:
        decision = "C. NO ADAPTIVE ADVANTAGE DEMONSTRATED"
    return {
        "decision": decision,
        "source_outer_domains": domain_count,
        "family_summary": family_summary,
        "router_source_gate": router_source_gate,
        "router_confirmation_gate": confirmation_gate,
        "selective_policies_passing_both_months": selective_support,
        "confirmation": {
            month: {
                "n": final[month]["n"],
                "brier_by_policy": {method: final[month]["policies"][method][
                    "probability"]["brier"] for method in METHODS},
                "log_loss_by_policy": {method: final[month]["policies"][method][
                    "probability"]["log_loss"] for method in METHODS},
                "source_threshold_coverage_by_policy": {
                    method: final[month]["policies"][method]["selective"][
                        "confidence"]["source_threshold"]["target_coverage"]
                    for method in METHODS},
            } for month in ("nov", "dec")
        },
        "production_research_default": "source-validation-selected fixed expert",
        "statistical_scope": (
            "Three source families and two related confirmation months; no family-level "
            "confidence interval or arbitrary-shift guarantee"
        ),
    }


def run_decision(result_dir: str | Path = "reports/results") -> dict:
    root = Path(result_dir)
    source = {family: json.loads((root / f"m4_2_nested_{family}.json").read_text(
        encoding="utf-8")) for family in FAMILIES}
    confirmation = json.loads((root / "m4_2_shopper_confirmation.json").read_text(
        encoding="utf-8"))
    return decide_m4_2(source, confirmation)


def save_decision(result: dict, output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path
