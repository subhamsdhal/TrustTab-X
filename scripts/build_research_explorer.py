"""Package a small, traceable dashboard view of completed result JSON files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "results"
OUTPUT = ROOT / "src" / "trusttabx" / "m5_web" / "research_results.json"
EXPECTED_SHA256 = {
    "acs_income_2018_geographic_expanded": "99be91c2b36a9b4d4a08eacd2519994fa50feb1414b91df855187ca1950d344b",
    "tableshift_diabetes_12000": "5d63e625794ec514a85259edd10bbe2e3563006408cab1d0c53aeec5b916cf9d",
    "bank_temporal_2008_2010": "75e17e0524099f6a79ef37c79629ec57e7d361796c57ed8ac52b07d5e7b93af5",
    "college_scorecard_2025_confirmation": "add11b3b6875ea3bd4a6c819804d9f850b36e56fcd3e0235f618f2e7cdef3781",
    "m4_2_shopper_confirmation": "98b851dacb09ea3e30bab66e2f0339d885d125c125f093a89831745748cf5fe2",
    "m4_2_research_decision": "19cc909954d7ef4f6648601c467c7c5b0a90ae1e45f5975ef68bf289056235f0",
}

MODELS = [
    "logistic_regression", "hist_gradient_boosting", "random_forest", "xgboost", "catboost_ohe",
]
POLICIES = [
    "static", "uniform", "weighted", "metadata_router", "simple_router", "calibrated_static",
]


def _source(name: str, provenance: dict[str, str]) -> dict:
    path = RESULTS / f"{name}.json"
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256[name]:
        raise ValueError(f"Verified result changed: {path.name}")
    provenance[f"reports/results/{path.name}"] = digest
    return json.loads(raw)


def _metrics(values: dict) -> dict:
    return {key: values.get(key) for key in ("brier", "log_loss", "ece_10_bins")}


def _policy_rows(values: dict[str, dict], order: list[str] = POLICIES) -> list[dict]:
    return [{"policy": policy, **_metrics(values[policy])} for policy in order if policy in values]


def build_explorer() -> dict:
    provenance: dict[str, str] = {}
    acs = _source("acs_income_2018_geographic_expanded", provenance)
    diabetes = _source("tableshift_diabetes_12000", provenance)
    bank = _source("bank_temporal_2008_2010", provenance)
    college = _source("college_scorecard_2025_confirmation", provenance)
    shopper = _source("m4_2_shopper_confirmation", provenance)
    decision = _source("m4_2_research_decision", provenance)
    if decision["decision"] != "C. NO ADAPTIVE ADVANTAGE DEMONSTRATED":
        raise ValueError("Unexpected research decision; review before packaging")

    datasets = []
    for key, title, source, family, windows, context in [
        ("acs", "ACSIncome", acs, "acs", [("ID test", "id_test"), ("OOD states", "ood_test")],
         "2018 ACS PUMS via Folktables; sampled geographic state holdout."),
        ("healthcare", "Healthcare admissions", diabetes, "diabetes",
         [("ID test", "id_test"), ("OOD admission sources", "ood_test")],
         "UCI Diabetes 130-US-hospitals dataset; admission-source domain holdout."),
        ("banking", "Bank marketing", bank, "bank",
         [("Near future", "near_future"), ("Final future", "final_future")],
         "UCI Bank Marketing; forward-only pre-call chronological windows, not IID."),
    ]:
        selected = source["results"]["five_experts"]
        shift = [{"label": label, **_metrics(selected[field]["metrics"])} for label, field in windows]
        family_summary = decision["family_summary"][family]
        mean = [{"policy": policy,
                 "brier": family_summary["mean_brier_by_policy"][policy],
                 "log_loss": family_summary["mean_log_loss_by_policy"][policy],
                 "ece_10_bins": None} for policy in POLICIES]
        datasets.append({
            "id": key, "title": title, "provenance": context, "protocol": source.get("benchmark"),
            "source_url": source.get("source_url") or (
                "https://www.census.gov/programs-surveys/acs/microdata.html" if key == "acs" else None),
            "result_files": [f"reports/results/{path}.json" for path in (
                {"acs": "acs_income_2018_geographic_expanded", "healthcare": "tableshift_diabetes_12000",
                 "banking": "bank_temporal_2008_2010"}[key], "m4_2_research_decision")],
            "selected_model": selected["selected_model"],
            "selection_brier": source["selection_brier_by_model"],
            "shift_kind": "ID / OOD" if key != "banking" else "Forward time",
            "shift": shift, "comparisons": [{"label": "Outer source-domain mean",
                                               "n_domains": family_summary["outer_domains"],
                                               "rows": mean}],
            "comparison_note": "Mean across nested source domains; separate from final ID/OOD or future holdout.",
            "limitations": source["limitations"],
        })

    college_names = list(college["source_selection_brier"])
    college_selected = college_names[college["selection"]["static_index"]]
    datasets.append({
        "id": "college", "title": "College Scorecard", "provenance":
        "US College Scorecard; 2025 held-out cohort with institution-disjoint partitions.",
        "protocol": college["protocol"], "source_url":
        "https://catalog.data.gov/dataset/college-scorecard",
        "result_files": ["reports/results/college_scorecard_2025_confirmation.json"],
        "selected_model": college_selected,
        "selection_brier": college["source_selection_brier"],
        "shift_kind": "Held-out cohort", "shift": [{"label": "2025 final cohort",
            **_metrics(college["policies"]["static"]["probability"])}],
        "comparisons": [{"label": "2025 final cohort", "n_domains": 1,
            "rows": _policy_rows({
                "static": college["policies"]["static"]["probability"],
                "uniform_ensemble": college["policies"]["uniform_ensemble"]["probability"],
                "weighted_ensemble": college["policies"]["weighted_ensemble"]["probability"],
                "confidence_expert": college["policies"]["confidence_expert"]["probability"],
                "reliability_router": college["policies"]["reliability_router"]["probability"],
                "calibrated_static": college["policies"]["calibrated_static"]["probability"],
            }, ["static", "uniform_ensemble", "weighted_ensemble", "confidence_expert",
                "reliability_router", "calibrated_static"])}],
        "comparison_note": "Single held-out cohort; College policy names match the saved confirmation file.",
        "limitations": ["Single 2025 cohort; small final sample and no independent family replication.",
                        "Outcome timing and institutional applicability require separate review."],
    })

    shopper_comparisons = []
    for label, key in [("November", "nov"), ("December", "dec")]:
        values = shopper["environments"][key]
        shopper_comparisons.append({"label": label, "n_domains": 1,
                                    "n_rows": values["n"], "rows": _policy_rows({
                                        policy: value["probability"]
                                        for policy, value in values["policies"].items()})})
    datasets.append({
        "id": "shopper", "title": "Online shoppers", "provenance":
        "UCI Online Shoppers Purchasing Intention; month-shift confirmation (November and December).",
        "protocol": shopper["protocol"], "source_url":
        "https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset",
        "result_files": ["reports/results/m4_2_shopper_confirmation.json",
                         "reports/results/m4_2_research_decision.json"],
        "selected_model": shopper["selected_expert"],
        "selection_brier": shopper["selection_brier"], "shift_kind": "Shifted months",
        "shift": [{"label": group["label"], **_metrics(next(
            row for row in group["rows"] if row["policy"] == "static"))}
            for group in shopper_comparisons],
        "comparisons": shopper_comparisons,
        "comparison_note": "November and December are related months in one dataset family; neither is an IID test.",
        "limitations": ["Exact session timestamps and user IDs are unavailable.",
                        "Pre-checkout availability of retrospective page features is not independently certified."],
    })

    return {
        "schema_version": 1, "decision": decision["decision"],
        "release_classification": "Local Demo Only",
        "models": MODELS, "datasets": datasets,
        "methodology": [
            "Preprocessing and model fits use source training partitions; selection, calibration and final evaluation remain separate.",
            "Nested source-domain comparisons use held-out states, admissions or chronological blocks; shopper months form one confirmation family.",
            "Brier, log loss and 10-bin expected calibration error are lower-is-better metrics. Missing metrics are shown as unavailable.",
            "ID/OOD, forward-time and held-out-cohort results are different protocols and should not be pooled into a single ranking.",
        ],
        "decision_explanation": (
            "Across 13 nested source domains in three families, the metadata router did not "
            "beat both the source-selected fixed expert and weighted ensemble in every family. "
            "On November shopper sessions it lost to both controls; its December gain over the "
            "weighted ensemble was below the preregistered margin. The two months are related, "
            "and source-fixed selective acceptance was inconsistent."
        ),
        "source_files_sha256": provenance,
    }


def main() -> None:
    OUTPUT.write_text(json.dumps(build_explorer(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
