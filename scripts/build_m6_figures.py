"""Regenerate M6 figures solely from the frozen, tracked result JSON files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "results"
FIGURES = ROOT / "reports" / "figures"
SOURCES = {
    name: RESULTS / f"{name}.json"
    for name in (
        "acs_income_2018_geographic_expanded",
        "tableshift_diabetes_12000",
        "bank_temporal_2008_2010",
        "m4_2_nested_acs",
        "m4_2_nested_diabetes",
        "m4_2_nested_bank",
        "m4_2_shopper_confirmation",
        "m4_1_source_router_diagnosis",
    )
}


def load(name: str) -> dict:
    return json.loads(SOURCES[name].read_text(encoding="utf-8"))


def save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=200, bbox_inches="tight")
    plt.close()


def build() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    # The bank comparison is forward-period versus later forward-period, not ID/OOD.
    specs = [
        ("ACS income", "acs_income_2018_geographic_expanded", "id_test", "ood_test"),
        ("Hospital readmission", "tableshift_diabetes_12000", "id_test", "ood_test"),
        ("Bank campaign", "bank_temporal_2008_2010", "near_future", "final_future"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for i, (label, source, first, second) in enumerate(specs):
        result = load(source)["results"]["five_experts"]
        values = [result[k]["metrics"]["brier"] for k in (first, second)]
        ax.plot([i - 0.12, i + 0.12], values, marker="o", linewidth=2, label=label)
        for x, y in zip((i - 0.12, i + 0.12), values):
            ax.annotate(f"{y:.3f}", (x, y), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(range(3), [s[0] for s in specs])
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_title("Frozen fixed-expert shift evaluation")
    ax.text(0.5, -0.23, "Left: source ID / near future. Right: target OOD / later future. Different tasks have different prevalence.", transform=ax.transAxes, ha="center", fontsize=8)
    save("shift_brier.png")

    nested = [load(f"m4_2_nested_{family}") for family in ("acs", "diabetes", "bank")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for ax, family, result in zip(axes, ("ACS", "Hospital", "Bank"), nested):
        domains = result["outer_domains"]
        labels = list(domains)
        router = [d["policies"]["metadata_router"]["probability"]["brier"] for d in domains.values()]
        static = [d["policies"]["static"]["probability"]["brier"] for d in domains.values()]
        weighted = [d["policies"]["weighted"]["probability"]["brier"] for d in domains.values()]
        x = np.arange(len(labels))
        ax.bar(x - 0.19, np.subtract(router, static), width=0.38, label="router - static")
        ax.bar(x + 0.19, np.subtract(router, weighted), width=0.38, label="router - weighted")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x, labels, rotation=45)
        ax.set_title(f"{family} ({len(labels)} source domains)")
    axes[0].set_ylabel("Brier difference (negative favors router)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Nested source-domain tests: inconsistent router advantage")
    save("nested_router_differences.png")

    diagnosis = load("m4_1_source_router_diagnosis")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, family in enumerate(("acs", "diabetes", "bank")):
        domains = diagnosis["families"][family]["domains"]
        values = [d["mean_pairwise_squared_error_correlation"] for d in domains.values()]
        ax.scatter(np.full(len(values), i) + np.linspace(-0.08, 0.08, len(values)), values, label=family)
        ax.plot([i - 0.18, i + 0.18], [np.mean(values)] * 2, color="black", linewidth=2)
    ax.set_xticks(range(3), ("ACS", "Hospital", "Bank"))
    ax.set_ylim(0.8, 1.01)
    ax.set_ylabel("Pairwise correlation of expert squared errors")
    ax.set_title("Expert errors overlap strongly on diagnosed source domains")
    save("expert_error_correlation.png")

    bank = load("bank_temporal_2008_2010")["results"]["five_experts"]
    fig, ax = plt.subplots(figsize=(5.3, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    for key, label in (("near_future", "near future"), ("final_future", "later future")):
        bins = [b for b in bank[key]["reliability_bins"] if b["count"] >= 20]
        ax.plot([b["mean_predicted"] for b in bins], [b["observed_rate"] for b in bins], marker="o", label=label)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed positive rate", title="Bank temporal calibration (bins with n >= 20)")
    ax.legend()
    save("bank_reliability.png")

    shopper = load("m4_2_shopper_confirmation")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, month in zip(axes, ("nov", "dec")):
        for policy in ("static", "weighted", "metadata_router"):
            threshold = shopper["environments"][month]["policies"][policy]["selective"]["confidence"]["source_threshold"]
            curve = threshold["target_risk_coverage_curve_descriptive"]
            line, = ax.plot([p["coverage"] for p in curve], [p["error_risk"] for p in curve], label=policy)
            ax.scatter(threshold["target_coverage"], threshold["target_error_risk"], color=line.get_color(), marker="x", s=70)
        ax.set(xlabel="Coverage", title=f"{month.title()} (n={shopper['environments'][month]['n']})")
        ax.axvline(0.8, color="gray", linestyle=":", linewidth=1)
    axes[0].set_ylabel("Accepted classification error")
    axes[1].legend(fontsize=8)
    fig.suptitle("Shopper: retrospective target-ranked curves; x = source-fixed threshold")
    save("shopper_risk_coverage.png")

    fit = shopper["fit_seconds_by_expert"]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.barh(["Selected random forest", "All five candidate fits"], [fit["random_forest"], sum(fit.values())])
    ax.set_xlabel("Recorded model fit seconds (single local run)")
    ax.set_title("Shopper candidate-fit cost; excludes pipeline overhead")
    save("shopper_fit_cost.png")

    manifest = {
        "generation_command": ".\\.venv\\Scripts\\python.exe scripts\\build_m6_figures.py",
        "sources": {name: {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in SOURCES.items()},
        "figures": sorted(p.name for p in FIGURES.glob("*.png")),
        "cautions": ["Source domains within a family are correlated, not independent dataset replicates.", "Risk-coverage curves are retrospective target-ranked descriptions; x markers show source-fixed thresholds.", "Fit-cost chart reports model fits only, from one local run."],
    }
    (FIGURES / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    build()
