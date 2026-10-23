"""The public explorer must match completed, tracked aggregate results exactly."""

import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "trusttabx_research_builder", ROOT / "scripts" / "build_research_explorer.py")
assert SPEC is not None and SPEC.loader is not None
explorer_builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(explorer_builder)
MODELS = explorer_builder.MODELS
OUTPUT = explorer_builder.OUTPUT
RESULTS = explorer_builder.RESULTS
build_explorer = explorer_builder.build_explorer


def test_packaged_results_rebuild_and_provenance():
    packaged = json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert packaged == build_explorer()
    assert packaged["decision"] == "C. NO ADAPTIVE ADVANTAGE DEMONSTRATED"
    assert packaged["release_classification"] == "Local Demo Only"
    assert len(packaged["datasets"]) == 5
    assert packaged["models"] == MODELS
    for path, digest in packaged["source_files_sha256"].items():
        assert hashlib.sha256((RESULTS.parent.parent / path).read_bytes()).hexdigest() == digest


def test_builder_rejects_modified_result(tmp_path, monkeypatch):
    name = "m4_2_research_decision"
    (tmp_path / f"{name}.json").write_text('{}', encoding="utf-8")
    monkeypatch.setattr(explorer_builder, "RESULTS", tmp_path)
    with pytest.raises(ValueError, match="Verified result changed"):
        explorer_builder._source(name, {})


def test_shift_and_selection_chart_values_are_source_values():
    data = build_explorer()
    by_id = {dataset["id"]: dataset for dataset in data["datasets"]}
    for key, filename, source_window, target_window in [
        ("acs", "acs_income_2018_geographic_expanded", "id_test", "ood_test"),
        ("healthcare", "tableshift_diabetes_12000", "id_test", "ood_test"),
        ("banking", "bank_temporal_2008_2010", "near_future", "final_future"),
    ]:
        source = json.loads((RESULTS / f"{filename}.json").read_text())
        selected = source["results"]["five_experts"]
        dataset = by_id[key]
        assert dataset["selected_model"] == selected["selected_model"]
        assert dataset["selection_brier"] == source["selection_brier_by_model"]
        for chart_row, window in zip(dataset["shift"], (source_window, target_window), strict=True):
            for metric in ("brier", "log_loss", "ece_10_bins"):
                assert chart_row[metric] == selected[window]["metrics"][metric]

    college = json.loads((RESULTS / "college_scorecard_2025_confirmation.json").read_text())
    assert by_id["college"]["shift"][0]["brier"] == college["policies"]["static"]["probability"]["brier"]
    assert by_id["college"]["shift"][0]["ece_10_bins"] is None
    shopper = json.loads((RESULTS / "m4_2_shopper_confirmation.json").read_text())
    for row, month in zip(by_id["shopper"]["shift"], ("nov", "dec"), strict=True):
        assert row["brier"] == shopper["environments"][month]["policies"]["static"]["probability"]["brier"]


def test_policy_chart_values_preserve_decision_and_confirmation():
    data = build_explorer()
    by_id = {dataset["id"]: dataset for dataset in data["datasets"]}
    decision = json.loads((RESULTS / "m4_2_research_decision.json").read_text())
    for key, family in (("acs", "acs"), ("healthcare", "diabetes"), ("banking", "bank")):
        rows = {row["policy"]: row for row in by_id[key]["comparisons"][0]["rows"]}
        source = decision["family_summary"][family]
        for policy, row in rows.items():
            assert row["brier"] == source["mean_brier_by_policy"][policy]
            assert row["log_loss"] == source["mean_log_loss_by_policy"][policy]
            assert row["ece_10_bins"] is None
    shopper = json.loads((RESULTS / "m4_2_shopper_confirmation.json").read_text())
    for group, month in zip(by_id["shopper"]["comparisons"], ("nov", "dec"), strict=True):
        assert group["n_rows"] == shopper["environments"][month]["n"]
        for row in group["rows"]:
            source = shopper["environments"][month]["policies"][row["policy"]]["probability"]
            assert row["brier"] == source["brier"]
            assert row["log_loss"] == source["log_loss"]
    college = json.loads((RESULTS / "college_scorecard_2025_confirmation.json").read_text())
    for row in by_id["college"]["comparisons"][0]["rows"]:
        source = college["policies"][row["policy"]]["probability"]
        assert row["brier"] == source["brier"]
        assert row["log_loss"] == source["log_loss"]


def test_javascript_chart_geometry_uses_exact_values_and_omits_missing():
    if not shutil.which("node"):
        pytest.skip("Node.js unavailable; source-to-chart value tests still run")
    script = Path(__file__).resolve().parents[1] / "src/trusttabx/m5_web/research.js"
    sample = build_explorer()["datasets"][0]["shift"]
    code = (
        "const chart=require(process.argv[1]);"
        f"const rows={json.dumps([{'label': row['label'], 'brier': row['brier']} for row in sample])};"
        "const out=chart.chartGeometry(rows.concat([{label:'Missing',brier:null}]),'brier');"
        "process.stdout.write(JSON.stringify(out));"
    )
    result = subprocess.run(["node", "-e", code, str(script)], capture_output=True,
                            text=True, check=True, timeout=10)
    geometry = json.loads(result.stdout)
    assert [bar["value"] for bar in geometry["bars"]] == [row["brier"] for row in sample]
    assert len(geometry["bars"]) == 2
    assert all(0 <= bar["fraction"] <= 1 for bar in geometry["bars"])
    assert geometry["axisMax"] >= max(row["brier"] for row in sample)
