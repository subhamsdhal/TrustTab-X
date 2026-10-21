"""Verify M7 numeric claims, portfolio copies, links and PDF structure."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "portfolio/final"
RESULTS = ROOT / "reports/results"


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def normalized_bytes(path: Path) -> bytes:
    content = path.read_bytes()
    return content.replace(b"\r\n", b"\n") if path.suffix.lower() in {
        ".md", ".json", ".py", ".txt"} else content


def near(value: float, expected: float, tolerance: float = 0.0000006) -> None:
    assert abs(value - expected) <= tolerance, (value, expected)


def claims() -> None:
    family_values = {
        "acs": (0.141614, 0.141437, 0.141332, 7),
        "diabetes": (0.189329, 0.189170, 0.188566, 4),
        "bank": (0.056951, 0.057372, 0.056665, 2),
    }
    wins = 0
    coverage = {"static": 0, "metadata_router": 0, "weighted": 0}
    for family, (static, router, weighted, n) in family_values.items():
        domains = load(f"m4_2_nested_{family}")["outer_domains"]
        assert len(domains) == n
        for policy, expected in (("static", static), ("metadata_router", router),
                                 ("weighted", weighted)):
            values = [d["policies"][policy]["probability"]["brier"] for d in domains.values()]
            near(sum(values) / n, expected)
            coverage[policy] += sum(
                d["policies"][policy]["selective"]["confidence"]
                ["source_threshold"]["target_coverage"] >= 0.8
                for d in domains.values()
            )
        wins += sum(d["policies"]["metadata_router"]["probability"]["brier"]
                    < d["policies"]["static"]["probability"]["brier"]
                    for d in domains.values())
    assert wins == 6
    assert coverage == {"static": 9, "metadata_router": 6, "weighted": 8}

    shopper = load("m4_2_shopper_confirmation")
    expected = {
        "nov": (0.190073, 0.189411, 0.196618),
        "dec": (0.103842, 0.102534, 0.102092),
    }
    for month, values in expected.items():
        policies = shopper["environments"][month]["policies"]
        for policy, value in zip(("static", "weighted", "metadata_router"), values):
            near(policies[policy]["probability"]["brier"], value)
    near(sum(shopper["fit_seconds_by_expert"].values()), 3.127, 0.001)
    near(shopper["fit_seconds_by_expert"]["random_forest"], 0.393, 0.001)

    college = load("college_scorecard_2025_confirmation")["policies"]
    near(college["static"]["probability"]["brier"], 0.263707)
    near(college["reliability_router"]["probability"]["brier"], 0.285757)
    smoke = load("m5_1_security_verification")["authenticated_http_smoke"]
    assert smoke["measured_prediction_requests"] == 100
    near(smoke["http_p95_ms"], 37.25, 0.01)
    assert smoke["service_predictions_including_initial_batch"] == 102

    demo = load("m7_local_demo")
    assert demo["artifact"]["tracked_pipeline_and_split_hashes_match"]
    assert demo["artifact"]["source_archive_hashes_match"]
    assert demo["drift"]["states"] == ["CT"]
    assert demo["unauthorized_model_status"] == 401
    assert demo["admin_only_delayed_labels_operator_status"] == 403
    assert demo["telemetry"]["requests"] == 101
    assert demo["telemetry"]["errors"] == 2
    near(demo["delayed_labels"]["metrics"]["brier"], 0.09970031895866144)


def package() -> None:
    manifest = json.loads((PACKAGE / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    actual = {str(p.relative_to(PACKAGE)).replace("\\", "/"):
              hashlib.sha256(normalized_bytes(p)).hexdigest()
              for p in PACKAGE.rglob("*") if p.is_file() and p.name != "PACKAGE_MANIFEST.json"}
    assert actual == manifest["files_sha256"]
    for source, target in manifest["copied_from_project"].items():
        if not source.endswith(".md"):
            assert normalized_bytes(ROOT / source) == normalized_bytes(PACKAGE / target), target
    for markdown in PACKAGE.rglob("*.md"):
        content = markdown.read_text(encoding="utf-8")
        for link in re.findall(r"\]\(([^)]+)\)", content):
            if link.startswith(("https://", "http://", "mailto:")):
                continue
            path = link.split("#", 1)[0]
            if path:
                assert (markdown.parent / path).exists(), (markdown, link)
    forbidden = ("*.joblib", "*.db", "*.zip", "*.env", "*.log", "*.csv")
    assert not any(path for pattern in forbidden for path in PACKAGE.rglob(pattern))
    pdf = PACKAGE / "TrustTab-X-manuscript.pdf"
    assert pdf.read_bytes().startswith(b"%PDF-") and pdf.stat().st_size > 100_000


if __name__ == "__main__":
    claims()
    package()
    print("M7 claims, copied artifacts, PDF header, portfolio links and exclusion check: PASS")
