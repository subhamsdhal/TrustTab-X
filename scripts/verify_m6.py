"""Fast integrity checks for the frozen M6 manuscript and figures."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "reports/paper/TrustTab-X-manuscript.md"
FIGURES = ROOT / "reports/figures"


def load(name: str) -> dict:
    return json.loads((ROOT / "reports/results" / f"{name}.json").read_text(encoding="utf-8"))


def close(actual: float, expected: float, tol: float = 0.000001) -> None:
    assert abs(actual - expected) <= tol, (actual, expected)


def main() -> None:
    manifest = json.loads((FIGURES / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["sources"].values():
        path = ROOT / item["path"]
        assert path.is_file(), path
        assert hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == item["sha256"], path
    for name in manifest["figures"]:
        path = FIGURES / name
        assert path.is_file() and path.stat().st_size > 1000, path

    for name, early, late, expected_early, expected_late in (
        ("acs_income_2018_geographic_expanded", "id_test", "ood_test", 0.131644, 0.145517),
        ("tableshift_diabetes_12000", "id_test", "ood_test", 0.201603, 0.250504),
        ("bank_temporal_2008_2010", "near_future", "final_future", 0.063753, 0.337574),
    ):
        result = load(name)["results"]["five_experts"]
        close(result[early]["metrics"]["brier"], expected_early)
        close(result[late]["metrics"]["brier"], expected_late)

    shopper = load("m4_2_shopper_confirmation")
    nov = shopper["environments"]["nov"]["policies"]
    dec = shopper["environments"]["dec"]["policies"]
    close(nov["metadata_router"]["probability"]["brier"], 0.196618)
    close(dec["weighted"]["probability"]["brier"] - dec["metadata_router"]["probability"]["brier"], 0.000442)
    close(sum(shopper["fit_seconds_by_expert"].values()), 3.127, tol=0.001)

    for family, expected in (("acs", 7), ("diabetes", 4), ("bank", 2)):
        assert len(load(f"m4_2_nested_{family}")["outer_domains"]) == expected

    documents = [PAPER, ROOT / "README.md", *sorted((ROOT / "docs").glob("M6_*.md"))]
    for document in documents:
        text = document.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if target.startswith(("https://", "http://", "mailto:")):
                continue
            path = target.split("#", 1)[0]
            if path:
                assert (document.parent / path).exists(), (document, target)
    text = PAPER.read_text(encoding="utf-8")
    assert "router superiority" in text and "local demo only" in text.lower()
    print("M6 evidence, figures, and local manuscript links: PASS")


if __name__ == "__main__":
    main()
