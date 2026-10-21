"""Assemble the final review directory from vetted project artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "portfolio/final"
COPIES = {
    "reports/paper/TrustTab-X-manuscript.pdf": "TrustTab-X-manuscript.pdf",
    "reports/paper/TrustTab-X-manuscript.md": "TrustTab-X-manuscript.md",
    "docs/M6_CLAIM_TRACEABILITY.md": "CLAIM_TRACEABILITY.md",
    "docs/M6_REPRODUCIBILITY.md": "REPRODUCIBILITY.md",
    "docs/M6_INDUSTRIAL_CASE_STUDY.md": "INDUSTRIAL_CASE_STUDY.md",
    "docs/FINAL_PROJECT_AUDIT.md": "FINAL_PROJECT_AUDIT.md",
    "reports/results/m7_local_demo.json": "M7_LOCAL_DEMO.json",
    "reports/results/m5_1_security_verification.json": "evidence/m5_1_security_verification.json",
    "reports/results/college_scorecard_2025_confirmation.json": "evidence/college_scorecard_2025_confirmation.json",
    **{f"reports/figures/{name}": f"figures/{name}" for name in (
        "shift_brier.png", "nested_router_differences.png", "expert_error_correlation.png",
        "bank_reliability.png", "shopper_risk_coverage.png", "shopper_fit_cost.png",
        "manifest.json",
    )},
}

figure_manifest = json.loads((ROOT / "reports/figures/manifest.json").read_text(encoding="utf-8"))
for name, metadata in figure_manifest["sources"].items():
    COPIES[metadata["path"]] = f"evidence/{name}.json"


def digest(path: Path) -> str:
    content = path.read_bytes()
    if path.suffix.lower() in {".md", ".json", ".py", ".txt"}:
        content = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def rebase_markdown(content: str, source: Path, destination: Path) -> str:
    target_map = {(ROOT / src).resolve(): (DEST / target).resolve()
                  for src, target in COPIES.items()}

    def fix(match: re.Match[str]) -> str:
        link = match.group(1)
        if link.startswith(("https://", "http://", "mailto:", "#")):
            return match.group(0)
        path, sep, fragment = link.partition("#")
        original = (source.parent / path).resolve()
        target = target_map.get(original, original)
        relative = os.path.relpath(target, destination.parent).replace("\\", "/")
        return f"]({relative}{sep}{fragment})"

    return re.sub(r"\]\(([^)]+)\)", fix, content)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for source, target in COPIES.items():
        src = ROOT / source
        if not src.is_file():
            raise FileNotFoundError(src)
        dst = DEST / target
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix == ".md":
            dst.write_text(rebase_markdown(src.read_text(encoding="utf-8"), src, dst),
                           encoding="utf-8")
        else:
            shutil.copyfile(src, dst)
    manifest = {
        "scope": "review package; no raw records, fitted model, credentials or audit logs",
        "copied_from_project": COPIES,
        "hash_normalization": "LF-normalized UTF-8 for Markdown/JSON/Python/text; raw bytes for PDF/PNG",
        "files_sha256": {str(p.relative_to(DEST)).replace("\\", "/"):
                         digest(p)
                         for p in sorted(DEST.rglob("*")) if p.is_file()
                         and p.name != "PACKAGE_MANIFEST.json"},
    }
    (DEST / "PACKAGE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Assembled {len(manifest['files_sha256'])} files in {DEST}")


if __name__ == "__main__":
    main()
