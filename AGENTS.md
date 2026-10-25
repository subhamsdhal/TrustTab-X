# Codex / coding-agent instructions for TrustTab-X

You are implementing a research-first Data Science framework, not a demo that fabricates model wins.

## Current repo state (2026-10-25)

- The original v0.1 synthetic smoke pipeline and completed bounded real-data studies remain reproducible from their recorded protocols and sources.
- The M4.2 decision is **no adaptive advantage demonstrated**. Research-only routers were evaluated; the local FastAPI backend serves a source-selected fixed ACSIncome expert. No TabPFN or other foundation-model experiment was completed.
- The Research Explorer, secured local demo, Docker Compose setup and Windows launchers are implemented. The release classification is **Local Demo Only**. See `docs/RELEASE_SCOPE.md`, `docs/FINAL_ENGINEERING_VERIFICATION.md`, `docs/AUTOPILOT_STATUS.md` and `RESULTS_VERIFICATION.md` for scope and evidence.

## Non-negotiable rules

1. Do not claim any model improves metrics unless the executed held-out experiment demonstrates it.
2. Keep fit/selection/calibration/test data disjoint and avoid fitting preprocessing on all data.
3. Under arbitrary distribution shift, conformal coverage is empirical only, not a distribution-free guarantee.
4. Separate and record TabPFN **code licence** from its **checkpoint weights licence**; do not deploy non-commercial model weights to a commercial service.
5. Use seeds, deterministic protocols where possible, dataset/version manifests and reproducible commands.
6. If a benchmark dataset has time/group partitions, do not randomly shuffle through them.
7. No credentials, private data, paid API calls or remote training workloads without explicit authorization.
8. Maintain passing tests and update docs + limitations with code changes.
9. Never change an unrelated repository or rewrite the user's previous project.

## Next task

There is no further V1 research or engineering milestone. `docs/NEXT_CODEX_TASK.md` records the project closure. Preserve the locked findings and local-demo boundary; any later work requires a separately scoped request. Keep offline fixture tests and never fetch OpenML data in CI.

## Dev commands

```bash
python -m pip install -e '.[dev]'
pytest -q
trusttabx --output outputs/smoke_baseline.json
```
