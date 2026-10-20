# Codex / coding-agent instructions for TrustTab-X

You are implementing a research-first Data Science framework, not a demo that fabricates model wins.

## Current repo state (2026-10-20)

- The original v0.1 synthetic smoke pipeline remains reproducible.
- M1 now includes a three-task OpenML-CC18 pilot; optional M2 classical experts and M3 stress tests have run. See `docs/AUTOPILOT_STATUS.md` and `RESULTS_VERIFICATION.md` for verified scope.
- The architecture **does not yet have** foundation models, a learned adaptive router, or a backend.

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

Continue from `docs/NEXT_CODEX_TASK.md`: expand the pinned pilot to repeated official folds and seeds, validate natural held-out domains and compute budgets, then evaluate a source-only router. Preserve offline fixture tests and never fetch OpenML data in CI.

## Dev commands

```bash
python -m pip install -e '.[dev]'
pytest -q
trusttabx --output outputs/smoke_baseline.json
```
