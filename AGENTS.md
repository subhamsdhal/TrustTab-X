# Codex / coding-agent instructions for TrustTab-X

You are implementing a research-first Data Science framework, not a demo that fabricates model wins.

## Current repo state

- v0.1 smoke pipeline is complete: `src/trusttabx`, `tests`, CLI, docs and GitHub Actions.
- It deliberately uses only synthetic data and classical baselines.
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

Implement `M1` from `docs/PROJECT_PLAN.md` with an optional OpenML-CC18 loader, pinned task IDs, dataset provenance, train-only preprocessing, and a lightweight 3-task benchmark CLI. Preserve the synthetic smoke test and add fixture-based offline unit tests. Do not fetch OpenML data in CI; integration test only on demand. Provide a benchmark manifest with task IDs and licences, not hard-coded unsupported claims.

## Dev commands

```bash
python -m pip install -e '.[dev]'
pytest -q
trusttabx --output outputs/smoke_baseline.json
```
