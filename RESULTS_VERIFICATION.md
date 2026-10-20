# Results verification — 2026-10-20

## Executed locally

- Original starter: `python -m pytest -q -p no:cacheprovider --basetemp=.pytest_tmp` in the project venv: 8 passed before M1 edits.
- After M1–M5 partial changes: a fresh second project-local venv installed from `.[dev]` with single-thread BLAS; **16 offline tests passed**, `ruff check src tests` passed, and `trusttabx --output outputs/clean_smoke_baseline.json` completed with IID/shifted coverages 0.91/0.905.
- `trusttabx --output outputs/smoke_baseline.json`: synthetic baseline completed; IID coverage 0.91 and shifted coverage 0.905.
- `trusttabx benchmark --output outputs/openml_m1_pilot.json`: tasks 3, 31, 10101 completed.
- `trusttabx benchmark --advanced-experts --output outputs/openml_m2_classical_pilot.json`: same three tasks completed with five experts.
- `trusttabx benchmark --shift-lab --output outputs/openml_m3_shift_pilot.json`: same three tasks with controlled stress cohorts completed.
- `trusttabx benchmark --advanced-experts --shift-lab --output outputs/openml_integrated_pilot.json`: completed on retry; this JSON also contains PR-AUC, ECE and selective-risk summaries.
- `trusttabx benchmark --all-folds --seed 42 --seed 43 --seed 44 --output outputs/openml_10fold_3seed_classical.json`: 90 runs completed (3 tasks × 10 official folds × 3 seeds).
- Same command with `--advanced-experts` and output `outputs/openml_10fold_3seed_advanced.json`: 90 matched runs completed. Files were paired by task ID, official fold and inner seed; all 90 keys matched.
- After adding fold/seed CLI support and changing binary log-loss input to the positive-class probability: a fresh dev-only venv passed **16 tests** and Ruff; task 31/fold 2/seed 43 with advanced experts completed without the earlier probability warning (`outputs/openml_fold2_warning_check.json`).

## Blocked or incomplete

- A broad initial optional-dependency install failed with `MemoryError`; the smaller install succeeded. An OpenBLAS default-thread memory error and one transient Windows stack overflow occurred; single-thread reruns succeeded.
- `gh auth status` reported an invalid saved GitHub token; there is no configured remote, so private repo creation/push is awaiting reauthentication.
- No natural-domain holdout, trained router, TabPFN checkpoint run, API, dashboard, Docker build or public/private remote push was executed at this checkpoint.
- OpenML's `Public` dataset licence field is recorded literally; underlying redistribution and commercial-use rights were not independently verified.

## Not independently verified

- No model improvement, publication, production-readiness, fairness property or coverage guarantee under arbitrary shift is claimed. The one-fold results are descriptive pilots.
