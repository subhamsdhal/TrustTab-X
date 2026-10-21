# Results verification — 2026-10-20

## 2026-10-21 M4.1 reliability checkpoint

Verified the checkout began at `d4dbb80` with a clean tree. Re-executed the locked exploratory router without changing its protocol; ACS, hospital and bank static/router Brier pairs reproduced exactly as 0.145517/0.148039, 0.250504/0.258926 and 0.337574/0.344292. The rerun is ignored under `outputs/router_prototype_m4_1_reproduction.json`; the earlier tracked result remains intact.

Preregistered the new College Scorecard family before opening outcome rows, downloaded and SHA256-pinned the public archive, then executed the fixed source/for-profit-target benchmark. Tracked aggregate outputs are `reports/results/college_scorecard_2025_confirmation.json`, `reports/results/m4_1_source_router_diagnosis.json` and `reports/results/bank_temporal_diagnosis.json`. The source-only family-held-out diagnosis reads only named training-family cross-fit arrays when fitting each router. The confirmation run uses a new 228-institution target. The router did not beat static RF on primary Brier; probability calibration did not help. No independent-environment confidence interval, official TableShift score, prospective education prediction, or 80% target selective-coverage success is verified.

Final local checks: `./.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --basetemp=.pytest_tmp` **29 passed**; `./.venv/Scripts/ruff.exe check src tests` passed; `git diff --check` passed. OpenBLAS, OMP and MKL were set to one thread for experiments. No raw records were committed or sent to a service.

## 2026-10-21 temporal and exploratory-router run

Executed `trusttabx bank-benchmark`, `trusttabx domain-crossfit --dataset bank`, `--dataset acs`, `--dataset diabetes`, and `trusttabx router-prototype` with BLAS/OpenMP threads set to 1. The bank archive was SHA-256 pinned and all earlier results preserved. Source cross-fit prediction archives contain source labels and remain Git-ignored; the aggregated bank, source-fold and router JSON files are tracked under `reports/results/`. The router did not improve Brier over the fixed expert on any of three final domains. No new-family router transfer, target adaptation, distribution-free coverage under shift, or independent statistical significance was verified. Final audit: 25 offline tests passed, Ruff passed, and `git diff --check` passed.

## ACSIncome geographic pilot

Executed the nine-state smoke and initial full run, preserved as ignored `outputs/acs_income_smoke.json` and `outputs/acs_income_2018_geographic.json`. Expanded to seven source states in five divisions plus six untouched New England states, then reran after making per-state sampling stable and hashing source files. Final reproducible output: ignored `outputs/acs_income_2018_geographic_expanded.json`, with an aggregate-only tracked copy in `reports/results/`. The earlier hospital result also has an aggregate-only tracked copy there. This is a TableShift-inspired bounded protocol, not an official score. The final rerun passed 19 offline tests, Ruff and `git diff --check`. No learned router or cross-dataset improvement was verified. See `docs/ACS_INCOME_DOMAIN_MANIFEST.md` for exact definitions and metrics.

## Natural-domain verification

Executed `./.venv/Scripts/trusttabx.exe natural-benchmark --max-fit-rows 12000 --output outputs/tableshift_diabetes_12000.json` successfully against the SHA-256-pinned public UCI archive. This is a patient-disjoint TableShift-derived variant, not an official TableShift score. The result file contains ID/OOD metrics and resource records. Executed full `pytest -q`: 18 passed; `ruff check src tests`: all checks passed. No learned router, TabPFN comparison, official TableShift package execution or repeated natural-domain evaluation has been verified. Benchmark data and JSON output remain Git-ignored local artifacts.

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
