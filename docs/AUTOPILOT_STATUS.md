# TrustTab-X autonomous build status

Checkpoint: 2026-10-20, Asia/Calcutta. Branch: `codex/trusttabx-autopilot` (local Git). This ledger describes executed work; no public deployment or publication occurred.

| Milestone | State | Evidence and remaining work |
| --- | --- | --- |
| M0 | **Done / reproduced** | Fresh project venv with Python 3.12.10; original 8 tests passed. `outputs/smoke_baseline.json`: synthetic logistic regression, IID coverage 0.91, shifted coverage 0.905, shifted feature alarm true. Synthetic metrics are plumbing checks. |
| M1 | **Done for the pinned CC18 subset** | OpenML-CC18 tasks 3, 31 and 10101 executed on all 10 official outer folds at inner seeds 42, 43, 44: 90 evaluations in `outputs/openml_10fold_3seed_classical.json`. Pinned dataset ID/version/name/target/MD5/licence metadata; exact split-index hashes and package versions in each record. Offline fixtures verify disjoint splits, train-only preprocessing and end-to-end evaluation. Natural-domain datasets and underlying dataset licence review remain. |
| M2 | **In progress** | Optional XGBoost 3.4.1 and CatBoost 1.2.10 ran with fixed 150-iteration configurations in matched 90-run `outputs/openml_10fold_3seed_advanced.json`; fit/inference time captured. Mean Brier changes versus three experts: task 3 0, task 31 −0.002394, task 10101 −0.002723. Five-expert Brier was worse in 5/30 credit and 5/30 blood paired runs. No equal-compute tuning, peak memory audit or foundation-model run. TabPFN 3.5 weights are non-commercial/non-production and were neither installed nor fetched. |
| M3 | **In progress** | Fixed MCAR 20% feature missingness, 0.5 train-SD numeric measurement noise and 70%-positive prior-resampled test cohort implemented and executed; see `outputs/openml_m3_shift_pilot.json`. Task 10101 prior cohort was 26 rows with coverage 0.731 versus IID 0.907. Feature corruption can change `p(Y|observed X)` and is not labelled pure covariate shift. No natural domain shifts, explicit concept shift, or severity sweep. |
| M4 | **Not started** | A learned router requires pre-registered source environments and genuinely held-out domains/datasets. The current three IID single-fold pilots do not justify training or claiming one. Next: expand the evaluation design before router fitting. |
| M5 | **In progress** | Split-conformal sets already existed. Added PR-AUC, 10-bin ECE, confidence-ranked risk–coverage points and AURC; 16 offline tests pass. `outputs/openml_integrated_pilot.json` contains real-task metrics. No repeated-task confidence intervals, class/group slices, post-hoc probability calibration or formal selective-risk guarantee. |
| M6 | **Not started** | No API, dashboard, Docker or hosted service. A demonstrator should wait for validated model/reliability choices. |
| M7 | **In progress** | Dataset cards, source-backed literature note, preliminary report and verification ledger added. No paper submission, video, resume claim or final benchmark report. |

## Audit and environment

- `D:\TrustTab-X` is the only project directory matching the required `AGENTS.md`, `pyproject.toml`, `src/` and `tests/`; no duplicate extracted project was merged or deleted. The original directory had no Git history; a local starter commit and working branch were created.
- The initial global-Python test attempt failed during collection because `trusttabx` was not installed. A later global-Python run passed 7 tests before sandbox temp creation failed. In the project venv, all original 8 tests passed.
- `pip install -e '.[dev,benchmarks]'` failed with `MemoryError` while unpacking large optional packages. Retrying `pip install -e '.[dev]' openml pandas` completed. XGBoost and CatBoost imports subsequently worked; exact versions are in the run JSON. The task cache is ignored under `data/raw/openml/`.
- The integrated pilot initially exited once with a Windows stack-overflow code before producing a task result. Single-task isolation and the full retry passed. One-thread `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` were used after an OpenBLAS thread-allocation failure.
- A second clean project-local venv (`work/clean-venv`, ignored by Git) installed only `.[dev]`; it passed **16 tests**, `ruff check src tests`, and the synthetic smoke command with the same 0.91/0.905 coverages. The subsequent 10-fold, three-seed runs still cover only three dataset-level units and cannot support broad cross-dataset significance.
- After the all-fold runs, the clean dev-only venv passed 16 tests and Ruff again. A task 31/fold 2/seed 43 advanced smoke run completed after changing binary log-loss to use the positive-class probability; no probability-sum warning appeared in that run.
- `gh auth status` reported an invalid saved token, and no Git remote is configured. Local work and commits can proceed; private remote creation/push requires reauthentication. No repository was published.

## Immediate next action

Follow `docs/NEXT_CODEX_TASK.md`: measure memory and compute, pre-register a genuinely natural held-out domain dataset, then revisit M4. Preserve the one-fold pilot and 10-fold results as historical evidence; do not use their test outcomes to tune subsequent model choices.
