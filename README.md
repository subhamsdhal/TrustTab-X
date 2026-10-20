# TrustTab-X

**Adaptive Tabular Foundation Models for Robust Prediction and Decision Intelligence Under Distribution Shift**

A research-first Data Science project developed for a Growtern Academy portfolio upgrade and for a reproducible academic/industry demonstration. Attribution to an organisation must accurately reflect its permission, supervision, project dates, and actual work.

> **Current implementation: research pilot, not the finished study.** Includes classical ML, independent model selection, split conformal prediction sets, a three-task real OpenML pilot, controlled stress tests, a drift monitor, tests, and machine-readable metrics. It does **not yet implement** a tabular foundation model, learned shift-aware routing, an API, or a validated enterprise benchmark. One-fold real-data results are descriptive only.

## Research question

Can an evidence-driven model-selection and abstention policy improve predictive reliability and risk–coverage trade-offs on previously unseen tabular datasets and distribution shifts compared with a fixed expert, without using test labels for adaptation?

**Primary study:** tabular binary classification; extension to multiclass/regression only after the primary protocol is stable.

## Current capability

- Four independent training purposes: `train`, `selection`, `calibration`, `test` (the test has IID and shifted cohorts).
- Reproducible classical binary classifiers: logistic regression, histogram gradient boosting, random forest.
- Selection based *only* on selection-split Brier score.
- Split conformal prediction sets calibrated only on an independent calibration split; non-singleton sets are treated as abstentions.
- Bonferroni-corrected two-sample Kolmogorov–Smirnov (KS) feature shift detector.
- IID/shifted evaluation with Brier, log loss, AUROC, accuracy, conformal coverage and abstention rate.
- Offline synthetic smoke experiment and three-task real OpenML pilot; JSON output; `pytest` and GitHub Actions CI.

## Quickstart

Requires Python 3.10+.

```bash
python -m venv .venv
# On macOS/Linux: source .venv/bin/activate
# On Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -e '.[dev]'
pytest -q
trusttabx --output outputs/smoke_baseline.json
```

**Important:** This initial experiment uses a generated **synthetic** dataset purely to verify plumbing. It does not demonstrate performance on real industry data.

## Working plan

Read [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) for the milestone roadmap, [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md) for research guardrails, and [`docs/CODEX_AUTOPILOT.md`](docs/CODEX_AUTOPILOT.md) for the autonomous end-to-end Codex prompt (or [`docs/CODEX_START.md`](docs/CODEX_START.md) for the smaller M1-only task). [`AGENTS.md`](AGENTS.md) provides instructions for coding agents.

## Executed real-data pilot (2026-10-20)

M1 now has a three-task OpenML-CC18 binary pilot using official outer test folds and independent fit, selection and conformal-calibration partitions. The optional M2 classical extension adds XGBoost and CatBoost; the M3 stress harness adds missingness, measurement noise and class-prior cohorts. These are one-fold, one-seed **pilots**, not a router or a validated shift-robustness claim. See the [task manifest](docs/OPENML_TASKS.md), [preliminary report](reports/RESEARCH_REPORT.md), [status](docs/AUTOPILOT_STATUS.md) and [verification ledger](RESULTS_VERIFICATION.md).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev,openml]'
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=.pytest_tmp
.\.venv\Scripts\trusttabx.exe benchmark --output outputs/openml_m1_pilot.json
# Optional, large classical extras: install xgboost and catboost, then:
.\.venv\Scripts\trusttabx.exe benchmark --advanced-experts --shift-lab --output outputs/openml_integrated_pilot.json
```

The OpenML cache stays in ignored `data/raw/openml/`. OpenML's literal `Public` dataset metadata is not independent commercial-use clearance. Current TabPFN default weights are separately licensed for non-commercial, non-production use and were not downloaded for this project.

## Datasets planned

- OpenML-CC18 classification benchmark tasks (retain task IDs, dataset versions and official splits; task-specific resource filters and preprocessing).
- Controlled synthetic covariate shifts, label shifts and missingness mechanisms (must be explicitly labelled as simulation).
- For external shifts: selected real domain-shift datasets with clear licensing and groups/domains, introduced in a separate milestone.

See the official [OpenML benchmarking suite](https://openml.github.io/docs/benchmark/) for its task and split conventions.

## Version- and licence-dependent foundation-model dependency

TabPFN is **not** installed or loaded by the baseline. As of the project planning date, the default TabPFN-3.5 model weights are under a **non-commercial** licence. The tabpfn *code* and model weights have distinct licences. Using the code or publishing model benchmarks does not grant unrestricted commercial deployment rights; confirm specific version/weights/terms before Growtern commercial use or a public enterprise demonstration. Research experiments and the commercially deployable service must be separately scoped until rights are resolved.

## Integrity

- No false precision, invented benchmarks, fabricated internships or publication claims.
- Report *where* the framework underperforms fixed experts.
- Split conformal coverage is **not guaranteed under arbitrary distribution shift**.
- The KS drift alarm is a statistical warning, not proof that individual predictions are incorrect or safe.
- Keep dataset-specific identifiers, preprocessing and evaluation protocols in the final report.
