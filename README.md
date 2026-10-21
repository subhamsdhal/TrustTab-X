# TrustTab-X

**Trustworthy tabular machine learning under distribution shift: reproducible research and a secured local demonstration**

A research-first Data Science project developed for a Growtern Academy portfolio upgrade and for a reproducible academic/industry demonstration. Attribution to an organisation must accurately reflect its permission, supervision, project dates, and actual work.

> **Current implementation: research framework and secured local M5.1 decision-support demo.** The M4.2 research gate found no adaptive advantage; the ACSIncome service uses one source-selected fixed expert. It is **local demo only** until Docker runtime and internal deployment controls are verified. See [M5 architecture](docs/M5_ARCHITECTURE.md), [security policy](docs/M5_SECURITY.md) and [research decision](docs/M4_2_RESEARCH_DECISION.md).

## Final research and portfolio package (M7)

The [final portfolio directory](portfolio/final/README.md) includes a checked [research PDF](reports/paper/TrustTab-X-manuscript.pdf), [editable manuscript](reports/paper/TrustTab-X-manuscript.md), academic overview, experimental summary, architecture and industry case study, resume/interview materials and a recorded local demo. The [reproducibility guide](docs/M6_REPRODUCIBILITY.md) provides fast verification and full-rerun prerequisites; [claim traceability](docs/M6_CLAIM_TRACEABILITY.md) maps conclusions to tracked JSON. Six [scientific figures](reports/figures/manifest.json) regenerate from those JSON files with `python scripts/build_m6_figures.py`. The [final audit](docs/FINAL_PROJECT_AUDIT.md) records the release boundaries.

The fixed expert's bounded ACS ID/OOD Brier was **0.131644/0.145517**; the modified hospital study was **0.201603/0.250504**; bank near/later-future was **0.063753/0.337574**, alongside a large prevalence change. These are different datasets and split designs, not a pooled score. The router beat static selection in only **6/13 correlated outer source domains**, and the new shopper family missed the preregistered improvement margin over a weighted ensemble. M4.2 decision **C** retains the source-selected fixed expert. Neither modified ACS/hospital study is an official TableShift score.

For technical reviewers: [industrial case study](docs/M6_INDUSTRIAL_CASE_STUDY.md), [MSc research summary](docs/M6_MSC_PORTFOLIO.md), [placement and interview handoff](docs/M6_PLACEMENT.md), [M5 API](docs/M5_API.md), [verification ledger](RESULTS_VERIFICATION.md) and [private GitHub handoff](docs/GITHUB_PRIVATE_HANDOFF.md). The manuscript has not been peer reviewed or published. The service remains release **B: local demo only**; Docker runtime, TLS, managed identity and full data-rights/privacy governance are open.

## Research question

Can an evidence-driven model-selection and abstention policy improve predictive reliability and risk–coverage trade-offs on previously unseen tabular datasets and distribution shifts compared with a fixed expert, without using test labels for adaptation?

**Primary study:** tabular binary classification; extension to multiclass/regression only after the primary protocol is stable.

## Current capability

- Reproducible classical benchmarks with separate fit, selection, calibration and final test roles; natural geographic, site and temporal shift experiments with locked negative findings.
- Five fixed classical candidates for the bounded ACS task; source-selection Brier chooses one expert for M5.1 service inference. The research routers are disabled in the service because M4.2 found no adaptive advantage.
- Conformal, calibration, selective-risk and drift research diagnostics with their shift limitations documented in the research report.
- A versioned ACSIncome pipeline artifact, authenticated FastAPI predictions, local drift/delayed-label monitoring, structured privacy-conscious audit metadata and a browser dashboard.
- Offline synthetic smoke, OpenML and natural-domain aggregate results, 42 automated tests, static container checks and a manual synthetic Docker verification workflow. Docker runtime verification and internal release gates remain open.

## Quickstart

### Local M5 ACSIncome demonstration

On Windows PowerShell, from `D:\TrustTab-X`:

```powershell
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
.\.venv\Scripts\python.exe -m pip install -c constraints/m5.txt -e '.[dev,m5]'
.\.venv\Scripts\python.exe -m trusttabx.m5_train --output outputs/m5_acs_model --mlflow
$env:TRUSTTABX_TRUST_LOCAL_ARTIFACT='1'
$env:TRUSTTABX_MODEL_DIR='outputs/m5_acs_model'
$env:TRUSTTABX_APPROVED_MANIFEST_SHA256=(Get-FileHash -Algorithm SHA256 -LiteralPath outputs/m5_acs_model/manifest.json).Hash.ToLowerInvariant()
$env:TRUSTTABX_API_KEY = .\.venv\Scripts\python.exe -c 'import secrets; print(secrets.token_urlsafe(48))'
$env:TRUSTTABX_ADMIN_KEY = .\.venv\Scripts\python.exe -c 'import secrets; print(secrets.token_urlsafe(48))'
.\.venv\Scripts\python.exe -m uvicorn trusttabx.m5_api:app --host 127.0.0.1 --port 8000 --no-access-log --timeout-keep-alive 5
```

The trainer needs the cached public 2018 ACS source files or network access on its first run. Open <http://127.0.0.1:8000/> for the dashboard, then enter the operator key from your local environment. The key stays in browser-tab memory; reload clears it. Interactive OpenAPI is at <http://127.0.0.1:8000/docs>. The model bundle and local SQLite MLflow tracker stay under ignored `outputs/m5_acs_model/`. Set the trust flag and approved digest only for a bundle you trained or independently approved: `joblib` deserialization executes pickle content. No target test labels are used in M5 selection. [API examples](docs/M5_API.md), [artifact and monitoring contract](docs/M5_MODEL_MONITORING.md), [security controls](docs/M5_SECURITY.md).

Docker Compose is supplied for an operator with Docker installed: train or provide an approved bundle and keys first, then follow [Docker verification](docs/M5_DOCKER.md). It binds only `127.0.0.1:8000`; container execution has not been verified on this host because Docker is unavailable.

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
