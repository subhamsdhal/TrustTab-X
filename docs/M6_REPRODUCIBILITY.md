# M6 reproducibility guide

Run commands from `D:\TrustTab-X` on branch `codex/trusttabx-autopilot`. Python 3.10+ is required. The development environment for this checkpoint is Windows PowerShell with the committed `constraints/m5.txt`; set BLAS thread limits on this constrained host. Raw archives, fitted bundles and caches remain ignored. No command below silently fetches a raw natural-domain dataset merely to regenerate the paper figures.

## Fast independent checks from tracked evidence

```powershell
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
.\.venv\Scripts\python.exe -m pip install -c constraints/m5.txt -e '.[dev,m5]'
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=.pytest_tmp
.\.venv\Scripts\ruff.exe check src tests scripts
.\.venv\Scripts\python.exe scripts\build_m6_figures.py
.\.venv\Scripts\python.exe scripts\verify_m6.py
git diff --check
```

`scripts/verify_m6.py` checks the current figure input hashes, expected recorded metrics, manuscript-relative links and figure presence. It does not rerun model training. The exact tracked aggregates are in `reports/results/`; `reports/figures/manifest.json` records SHA-256 after CRLF-to-LF normalization of JSON inputs so Git checkouts on different platforms agree. The manuscript is `reports/paper/TrustTab-X-manuscript.md`.

The complete manuscript is maintained in Markdown. M7 generated a checked A4 PDF from it using ReportLab, without Pandoc or TeX. On this host the Codex bundled Python runtime includes `reportlab`, `pypdf` and Poppler. From the project root, use the bundled Python path returned by `load_workspace_dependencies`:

```powershell
& 'C:\Users\hp\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\build_m7_pdf.py
pdfinfo reports/paper/TrustTab-X-manuscript.pdf
```

On another Windows host, install ReportLab in a local Python environment and run `python scripts/build_m7_pdf.py`; the builder uses local Cambria/Calibri fonts. Render every PDF page with Poppler for visual QA. If Pandoc and TeX are already installed, an alternative build is:

```powershell
pandoc reports/paper/TrustTab-X-manuscript.md --from markdown --pdf-engine=xelatex --resource-path=reports/paper -o reports/paper/TrustTab-X-manuscript.pdf
```

Inspect the rendered PDF before distributing it; the Markdown manuscript remains the editable source of record.

For the real local service demonstration on cached public ACS rows, run `python scripts/run_m7_local_demo.py` with the project `.venv` and one-thread BLAS settings. It requires the already approved local artifact and cached source archives, rechecks tracked provenance, and writes only aggregate results to `reports/results/m7_local_demo.json`. It does not silently fetch data for the portfolio. `python scripts/build_m7_portfolio.py` assembles `portfolio/final/` and its SHA-256 manifest after the paper and audit are current.

## Full experiment prerequisites

The complete natural-domain study requires public upstream archives, documented in [ACS](ACS_INCOME_DOMAIN_MANIFEST.md), [hospital](TABLESHIFT_DOMAIN_MANIFEST.md), [bank/source protocol](M4_2_SOURCE_PROTOCOL.md), and [OpenML task manifest](OPENML_TASKS.md). Preserve their archive hashes, task versions, source/target domains and split hashes. Place raw data only in ignored `data/raw/` paths specified by each manifest; inspect usage rights before redistribution or a service. The results in this repository are locked point observations from the recorded runs. Do not adjust a model after inspecting a final target and call the same target independent confirmation.

The command-line experiment modules and their arguments are defined under `src/trusttabx/`; invoke `--help` before a full rerun. For the original synthetic smoke and OpenML CC18 pilot:

```powershell
.\.venv\Scripts\trusttabx.exe --output outputs/smoke_baseline.json
.\.venv\Scripts\python.exe -m pip install -c constraints/m5.txt -e '.[dev,openml,m5]'
.\.venv\Scripts\trusttabx.exe benchmark --output outputs/openml_m1_pilot.json
```

The natural benchmark commands and per-run source versions are preserved in the corresponding manifest and `RESULTS_VERIFICATION.md`. Full natural reruns are substantially more expensive than figure regeneration; do them only after confirming source access, package versions and compute budget. For the ACS service artifact and loopback demo, use the [README quickstart](../README.md#local-m5-acsincome-demonstration) and [API contract](M5_API.md). Docker verification remains a separate open gate under [M5 Docker](M5_DOCKER.md); it was not run on this host.

The core natural-domain entry points, with raw archives already placed at the paths shown, are:

```powershell
.\.venv\Scripts\trusttabx.exe acs-benchmark --cache-dir data/raw/acs2018 --seed 264738 --max-fit-rows 14000 --source-per-state 6000 --target-per-state 3000 --output outputs/acs_income_2018_geographic_rerun.json
.\.venv\Scripts\trusttabx.exe natural-benchmark --archive data/raw/tableshift/dataset_diabetes.zip --seed 264738 --max-fit-rows 12000 --output outputs/tableshift_diabetes_rerun.json
.\.venv\Scripts\trusttabx.exe bank-benchmark --archive data/raw/bank/bank-additional.zip --seed 264738 --output outputs/bank_temporal_rerun.json
```

These commands use current CLI defaults plus explicit core settings. Before comparing a rerun to the locked JSON, confirm every recorded split hash, source archive hash, package version and protocol setting. The later nested M4.2 and shopper confirmation are controlled experiments documented in `docs/M4_2_SOURCE_PROTOCOL.md` and `RESULTS_VERIFICATION.md`; run them from the frozen source procedure with a separate output path, never overwrite the tracked final record or reuse inspected final labels for policy selection.

## Research integrity checklist

- Fit, selection, probability calibration, acceptance-threshold calibration and final evaluation have distinct roles. Household/patient disjointness and forward bank ordering are recorded in the source manifests.
- The ACS and hospital studies use modified bounded protocols, not official TableShift scores. Bank near/final are both future periods.
- The 13 M4.2 outer source domains are nested observations in only three original families; do not treat them as independent dataset replicates. November and December shopper results are one additional family.
- Source-fixed thresholds can miss 80% target coverage. Target-ranked risk–coverage curves are descriptive and cannot set operational thresholds.
- The router failed the M4.2 gate and is research-only. M5.1 is a local demo, not a production release.
