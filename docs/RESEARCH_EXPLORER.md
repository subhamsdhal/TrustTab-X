# Research Explorer

The dashboard at <http://127.0.0.1:8000/> opens on a read-only Research Explorer. Its separate **Secure live demo** tab preserves authenticated ACSIncome predictions, model metadata, telemetry and drift monitoring. The public explorer exposes only aggregate, completed results and the existing research manuscript; it does not expose raw rows, model weights, API keys or live inference.

## What each view means

| Dataset | Shift chart | Policy comparison | Provenance |
| --- | --- | --- | --- |
| ACSIncome | ID test vs held-out geographic states | Mean across seven nested source-state domains | 2018 ACS PUMS via Folktables; `acs_income_2018_geographic_expanded.json` and M4.2 decision |
| Healthcare admissions | ID test vs held-out admission-source domains | Mean across four nested source domains | UCI Diabetes 130-US-hospitals; `tableshift_diabetes_12000.json` and M4.2 decision |
| Bank marketing | Near-future vs final-future windows (neither IID) | Mean across two nested forward source blocks | UCI Bank Marketing; `bank_temporal_2008_2010.json` and M4.2 decision |
| College Scorecard | One 2025 held-out for-profit cohort | Same final cohort, six policies | Public College Scorecard 2025 confirmation JSON |
| Online shoppers | November vs December shifted months (neither IID) | Each month separately, six policies | UCI shopper confirmation JSON |

The model-selection chart uses the five source-selection Brier scores for each dataset. It is separate from the final test charts. Policy names follow their saved results: source-selected fixed expert, uniform/validation-weighted ensembles, metadata and simplified routers for M4.2, and the distinct confidence expert and reliability router for College Scorecard. Do not compare absolute scores across families as if they shared a common target or prevalence.

Brier score, log loss and ten-bin expected calibration error (ECE) are displayed where the source JSON reports them. Missing ECE values are explicitly unavailable. The bars start at zero and show source precision in native SVG tooltips. Hover or focus-independent screen-reader labels provide exact values. The manuscript is served at `/research/manuscript` and copied into the local Docker image; no external chart service or CDN is used.

## Rebuild and verify

From the project root:

```powershell
.\.venv\Scripts\python.exe scripts/build_research_explorer.py
.\.venv\Scripts\python.exe -m pytest -q --basetemp=work/pytest-explorer
.\.venv\Scripts\python.exe -m ruff check .
docker compose --file compose.yaml --project-name trusttabx-local --env-file .env up -d --build --no-deps trusttabx
```

The builder reads six tracked `reports/results/*.json` files, verifies their pinned SHA-256 digests and locked M4.2 decision, then writes `src/trusttabx/m5_web/research_results.json` with provenance for every source. Tests compare the packaged values and browser chart geometry with their exact source values and reject changed source files. Never edit the packaged file by hand; regenerate it from the completed results.

The research decision stays **No adaptive advantage demonstrated** and the release stays **Local Demo Only**. The source families are few, shopper months are related, and College Scorecard is a single cohort. No arbitrary-shift reliability or production deployment claim follows from this interface.
