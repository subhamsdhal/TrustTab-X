# Guided local portfolio demonstration

## Reproduce the recorded demonstration

Run from the project root with the existing approved `outputs/m5_acs_model` bundle and cached public ACS 2018 source files. The script rechecks pipeline/split/source-file SHA-256 values against tracked records, generates new secrets in process memory, starts FastAPI through `TestClient`, and writes only aggregate evidence to `reports/results/m7_local_demo.json`. It does not expose a network service or write records, labels, keys or a model binary to the portfolio.

```powershell
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
.\.venv\Scripts\python.exe scripts\run_m7_local_demo.py 2> outputs\m7_demo_audit.log
```

The audit log path is Git-ignored. Do not display secrets or raw ACS rows in a portfolio presentation. An optional interactive loopback dashboard launch is documented in the project README and requires separately generated keys and an approved digest.

## Five-minute walkthrough

1. **Identity and artifact (45 s).** Show the aggregate record's model version `m5-acs-fixed-1`, `hist_gradient_boosting`, schema `acs-income-2018-v1`, pipeline digest and passing source/split hash checks. Explain why the approved manifest digest is supplied separately from the bundle before deserialization.
2. **Authentication, dashboard and prediction (60 s).** `/health` and the local dashboard returned 200. `/model` without a key returned 401; with an operator key it returned the fixed model identity. One real public ACS source ID-test row returned class 0 and probability 0.0059174104. No source record or key is exported. The 100-record batch is a feature-only filtered, service-eligible subset because unknown categorical codes are rejected by design.
3. **Drift (60 s).** The first 100 deterministic target rows were all CT under the state ordering. `/monitor/drift` flagged feature and probability shift; AGEP flagged, WKHP did not. No CT labels were used. The endpoint explicitly said realized Brier, calibration and accuracy were unknown without labels. This one-state, small-batch illustration is **not** a population-wide or statistically calibrated drift finding.
4. **Monitoring, audit and delayed labels (60 s).** The operator key received 403 on the admin-only delayed-label endpoint. The admin call on the first 100 schema-accepted, household-disjoint source ID-test rows produced descriptive Brier 0.0997003, with observed positive rate 0.24. This selected demo subset is **not** the locked full ID score or new validation. In-memory telemetry counted 101 successful predictions and two intentional authorization errors; privacy-conscious audit lines remain only in the ignored local log.
5. **Research decision (75 s).** Show the manuscript's fixed/router figure and the source-fixed threshold markers. M4.2 found no adaptive advantage; the router is research-only. The local service is release B. Docker, TLS, managed identity, audit governance and source-data rights approval remain open.

The [aggregate demo record](M7_LOCAL_DEMO.json) contains the exact machine-produced outputs; do not substitute this walkthrough for the original evidence or a production readiness assessment.
