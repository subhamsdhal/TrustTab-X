# Next Codex task — expand the research protocol before router training

## Next executable task after ACSIncome

Read `docs/ACS_INCOME_DOMAIN_MANIFEST.md`, `docs/TABLESHIFT_DOMAIN_MANIFEST.md`, `docs/M4_RESEARCH_SPEC.md` and both final JSON outputs. Pre-register a further **independent public dataset family or temporal/site shift** with its own source environments; alternatively design nested leave-one-division-out ACS meta-validation across several held-out divisions while reserving New England untouched, but do not count states from one division as independent dataset-level replicates. For a cross-dataset router, standardise only comparable metadata (expert probability, entropy, inter-expert disagreement, source-validated loss and feature-shift summaries); do not merge ACS and hospital raw columns. Use domain-held-out source predictions for router fitting, separate source calibration, static and ensemble controls, and group/domain-level intervals. Keep the New England and hospital final tests out of all router decisions. Re-run `pytest -q` and `ruff check src tests`, then commit only verified work.

## Next executable task after the natural-domain checkpoint

Read `docs/TABLESHIFT_DOMAIN_MANIFEST.md`, `docs/M4_RESEARCH_SPEC.md` and `outputs/tableshift_diabetes_12000.json`. Obtain a **second independent public natural-domain benchmark** with defensible group/time splits and licensing, then pre-register multi-environment source training and held-out domain evaluation. Preserve patient disjointness. Quantify model redundancy further without target test tuning. Run repeated seeds/domains and grouped uncertainty intervals before a learned router. The current comparison charges five shared fits to both selectors; a separate deployment cost frontier for actually using only three models remains incomplete. Do not claim an exact TableShift score.

Run `./.venv/Scripts/python.exe -m pytest -q; ./.venv/Scripts/ruff.exe check src tests`. Reproduce with `./.venv/Scripts/trusttabx.exe natural-benchmark --max-fit-rows 12000 --output outputs/tableshift_diabetes_12000.json`.

Start in `D:\TrustTab-X` on `codex/trusttabx-autopilot`. Read `AGENTS.md`, `docs/EXPERIMENT_PROTOCOL.md`, `docs/OPENML_TASKS.md`, `docs/AUTOPILOT_STATUS.md` and `reports/RESEARCH_REPORT.md`. Run the full offline suite and lint first. Set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` on this constrained Windows host.

1. The benchmark CLI now supports all 10 official CC18 outer folds and fixed seeds 42/43/44; both three- and five-expert 90-run files are saved. Add resumable per-fold output and independent validation of run-file completeness. Keep each fold's official test rows untouched by its own fit, selection and calibration steps. Construct uncertainty intervals at the task level or with properly grouped resampling, not independent-row confidence intervals.
2. Pre-register one or more public datasets with documented site/time/group boundaries and terms allowing research use. Implement the natural split faithfully. Investigate potential entity leakage in the current three tasks and the weak `Public` licence field before any redistribution or service use.
3. Define a finite, resource-matched tuning budget for classical experts, capture peak memory, and decide whether to run a **research-only** TabPFN comparison after reviewing the exact checkpoint terms, authentication and CPU/GPU cost. Keep non-commercial weights out of service artifacts.
4. Only after these source/target environments exist, train a source-only router and compare with validation-selected static expert, confidence threshold and ensemble at matched coverage and compute. Hold out target domains/datasets; report negative results and ablations.
5. Extend M5 reliability with calibration and grouped intervals, then begin M6/M7 only with model and usage boundaries established. Update `docs/AUTOPILOT_STATUS.md` and `RESULTS_VERIFICATION.md`; commit verified work on the branch.

Current exact commands:

```powershell
cd D:\TrustTab-X
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=.pytest_tmp
.\.venv\Scripts\ruff.exe check src tests
.\.venv\Scripts\trusttabx.exe benchmark --advanced-experts --shift-lab --output outputs/openml_integrated_pilot.json
```

Do not treat the task 31 test result as a reason to tune on that test fold. Keep the old output as a pilot record, then run a newly pre-registered expanded experiment.
