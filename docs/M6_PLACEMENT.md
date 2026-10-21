# Placement and interview handoff

## Resume-ready project entry

**TrustTab-X — tabular ML reliability research and local API demonstration**. Developed leakage-aware natural-shift evaluations across ACS Income, hospital readmission and bank marketing, separating source fit, model selection, calibration and final tests. Evaluated Brier, calibration and source-fixed selective risk; documented a preregistered negative adaptive-routing result instead of claiming a model win. Built a versioned, digest-approved fixed-expert ACS artifact with authenticated local FastAPI inference, schema validation and drift/audit diagnostics. Verified 42 offline tests at M5.1 and measured loopback p95 latency of 37.25 ms in a 100-request local smoke. Container and internal production gates remain open.

Use this wording only if it matches the applicant's own contribution. The project context names Growtern Academy, but no employment, endorsement, dates or authorization are inferred here.

## Interview narrative

**Problem.** A model selected on source data can be badly calibrated after geographic or temporal changes. A router might exploit expert diversity but adds fit, selection and operating cost.

**Approach.** Define domain boundaries, independent split roles and a locked primary Brier criterion. Compare source-selected static, weighted and router policies in nested source domains; then use one preregistered new shopper family. Audit actual source-fixed 80% acceptance rather than retrospectively resetting the target cutoff.

**Result.** The router won only 6/13 correlated source domains; weighted family means were lower. It missed its 0.002 shopper improvement margin. Bank later-future Brier was 0.337574 versus 0.063753 near-future, alongside major prevalence change. The correct engineering choice was a fixed expert with monitoring, not adaptive deployment.

**Trade-off.** Five shopper candidate fits took 3.127 s summed versus 0.393 s for the selected random forest. That one-run model-fit comparison does not include end-to-end cost. The service is a local demonstration pending Docker and governance controls.

## Demo path and questions

Start with the [README local quickstart](../README.md#local-m5-acsincome-demonstration), show health, one authenticated prediction, schema rejection and the local dashboard using [API examples](M5_API.md). Keep keys private and use only the approved local bundle. Show the [paper figures](../reports/paper/TrustTab-X-manuscript.md), especially the router loss and source-fixed threshold marks. Explain why target-ranked risk–coverage curves cannot set a deployed threshold; why three families do not justify a narrow confidence interval; why bank prevalence change complicates interpretation; and what is still required for release B to become an approved internal service.
