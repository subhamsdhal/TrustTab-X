# TrustTab-X — research-to-industry implementation plan

**Project identity:** advanced Data Science and statistical ML under Growtern Academy (permission and attribution as represented by the student; dates and completed achievements must be truthful).

**Scope:** binary classification first, reproducible multi-dataset benchmarks, shift-aware model choice and selective prediction; industry demonstrator afterward.

## Research hypotheses and contributions

- **H1 — reliability under shift:** Does a validation-trained, shift-aware expert selection/routing policy improve out-of-domain Brier/log loss and selective prediction **versus the best static expert chosen without test labels**?
- **H2 — calibrated uncertainty:** Do conformal/selective strategies meet stated marginal coverage in IID settings, and how do coverage and abstention change under distribution shift?
- **H3 — generalisation:** Does any improvement survive held-out dataset/domain evaluation, budget controls and ablation against simpler, fewer-parameter alternatives?
- **H4 — deployment:** Can the resulting prediction service provide repeatable, auditable decisions and alert when uncertainty or input drift warrants review?

Novelty is a *hypothesis*, not an assumed property of using TabPFN, conformal prediction or routing. Conduct a literature review before claiming a publishable contribution.

## Sequence and deliverables

| Milestone | Focus | Output | Gate |
|---|---|---|---|
| M0 (delivered in starter) | Repo, leakage-safe four-way splits, fixed model selection, baseline uncertainty, controlled covariate shift, unit tests | Python package, JSON baseline, CI | All tests pass, result reproducible |
| M1 | Real-data benchmark (OpenML task IDs and public enterprise-like datasets), categorical preprocessing inside training folds | Dataset cards, versioned manifests, standard benchmark | No leakage; reproduce official tasks |
| M2 | TabPFN adapter plus XGBoost/CatBoost/HistGB/LogReg, common probability interface; resource budget tracking | Per-dataset, per-model benchmark reports | Fair tuning budget, licensing check |
| M3 | Shift harness: covariate, missingness, prior/label and selected real domain shifts; dataset-specific shift semantics | Controlled shift pairs, diagnostic report | Train/test splits and shift severity fixed |
| M4 | Main research: train shift-aware strategy/metadata router on *source* validation environments; compare against static best, simple threshold, oracle upper bound | Tested learned routing with OOD abstention, ablations | Held-out dataset/domain & no target-label leakage |
| M5 | Reliability: calibrated probabilities, conformal strategies, selective risk/coverage, class-wise diagnostics, uncertainty and resource audits | Research-grade evaluation suite | Marginal guarantees correctly scoped |
| M6 | Enterprise demo: scoring API, data validation, drift monitor, audit events, MLflow runs, Docker, documentation | Runnable service/demo, CI security baseline | Functional integration tests |
| M7 | Dissemination: research report, plots, failures, compute and licences, demo video, accurate internship portfolio | Reproducible artifact, presentation, resume bullets | Re-runs from pinned versions and recorded seeds |

**Suggested planning horizon:** approximately 12–16 weeks of serious part-time work, dependent on data and GPU availability; not a completion promise. Do not speed past M1/M2 and assume a result.

## Proposed tech stack

- Core: Python, NumPy, SciPy, Pandas, scikit-learn.
- Traditional experts: XGBoost, CatBoost, LightGBM if licensing/environment permit.
- Research-only foundation model: TabPFN with exact model version and checkpoint license recorded.
- Experimentation: OpenML, MLflow, Python test harness, lint/type checks.
- Service: FastAPI + Pydantic, Docker, structured logging, OpenTelemetry/Prometheus in later phase.
- Report: reproducible notebooks/scripts, figure-generation code, paired dataset-level uncertainty intervals.

## Success criteria (research, not arbitrary target numbers)

1. Strong baselines and a held-out evaluation protocol exist before any novel component is trained.
2. Proposed adaptive approach beats a well-chosen fixed baseline on meaningful *held-out* settings, or negative results are transparently reported.
3. Selective risk is measured at matched coverage/abstention rates (don't mislead by abstaining on all difficult samples).
4. Compute, inference latency, model weights licenses and operational cost are reported.
5. Research claims are reproducible with commands, dataset versions, seeds and reference citations.

## Resume/portfolio framing

**Only after implementation:** 'Designed an adaptive, reliability-aware tabular ML evaluation framework at Growtern Academy; benchmarked classical and foundation models across public datasets and controlled shifts; measured selective prediction, drift and calibrated uncertainty; deployed reproducible prediction service.'

Insert quantitative gains only if actual final benchmarks support them. Never imply the company collected the public datasets or endorsed an academic publication unless it did.
