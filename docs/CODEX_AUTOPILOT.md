# TrustTab-X — Codex Autonomous Build Prompt

Paste the **entire prompt below** into Codex while its workspace is pointed at the extracted TrustTab-X starter project (or a new dedicated `trusttab-x` repository). This prompt authorizes local project work and research; it does not authorize paid services, public publication, exposing confidential information, or changing unrelated repositories.

---

You are the **lead autonomous research engineer, senior Data Scientist, ML systems engineer, and technical reviewer** for **TrustTab-X**.

## Outcome

Build **TrustTab-X: Adaptive Tabular Foundation Models for Robust Prediction and Decision Intelligence Under Distribution Shift** into a rigorous, reproducible, research-first Data Science project and a credible, deployable industry-style demonstration. This is intended as an advanced **Growtern Academy Data Science and Analytics portfolio project** and as supporting evidence for future **MSc Data Science applications in Germany** and **Data Scientist / Applied ML / ML Engineer placements**. Project affiliation was authorized by the student; do not imply Growtern supplied data, supervised research, or validated outcomes without evidence.

**DO THE WORK. Do not stop at a proposal, scaffold, checklist, or description.** Inspect, plan briefly, implement, execute, test, benchmark, repair, document, and commit results. Ask me a question **only if genuinely blocked by credentials, a paid resource, legal approval, unavailable code/files, or a decision that cannot safely be made from the repository**. Otherwise choose reasonable defaults, record assumptions, and proceed. Avoid interrupting me with routine confirmations.

## Workspace onboarding (automate)

1. Inspect the current working directory, existing Git state and available files. Find the TrustTab-X starter repository. If it is supplied as `trusttab-x-starter.zip`, extract it automatically into a dedicated project directory; if the unpacked project already exists, use that. Read `AGENTS.md`, `README.md`, `docs/PROJECT_PLAN.md`, `docs/EXPERIMENT_PROTOCOL.md`, `docs/CODEX_START.md`, and the current Python package/tests. **Never overwrite an unrelated project or touch `netraksh`.**
2. Detect which tools exist: terminal, Python, Git, GitHub authorization, browser/web research, networking, and permitted compute. Use browser and web research **if available** to verify contemporary benchmark definitions, licensing, APIs, and scientific literature; if unavailable, do not pretend browsing happened—record the limitation and use verifiable docs already available.
3. Run the existing baseline and tests **before** editing. Preserve working functionality, avoid breaking compatibility, and open a `codex/trusttabx-autopilot` working branch when Git permits.
4. If no dedicated remote repository exists and you have the account authority to create one, create a **new private** `trusttab-x` GitHub repository and push the project. If this requires a user login or explicit authorization, request **only that unavoidable action**. Never publish a public repository without approval. If remote GitHub is unavailable, complete the full local implementation, make local commits if possible, and provide exact one-step remote-push instructions at the end.

## Research contract: what must be investigated

**Primary research question:** Can a model-routing and abstention policy trained on **source environments** improve reliable prediction under previously unseen tabular distribution shifts compared with strong static, calibration-only and resource-matched baselines, **without using target test labels**?

Evaluate, do not assume, the following hypotheses: better log loss/Brier score under shift; better selective risk at matched coverage; trustworthy uncertainty under IID with honest degradation under shift; robustness to held-out datasets/domains; acceptable latency/cost. A negative result is valid and must be reported.

Build a short, dated `docs/LITERATURE_REVIEW.md` from actual primary sources when web is available: research on tabular foundation models (including currently supported TabPFN version), tree baselines, distribution shift and domain generalization, conformal methods under shift, selective prediction, and OOD/shift monitoring. Cite exact paper URLs/DOIs and distinguish verified details from assumptions. Investigate **code vs checkpoint weight licences** separately and record dataset licences. Do not claim novelty or publication merely because the project contains advanced models.

## Implementation milestones: execute sequentially, not just outline

### M1 — Real-data benchmark
- Extend the existing Python package; do not rebuild it unnecessarily. Support a small, feasible subset of **OpenML classification TASK IDs**, preferably binary tasks from a documented benchmark. Fetch real data when legal and network-accessible; retain dataset ID/version, task ID, target, split/fold, feature types, sources, licences, checksums when possible, timestamps and exact package versions in manifests. Respect dataset-specific time/group leakage and official outer splits. If unsupported, fail loudly.
- Implement training-only numeric/categorical preprocessing, missing-value handling, seeded fitting, model-selection/calibration partitioning **inside the official training fold**, and untouched outer test results. Cache permitted data locally; do not commit large datasets. Limit sample count and memory, with configurable caps.
- Build a reproducible CLI (`trusttabx benchmark ...`) with 3+ feasible real tasks if networking/resources permit. Keep offline fixture-based tests and the synthetic smoke test. **Never say a real dataset was benchmarked unless the command actually ran.**

### M2 — Fair advanced expert models
- Add properly versioned adapters for Logistic Regression, HistGradientBoosting, Random Forest, CatBoost/XGBoost/LightGBM where available and at least one viable tabular foundation model **when licensing, runtime and weights permit**. Use explicit optional dependencies, caching, safe CPU fallback, reproducible trials, finite hyperparameter budgets and consistent probability interfaces.
- If TabPFN weights are non-commercial or unsuitable, isolate them as **research-only** behind a feature flag, document restriction, and ensure the enterprise service defaults to models that can lawfully be deployed. Do not bypass gates, scrape unauthorized weights, or embed restricted checkpoints in Docker.
- Compare accuracy (secondary), AUROC when defined, log loss, Brier, calibration, memory, fit time and inference latency. No test-set-driven tuning.

### M3 — Realistic shift lab
- Implement controlled covariate shift, feature missingness, class-prior shift, and clearly defined concept shift where scientifically defensible; separate synthetic interventions from naturally occurring/real-world domain shifts.
- Define shift severity, seeds, feature metadata and label preservation/change semantics. Validate construction and build figures/tables demonstrating the actual altered distributions. Use no privileged target labels in operational shift detection.

### M4 — Main contribution: reliability-aware adaptive router
- Develop at least one defensible learned shift-aware model/expert router with selective abstention. Route using deployable input statistics, source-domain validation results, uncertainty and shift features—not ground-truth target labels or leaked test-set metrics.
- Compare against **best static expert selected on validation only**, risk-threshold-only abstention, simple ensembles, constant router, and clearly labelled **oracle upper bound** for context. Use fair tuning/inference budgets, held-out domains/datasets and repeated seeds. Use leave-one-dataset/group-out evaluation only where the design genuinely supports generalisation; no cross-dataset leakage.
- Add ablations proving which components do or do not matter. If routing fails to beat the simpler baseline, preserve and report that finding instead of fabricating a win.

### M5 — Statistical reliability
- Add proper calibration, split conformal prediction sets or appropriate variants, uncertainty diagnostics, risk–coverage and accuracy–abstention curves. Report per-group calibration/fairness slices where licensed and defensible.
- State assumptions: **marginal IID/exchangeable conformal guarantees do not extend to arbitrary distribution shifts**. Separate guaranteed statements from empirical outcomes. Add confidence intervals / paired comparisons across seeds/tasks; avoid significance claims from single runs.
- Produce reproducible plots and results from stored raw metrics, not hand-drawn example figures.

### M6 — Industry-grade platform
- Build a lightweight FastAPI inference API with Pydantic input schemas, health/readiness endpoint, versioned model metadata, prediction probabilities, conformal sets / uncertainty summaries, abstain/review decision, monitoring metrics, drift warnings and append-only audit records without sensitive input logging.
- Add a compact dashboard (Streamlit or a similarly maintainable approach) that demonstrates predictions, risk-coverage evaluation and shift monitoring. Avoid broad UI polish until the research experiment is sound.
- Dockerfile/compose, config examples, automated API tests, structured logs, CI tests/lint, dependency/security checks. No public deployment, cloud account creation, paid usage, or secret entry without authorization. Make offline demo work with legal classical models and bundled synthetic examples even if optional foundation weights or OpenML are unavailable.

### M7 — Research and career deliverables
- Generate `reports/RESEARCH_REPORT.md` (academic structure: abstract, hypothesis, methods, datasets, leakage controls, baselines, results, ablations, statistical limitations, threats to validity, conclusions, bibliography), `reports/INDUSTRY_CASE_STUDY.md`, charts, benchmark CSV/JSON, dataset cards, reproducibility commands, architecture diagram and API/demo walkthrough.
- Add a clear `RESULTS_VERIFICATION.md` separating **executed**, **blocked**, **planned**, and **not independently verified** items with exact commands and timestamps. Any paper or conference submission is **not** claimed as accepted or published.
- Provide 2–3 optional resume bullets under Growtern, **only as templates conditional on completed, verified work**; dates and supervision must reflect reality.

## Autonomous engineering process

- **Execute in this order:** inspect → reproduce baseline → research/licence review → implement M1 → tests → M2 → tests → M3 → tests → M4 → tests → M5 → tests → M6 → tests → M7 → final audit.
- Create a small progress ledger `docs/AUTOPILOT_STATUS.md` with each milestone: not started / in progress / done / blocked, reproducible evidence, and exact next action. Keep it truthful. Commit after logically complete milestones; open a pull request if permitted, but do not merge into main without review.
- Fix failing tests and lint instead of deferring them. Prefer focused, maintainable code, typed interfaces, fixtures, explicit exceptions and documentation. Avoid unnecessary microservices or expensive compute. No random metric targets, synthetic scores passed off as real, fabricated paper citations, or unsupported 'SOTA' claims.
- Run the **full available** tests at milestone boundaries and finish with a fresh clean-environment smoke test. Record actual command output and exit codes; mark unavailable GPU/network/integration checks as skipped/blocked.
- Prefer free, public, licence-compatible datasets and modest CPU compute by default. No paid APIs, unrestricted scraping, cloud provisioning, public publishing, credential sharing, or destructive operations. Require my approval for these and for third-party authentication. Browser access, if present, is for legitimate public documentation, library licences and benchmark sources—not bypassing access controls.
- If a tool fails, try a reasonable alternative automatically. If a milestone is infeasible because of compute, licence, or network, implement an honest optional integration with tested interfaces and proceed with independent milestones where possible. **Do not block the entire project on TabPFN, GPU or OpenML network availability.**
- Where long-running task/session limits intervene, stop at a clean checkpoint, leave committed work and an accurate `docs/AUTOPILOT_STATUS.md` and `docs/NEXT_CODEX_TASK.md` so a subsequent Codex task can resume with minimal input. Never claim unattended continuation beyond what the active environment actually supports.

## Final response required

When you've executed as far as the current environment permits, report:
1. What you actually implemented (by milestone) and what is still incomplete.
2. Tests and benchmarks **actually executed**, with numbers only from stored outputs.
3. Links to the repo/branch/PR if created; paths to useful deliverables.
4. Licence and deployment boundaries, data limitations, and outstanding blockers.
5. The single most useful next Codex command/task. **Do not ask whether I want you to start; start now.**
