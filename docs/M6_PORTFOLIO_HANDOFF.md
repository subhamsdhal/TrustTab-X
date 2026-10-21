# M6 academic and placement portfolio handoff — draft

## Accurate project description

TrustTab-X is a reproducible tabular reliability research framework with a secured **local** ACSIncome 2018 fixed-expert decision-support demonstration. The M4.2 nested and additional-family evidence did **not** demonstrate an adaptive-routing advantage; adaptive routers remain research prototypes. The M5.1 service uses a source-validation-selected histogram gradient boosting pipeline, versioned and hash-checked artifacts, independent operator approval, API-key roles, schema validation, local monitoring and a dashboard. It is **local demo only**, not a production decision system or validated improvement over static baselines.

## Evidence suitable for a report or interview

- Research methods and negative results: `docs/M4_2_RESEARCH_DECISION.md`, `reports/RESEARCH_REPORT.md` and tracked aggregate result JSONs. Describe dataset boundaries, split isolation, limited independent families and target-label restrictions.
- Engineering artifact: `reports/results/m5_acs_artifact_summary.json`, `docs/M5_ARCHITECTURE.md`, `docs/M5_API.md`, `docs/M5_MODEL_MONITORING.md` and `docs/M5_SECURITY.md`. Raw ACS records and fitted model binaries are ignored.
- Quality evidence: M5.1 42-test offline suite, lint, dependency check, post-upgrade vulnerability audit and measured authenticated local HTTP smoke. Exact conditions and limits are in `RESULTS_VERIFICATION.md`.
- Deployment limitation: synthetic container test workflow exists, but Docker was unavailable on the development host; image build/run and Linux artifact compatibility remain unverified. Internal deployment additionally requires TLS/managed identity, log governance, dataset-use/privacy review and security approval.

## Suggested honest portfolio wording

“Built a reproducible tabular ML research framework and a locally secured ACSIncome decision-support API with split-safe fixed-expert selection, versioned model artifacts, drift diagnostics and negative-result reporting. Implemented API contract/security tests and measured loopback inference. Adaptive routing did not show a consistent advantage; container and internal deployment gates remain open.”

Do not call this a deployed enterprise service, claim selective guarantees under shift, claim TabPFN production use, or present the router as a demonstrated improvement. The next M6 deliverable is a polished research methods/results narrative and portfolio presentation grounded in the locked evidence, after the Docker gate is explicitly labelled or completed.
