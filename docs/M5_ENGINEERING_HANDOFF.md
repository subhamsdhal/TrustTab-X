# M5 fixed-baseline engineering handoff

Research gate: [M4.2 decision](M4_2_RESEARCH_DECISION.md) is **C, no adaptive advantage demonstrated**. Build the service around a source-validation-selected **single fixed expert**. Keep the five-expert weighted ensemble behind a research/benchmark flag and the two routers disabled for ordinary scoring. The project may enter M5 engineering; it is not yet a production deployment.

## Training contract

Input: a versioned dataset adapter with target definition, feature names/types, availability time, licence, entity/group ID if available, domain/time ID, and immutable split manifest. Split fit, model selection, probability calibration and threshold calibration before any final test. Fit imputation/encoding/scaling only on fit rows. Compare the five fixed classical candidates within a finite documented budget and choose the minimum source-selection Brier expert. Persist the chosen pipeline, label mapping, feature schema, split hashes, dataset/source SHA256, seed, package versions, fit duration, peak working set and licence metadata. Optional sigmoid calibration must be chosen on **source validation**, fitted on independent probability-calibration data, and rejected if it fails the source criterion; the M4.2 shopper result shows it may transfer badly. Never choose a method or threshold using final target labels.

## Inference contract and audit record

The scorer accepts one validated record or a declared batch with a schema/version identifier. It returns probability, selected expert ID and model version, predicted class at a separately versioned decision threshold, and uncertainty/abstention status. Reject missing required fields, unseen schema versions and invalid numeric ranges with structured errors. If a source-fixed abstention threshold is used, return its source calibration coverage target and the actual action; do not imply OOD coverage. Batch-adaptive thresholds, if offered, require an explicit transductive mode and must identify the exact batch and unlabeled-score quantile used.

For each request log a generated decision ID, UTC timestamp, schema/model/policy versions, feature validation outcome, probability, class/threshold, uncertainty score, acceptance/abstention, reason code, and domain/drift flags. Do not log raw sensitive fields by default. Audit records must support replay from a model artifact and schema, with access controls, retention policy and redaction rules set before real data arrives.

## Reliability and domain monitoring

Track input missingness, category frequencies, numeric quantiles, model probability histogram, acceptance rate, model latency and error counts by declared domain and time window. Compare against source-fit/selection reference distributions using predeclared drift statistics (for example categorical total variation and numeric quantile shifts), with alert thresholds selected from source validation, not final target outcomes. Once delayed labels are available, track Brier/log loss, calibration bins, AUROC/AUPRC when both classes occur, and accepted error at **actual** coverage. Route large drift, calibration deterioration or low acceptance to human review and possible retraining; no unsupervised correction is presumed safe merely from a prevalence change. Slice results by permitted, documented groups without inferring attributes that are unavailable or inappropriate.

## Build order and acceptance gates

1. Package an offline trainer and artifact manifest for the fixed expert; add fixture tests for split isolation, schema round-trip and reproducible scoring.
2. Add an optional local MLflow tracker for parameters, split hashes, metrics, artifacts and resource measurements. Default to a local file store; no hosted account or paid backend. Raw records remain outside tracked artifacts.
3. Implement FastAPI `/health`, `/model`, `/predict`, and `/predict-batch` with Pydantic input validation and explicit model loading. Unit and integration tests must cover invalid schemas, deterministic probabilities, audit IDs, abstention and batch-mode distinction. No high-stakes decision workflow is in scope.
4. Add Docker build and a local smoke run with a non-sensitive fixture model; pin dependencies, use a non-root runtime user and document resource needs. Do not deploy publicly without authorization and security review.
5. Add monitoring, drift alerts and audit export with a small synthetic fixture. Measure p50/p95 inference latency, throughput and memory on the target hardware; compare the single model with the five-expert research benchmark at the same input size.
6. Publish a reproducibility package, dataset cards, negative results, method/compute table and accurate portfolio description. Do not claim the unsupported router as a contribution or production feature; any new research confirmation requires a newly preregistered environment.
