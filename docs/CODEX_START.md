# Launch from Codex

This ChatGPT session does not have a direct action to start a Codex cloud job. Prepare a **new, dedicated `trusttab-x` GitHub repository** instead of editing an unrelated repository, and open it in Codex Cloud.

## Repository setup

1. Download and unzip the starter `trusttab-x-starter.zip`.
2. Create a **new** GitHub repository, for example `trusttab-x`. Keep it private until licences, data and attribution have been checked.
3. In the extracted folder:

   ```bash
   git init
   git add .
   git commit -m "chore: initialize TrustTab-X research baseline"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/trusttab-x.git
   git push -u origin main
   ```

4. Open **Codex** from ChatGPT, select/create a Cloud environment for this repository, and choose **Code** mode. If the repo isn't visible, grant it repository access in the GitHub/Codex connection settings.
5. Paste the task below. Review the diff/tests before merging the Codex pull request.

## First Codex coding task — M1

```text
Repository: trusttab-x
Read AGENTS.md, README.md, docs/PROJECT_PLAN.md and docs/EXPERIMENT_PROTOCOL.md first.

Implement milestone M1: a reproducible real-data tabular benchmark harness for
binary classification. Preserve the current offline synthetic smoke test.

Requirements:
1. Add an optional OpenML loader that fetches a small, explicitly recorded
   subset of OpenML-CC18 classification TASK IDs (not just arbitrary dataset IDs).
   Enforce a conservative size cap and allow filtering binary tasks; fail
   clearly if a task is unsupported. Do not assume a task is binary.
2. Store task ID, dataset version, URL/source, target, sample size, feature
   types, licence where available, and split protocol in a JSON manifest.
3. Add train-only preprocessing: per-column numeric imputation/scaling and
   categorical imputation/encoding via ColumnTransformer inside pipelines.
4. Implement a fair baseline benchmark on multiple real tasks; use separate
   train, model-selection, calibration and blind test partitions. Respect
   official folds and grouping/time metadata when relevant.
5. Report Brier, log loss, AUROC where valid, conformal empirical coverage,
   mean prediction-set size, abstention rate, and model training/inference time.
6. No TabPFN integration yet; that is M2 and requires explicit licence review.
7. Add offline fixture-based unit tests and docs. No network in regular CI.
8. Run pytest and a smoke experiment. Document any unexecuted live-download
   test as unverified, not as passing.
9. Do not claim research novelty or better performance without evidence.

Commit as a focused PR: 'feat: add reproducible OpenML benchmark harness'.
Include the test logs and limitations in the PR summary.
```

## Subsequent Codex tasks

**M2**: `Build separately licensed model adapters for XGBoost, CatBoost and an explicitly pinned research-only TabPFN checkpoint. Add resource tracking and controlled equal-budget comparisons. Add tests; do not allow the non-commercial checkpoint in the commercial demo image.`

**M3**: `Create controlled covariate, prior, missingness and concept shifts with correctly named shift mechanisms, plus held-out natural shifts where datasets support them. Validate shift construction with statistical tests and figures.`

**M4**: `Implement a source-trained shift-aware expert router with an independent conformal calibration partition and leave-one-dataset-out evaluation. Compare against fixed expert, simple abstention and fair resource-matched ensembles. No target/test labels for adaptation.`

**M5/M6**: `Build risk-coverage diagnostics, robustness ablations, FastAPI service, drift monitoring, logs, Docker, tests, research report generation. Keep model weights licensing separate from code.`
