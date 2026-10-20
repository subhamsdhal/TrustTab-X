# TrustTab-X preliminary research report

**Checkpoint:** 2026-10-20. **Status:** three-task, one-fold pilot. This is not the final study or evidence of an adaptive router.

## Question and design

The eventual question is whether source-trained expert routing and abstention improve reliable prediction on unseen domains over a static validation-selected expert. The current work establishes an IID tabular benchmark and controlled stress harness before router development. Three pinned [OpenML-CC18 tasks](../docs/OPENML_TASKS.md) were run on official repeat 0/fold 0/sample 0. Within each official training fold, 60% of rows fit preprocessing and experts, 20% selected the minimum-Brier expert, and 20% calibrated split-conformal sets. The official outer test fold was used once for evaluation. The positive class follows sorted training labels and is stored per task in JSON.

## Executed results

All values below come from `outputs/openml_integrated_pilot.json` using the five-expert option (logistic regression, HistGradientBoosting, random forest, XGBoost, CatBoost after one-hot encoding), seed 42 and alpha 0.1. These are descriptive results for one fold and seed.

| Task | Selected expert | Test n | Brier ↓ | Log loss ↓ | AUROC ↑ | Accuracy ↑ | Conformal coverage | Abstention | ECE (10 bins) | Confidence AURC ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3, kr-vs-kp | HistGradientBoosting | 320 | 0.005255 | 0.025141 | 0.999883 | 0.996875 | 0.890625 | 0.109375 | 0.015442 | 0.000079 |
| 31, credit-g | CatBoost (one-hot) | 100 | 0.161748 | 0.491197 | 0.794286 | 0.730000 | 0.950000 | 0.440000 | 0.077609 | 0.121897 |
| 10101, blood transfusion | logistic regression | 75 | 0.155100 | 0.463873 | 0.780214 | 0.746667 | 0.906667 | 0.333333 | 0.069812 | 0.094273 |

The three-expert M1 pilot selected logistic regression on task 31 with test Brier 0.157354. Adding XGBoost and CatBoost led the selection split to choose CatBoost, but its test Brier was **worse** at 0.161748. This is a negative pilot observation, not a paired significance claim or a reason to tune on test labels. Task 3's 0.890625 conformal coverage is below the nominal 0.9 level in one 320-row fold; marginal coverage is a repeated-sampling guarantee under assumptions, not an exact per-fold quota.

## Controlled stress results

The fixed M3 harness applies 20% MCAR feature-cell missingness, numeric measurement noise at 0.5 training standard deviations, and class-label-stratified downsampling toward 70% positive prevalence. These are **constructed evaluation cohorts**. Missingness/noise preserve labels but can alter the observed conditional relation; they are not automatically pure covariate shift. Prior cohort construction reads test labels only to select the evaluation subset, never to fit or choose the model or thresholds.

On task 10101 in the integrated run, the prior cohort contained only 26 rows (realized positive fraction 0.6923). Brier rose from 0.155100 IID to 0.343586; empirical conformal coverage fell from 0.906667 to 0.730769. No shift-robust guarantee follows, and the small cohort makes this estimate noisy. Task 31's test fold already had 70% positive labels, so the prior intervention retained all 100 test rows and caused no actual prior shift; that result should not be interpreted as robustness.

## Limitations and threats to validity

- Three datasets, one outer fold and one seed, with no task-level interval or multiplicity control. The chess dataset consists of game states and is not an enterprise record set.
- No natural domain, time or site shift was tested. Hidden duplicate entities or group structure remain possible from the available metadata.
- Selection and calibration are disjoint, but the final model is fitted only on the 60% inner fit partition; a retrained full-fold model would need a new calibration protocol.
- Fixed expert configurations are not an equal-compute tuned comparison. CatBoost uses common one-hot features, not native categorical handling. Inference time is recorded but peak memory is absent.
- The OpenML licence field says `Public`; this is not independent clearance for redistribution or commercial decisions. Credit and health examples are research benchmarks only.
- Split-conformal coverage is marginal under exchangeability. Arbitrary shift invalidates that standard guarantee. Confidence-ranked AURC and 10-bin ECE are descriptive and do not certify safe individual decisions.
- TabPFN, a learned router, fairness slices and an enterprise service were not evaluated or built.

## Reproducibility and references

Commands, dataset IDs/versions/MD5s, split-index hashes, package versions, and exact JSON metrics are in [the task manifest](../docs/OPENML_TASKS.md), [status ledger](../docs/AUTOPILOT_STATUS.md) and `outputs/`. The [literature review](../docs/LITERATURE_REVIEW.md) cites primary papers and separate TabPFN code/weight licence terms. The next experiment should pre-register multiple official folds/seeds and real held-out domains before training a router.
