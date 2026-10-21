# TrustTab-X preliminary research report

## M4.1: why routing failed and independent institution-type confirmation

The prior negative ACS, hospital and bank results were reproduced exactly. The saved source cross-fit predictions show strongly correlated expert squared errors: mean pairwise correlation 0.936 for ACS, 0.936 for hospital and 0.998 for bank. Mean inter-expert probability standard deviations were only 0.041, 0.035 and 0.015. A label-aware per-row oracle has lower source Brier, but it uses the answer and is strictly a diagnostic upper bound. The best expert by source-domain Brier varied (ACS: HistGB 4/7, and logistic/XGBoost/CatBoost one each; hospital: CatBoost 2/4, HistGB and XGBoost one each; bank: logistic 2/3, CatBoost one), but this does not imply a router can identify the winner without labels. Probability-only metadata may identify general difficulty (router predicted-risk vs realized squared-error Spearman averaged 0.773/0.495/0.524 across source domains), yet does not reliably identify a *better expert*. Five separate ridge regressions have 12 correlated probability-derived inputs, with only three original dataset families and no explicit raw-feature shift descriptor. Thus correlated expert errors, weak differential signal, heterogeneous probability scales and scarce independent environments plausibly explain the failures; the experiment does not isolate a unique causal root cause. No target-domain router tuning was performed.

In source-only leave-one-family-out meta-transfer, the router beat a fixed expert selected on the other two families in 7/14 source domains. Mean per-domain Brier changes (router minus static) were ACS -0.000277 over seven states, hospital +0.000833 over four admission codes, and bank +0.000053 over three forward windows. ACS's seven states are related, not seven independent dataset confirmations. This nested *meta* split excludes all held-out family labels from router fitting, but the original base-model cross-fits were not rebuilt for a fully nested model-selection study. The result is exploratory evidence of weak/inconsistent transfer.

The bank fixed HistGradientBoosting model predicted a mean 0.06032 in near-future and 0.06496 in final-future, while observed subscription rates were 0.06846 and 0.38459. Final Brier/log loss were 0.337574/1.099454. Contact method, month and previous outcome also shifted substantially (total variation 0.513, 0.451 and 0.431 versus the fit window), and earlier calibration-window prevalence was 0.16800. This supports a calibration failure under a mixed temporal distribution change. Prevalence change alone cannot establish pure label shift or justify prior correction. Unknown customer recurrence and approximate campaign chronology limit causal interpretation.

The independent locked confirmation used 2025 College Scorecard institution records with public/private nonprofit source and for-profit target, preregistered before outcome inspection. On 228 final institutions, source-selected random forest Brier/log loss were **0.263707/0.724673**. Source-only sigmoid calibration worsened them to **0.266892/0.732861**; uniform and source-weighted ensembles scored **0.268967/0.738449** and **0.268696/0.737673**. The unchanged router scored **0.285757/0.797401**, a Brier loss of **0.022050** to fixed RF. The router's AUROC (0.61294) exceeded RF's (0.56148), but its probabilities were worse by the primary proper score. This single 228-row target is noisy and cannot support an independent-environment interval.

Source-calibrated RF confidence abstention accepted 68.9% of target institutions at 46.5% error risk; expert-disagreement abstention accepted 76.8% at 45.1%. Router predicted-error abstention accepted 75.4% at 42.4%. None met the prespecified minimum 80% target coverage, so none satisfies the selective objective as deployed. Target-ranked, matched-coverage curves are descriptive only. Five College experts fit in about 2.865 seconds locally, while the source-selected RF alone fit in about 0.189 seconds; the latter is substantially cheaper at deployment. Peak Windows process working set was 595,964 KiB. No statistical significance is claimed across one confirmation domain. The single-year derived split and retrospective covariates are not the official TableShift task; see `docs/COLLEGE_SCORECARD_DOMAIN_MANIFEST.md`.

## Third natural shift and exploratory router (2026-10-21)

The preregistered UCI Bank Marketing pre-call chronological evaluation is detailed in `docs/BANK_TEMPORAL_MANIFEST.md`. The source-selected HistGradientBoosting expert went from near-future Brier 0.063753 to final-future 0.337574 as prevalence changed from 0.06846 to 0.38459; final ECE was 0.31963. This is one late holdout without unique timestamps or customer identifiers. Source-only cross-fitting across ACS, hospital and bank produced 14 folds and 51,917 held-out predictions. A locked regularised router using common expert-probability metadata was then tested against fixed expert, uniform/weighted ensemble and confidence controls. It lost to the fixed expert on Brier in all three target settings (ACS 0.148039 vs 0.145517; hospital 0.258926 vs 0.250504; bank 0.344292 vs 0.337574). The bank uniform ensemble reached 0.330909. These negative results are retained in `reports/results/router_prototype_3families.json`. See `docs/DOMAIN_CROSSFIT_MANIFEST.md` for scope and limits. The primary shift-aware, externally validated M4 contribution remains unproven.

## Second natural-domain result: 2018 ACSIncome geography

The bounded, household-disjoint Folktables/TableShift-inspired protocol in `docs/ACS_INCOME_DOMAIN_MANIFEST.md` was executed on seven source states and held-out New England. Both the three- and five-candidate pools selected HistGradientBoosting on source selection data; their ID/OOD Brier scores were 0.131644/0.145517 and log losses 0.406822/0.441380. Five candidates did not improve prediction over three in this run. State-level results and resource measurements are in `outputs/acs_income_2018_geographic_expanded.json`. This is one survey year and one target division. ACS and diabetes have incompatible raw schemas, and only two independent dataset families are evaluated. A learned cross-dataset router is therefore still at the research gate; `docs/M4_RESEARCH_SPEC.md` requires domain-held-out source predictions and comparable metadata features before fitting one.

## Natural-domain pilot update (2026-10-20)

An executed patient-disjoint, TableShift-derived UCI diabetes admission-source holdout is documented in `docs/TABLESHIFT_DOMAIN_MANIFEST.md`; machine-readable metrics and split hashes are in `outputs/tableshift_diabetes_12000.json`. Shared five-model training cost was 3.818 seconds; the 3-candidate selector chose random forest and the 5-candidate selector chose CatBoost on source selection Brier. ID/OOD Brier was 0.201856/0.256984 versus 0.201603/0.250504, respectively. OOD log loss was 0.710901 versus 0.696820. Both suffered calibration degradation under shift. This is one exploratory domain pair, with a modified patient-disjoint split and limited features, so it does not establish broad generalisation or justify a trained router. The prospective method is in `docs/M4_RESEARCH_SPEC.md`.

**Checkpoint:** 2026-10-20. **Status:** three-task, 10-fold, three-seed classical benchmark plus a one-fold shift pilot. This is not the final study or evidence of an adaptive router.

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

## Expanded 10-fold, three-seed checkpoint

The CLI was extended to all 10 official folds and inner seeds 42, 43 and 44. `outputs/openml_10fold_3seed_classical.json` and `outputs/openml_10fold_3seed_advanced.json` each contain 90 completed task/fold/seed records, paired exactly by those three keys. Means below are descriptive averages of the 30 runs per task. The three dataset tasks, rather than their 90 correlated evaluations, are the cross-dataset units.

| Task | Three-expert mean Brier ↓ | Five-expert mean Brier ↓ | Paired mean change (five minus three) | Five-expert runs worse |
| --- | ---: | ---: | ---: | ---: |
| 3, kr-vs-kp | 0.007781 | 0.007781 | 0.000000 | 0/30 |
| 31, credit-g | 0.171396 | 0.169002 | −0.002394 | 5/30 |
| 10101, blood transfusion | 0.159542 | 0.156818 | −0.002723 | 5/30 |

The five-expert selector chose HistGradientBoosting in all 30 chess runs; on credit it chose CatBoost 15/30 times and random forest 10/30 times, and on blood it chose logistic regression 11/30 times and CatBoost 14/30 times. This shows model choice is dataset and inner-split dependent. The advanced run emitted a few scikit-learn probability-sum warnings when two-column probabilities were sent to log-loss; the evaluation code now supplies the positive-class column, which represents the same binary scoring quantity. No broad superiority inference follows from three tasks or these fixed configurations.

## Controlled stress results

The fixed M3 harness applies 20% MCAR feature-cell missingness, numeric measurement noise at 0.5 training standard deviations, and class-label-stratified downsampling toward 70% positive prevalence. These are **constructed evaluation cohorts**. Missingness/noise preserve labels but can alter the observed conditional relation; they are not automatically pure covariate shift. Prior cohort construction reads test labels only to select the evaluation subset, never to fit or choose the model or thresholds.

On task 10101 in the integrated run, the prior cohort contained only 26 rows (realized positive fraction 0.6923). Brier rose from 0.155100 IID to 0.343586; empirical conformal coverage fell from 0.906667 to 0.730769. No shift-robust guarantee follows, and the small cohort makes this estimate noisy. Task 31's test fold already had 70% positive labels, so the prior intervention retained all 100 test rows and caused no actual prior shift; that result should not be interpreted as robustness.

## Limitations and threats to validity

- Three datasets, ten official outer folds and three inner seeds in the expanded classical comparison, with no task-level interval or multiplicity control. The shift study remains one fold and seed. The chess dataset consists of game states and is not an enterprise record set.
- No natural domain, time or site shift was tested. Hidden duplicate entities or group structure remain possible from the available metadata.
- Selection and calibration are disjoint, but the final model is fitted only on the 60% inner fit partition; a retrained full-fold model would need a new calibration protocol.
- Fixed expert configurations are not an equal-compute tuned comparison. CatBoost uses common one-hot features, not native categorical handling. Inference time is recorded but peak memory is absent.
- The OpenML licence field says `Public`; this is not independent clearance for redistribution or commercial decisions. Credit and health examples are research benchmarks only.
- Split-conformal coverage is marginal under exchangeability. Arbitrary shift invalidates that standard guarantee. Confidence-ranked AURC and 10-bin ECE are descriptive and do not certify safe individual decisions.
- TabPFN, a learned router, fairness slices and an enterprise service were not evaluated or built.

## Reproducibility and references

Commands, dataset IDs/versions/MD5s, split-index hashes, package versions, and exact JSON metrics are in [the task manifest](../docs/OPENML_TASKS.md), [status ledger](../docs/AUTOPILOT_STATUS.md) and `outputs/`. The [literature review](../docs/LITERATURE_REVIEW.md) cites primary papers and separate TabPFN code/weight licence terms. The next experiment should pre-register multiple official folds/seeds and real held-out domains before training a router.
