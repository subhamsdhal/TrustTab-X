# Statistical and research evaluation protocol (pre-register before comparing methods)

## M4.2 final finite evaluation addendum (2026-10-21)

Nested source-domain folds re-fit each expert and preprocessing after excluding the entire outer environment. Inside each outer training set, disjoint entity/time-aware fit, selection, probability-calibration and threshold-calibration partitions determine the fixed expert, ensemble weights, ridge router training and abstention cutoffs. No hyperparameter search was conducted; five expert configurations and Ridge alpha=100 were fixed. ACS folds are household-disjoint, hospital folds patient-disjoint, and bank folds forward-only. Source folds are correlated within three dataset families; only family-level patterns support transfer discussion.

`docs/M4_2_CONFIRMATION_PREREGISTRATION.md` was committed as `9d5ad86` before UCI shopper labels were opened. November and December are untouched seasonal confirmation environments from one independent dataset family, with source months and feature timing defined in `docs/SHOPPER_MONTH_SHIFT_MANIFEST.md`. Fixed source uncertainty thresholds and separately identified unlabeled-batch transductive thresholds are both evaluated. The locked A/B/C research rule and its executed result are in `docs/M4_2_RESEARCH_DECISION.md`. The negative outcome ends this finite router evaluation; previously inspected targets may not be reused as fresh confirmation for a revised method.

## M4.1 confirmation and selective objectives (2026-10-21)

`docs/M4_1_PREREGISTRATION.md` was saved before College Scorecard outcome inspection. The 2025 single-cohort institution-type target is a fresh family-level confirmation for the unchanged probability-metadata router; ACS/hospital/bank final outcomes are exploratory audits. Source institution IDs are disjoint across fit, selection, probability-calibration and threshold-calibration; for-profit institutions are untouched until the locked comparison. The new family has one target environment, so no across-environment confidence interval or broad generalisation claim is available. Existing source cross-fit artifacts are used separately for leave-one-entire-family-out meta-transfer diagnosis; those 14 source domains are not 14 independent datasets.

Objective 1 is to minimize expected proper probability loss, primarily Brier and secondarily log loss, subject to an explicit training/inference cost budget. Objective 2 is to minimize accepted-prediction error subject to **actual** target coverage at least 0.8. Source-calibration thresholds target 0.8 but may violate the target constraint under shift; if so, a lower accepted risk does not meet Objective 2. Target-ranked risk–coverage and fixed-risk coverage curves are descriptive oracle rankings, not fitted thresholds. The confirmation protocol forbids final labels in model/threshold selection. No label-shift or unsupervised target adaptation is claimed or attempted.

## Chronological and source-domain addendum (2026-10-21)

The UCI bank protocol was recorded in `docs/BANK_TEMPORAL_PREREGISTRATION.md` before target labels were read. Fit, selection, calibration, near-future and final-future ranges are contiguous in the supplied order. Near-future is a later environment, not an IID test. Source-only domain cross-fitting is implemented in `src/trusttabx/domain_eval.py`: geographic/institutional domains use leave-one-domain-out, while temporal folds train only on earlier blocks. Entity groups must not cross fit/validation where available. Reserved final rows are excluded by construction. `docs/ROUTER_PROTOTYPE_PREREGISTRATION.md` locks the exploratory metadata-only router and fixed controls before final evaluation. The three final-domain scores are descriptive, with no row-IID significance test or arbitrary-shift conformal guarantee.

## Experimental units

- Begin with **binary classification**. Select a fixed set of suitable OpenML tasks before examining comparative outcomes.
- Record task/dataset identifiers, version, feature schemas, target leakage risks, data origin, usage licences and row/column dimensions.
- Do not treat independent rows from one dataset as independent benchmark tasks when reporting cross-dataset significance.
- Real domain shifts must be identified through documented site/region/time partitions, not invented after seeing results.

## Splits and anti-leakage

1. `train`: fitting preprocessing and experts; feature transformations must be learned from training data only.
2. `selection`: comparing static model candidates, fitting nonconformal components/routing policy from source validation environments.
3. `calibration`: held out for calibration/conformal thresholds **after** model and policy choice.
4. `test`: final blinded IID/domain-shift evaluation only. Never use its labels to select architectures, tune shift parameters or thresholds.

For router training, use nested/out-of-fold predictions from source environments, reserving held-out datasets/domains for final meta-generalisation. Keep conformal calibration disjoint from router fitting. Under time/group structure, honor chronology/groups and document any compromises; generic random splits are not acceptable for every dataset.

## Shifts

- **Covariate shift:** change p(X), keep p(Y|X) fixed in simulation; record feature-level perturbations.
- **Label/prior shift:** change class proportion with (approximately) stable p(X|Y); record exact construction and evaluate appropriate prior-correction baselines.
- **Missingness shift:** MCAR, MAR, MNAR with mechanisms recorded; baseline imputers trained on train only.
- **Concept shift:** if changing p(Y|X), label this distinct mechanism; don't call it pure covariate shift.
- **Natural shift:** use held-out geographic/time/site cohorts and publish the grouping logic.

The current synthetic smoke test covers only the first of these, using p(X) changes while preserving the same outcome generation equation.

## Experts and ablations

- Classical: LogisticRegression, HistGB, XGBoost/CatBoost, Random Forest.
- Foundation: TabPFN (pin checkpoint, include weights licence); optional second model if resource budget allows.
- Fixed strategies: best selection-split expert; naive ensemble; drift-threshold abstention; reliability-only abstention; always-select-foundation.
- Research strategy: source-trained meta-router using dataset/instance shift and reliability features, with a no-target-label adaptation budget. Compare against resource-matched controls.
- Ablations: no shift signal, no calibration, no foundation model, no router, fixed vs data-dependent abstention.

An uncomputed oracle can only be marked as a *diagnostic upper bound* and must never be passed off as a deployable model.

## Metrics

- Proper scoring: Brier score and log loss; secondary AUROC, accuracy and PR-AUC where appropriate.
- Reliability: calibration curves/ECE with stated bins, calibration error, class-wise behaviour, coverage and set sizes.
- Selective prediction: risk–coverage curves, area under risk–coverage curve (AURC), conditional singleton accuracy and abstention rate.
- Shift: difference in each metric between IID and shifted cohorts, paired when possible.
- Efficiency: elapsed training/inference time, hardware, memory, data and compute budget.
- Fairness: where dataset attributes and permits allow, document group-level error disparities without inferring demographics from unavailable data.

## Statistical comparison

- Multiple fixed seeds and disjoint held-out tasks/domains, not just one favorable benchmark.
- For matched experiments, compute paired per-dataset effect sizes and confidence intervals (bootstrap over datasets or suitable grouped units, not naïve independent row bootstrap across all tasks).
- Document selection bias, multiplicity and sensitivity to shifts/threshold choices.
- Preserve negative results and limitations in the research report.

## Conformal validity

The v0.1 score is `1 - p_true`; with n calibration samples, the order-statistic threshold is at rank `ceil((n+1)*(1-alpha))`. Marginal coverage needs exchangeability plus held-out calibration with a fixed predictor. Under arbitrary shift, nominal marginal coverage **is not guaranteed**; evaluate empirical coverage and do not state otherwise. Selected singletons are not themselves guaranteed 1-alpha accuracy.

## Licensing and ethics

Keep a manifest of downloaded dataset licences; don't commit private or sensitive records. TabPFN model-weight licences are distinct from the package code licence and may bar commercial deployment. Do not send client/customer data to hosted ML APIs without permission. Do not claim production readiness for high-stakes lending, medical or employment decisions based on research benchmarks.
