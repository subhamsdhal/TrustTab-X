# UCI shopper seasonal confirmation manifest

## Provenance and locked timing

The [UCI Online Shoppers Purchasing Intention dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset), ID 468, DOI 10.24432/C5F88Q, is a public one-year collection of 12,330 sessions under CC BY 4.0. Its `Revenue` column indicates whether a session resulted in a purchase. The repository page says sessions belong to different users, but the released CSV has no user identifier with which to verify that assertion. The local ignored archive `data/raw/uci_shoppers/online_shoppers_468.zip` came from `https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip`; SHA256 `2972e6184d3ad7beaaa831d9fc2b059dc3ee29df69d1ec593c466a5cd8485d14`. Raw rows were not committed.

The [confirmation preregistration](M4_2_CONFIRMATION_PREREGISTRATION.md) was committed as `9d5ad86` **before the archive was downloaded or its outcome labels opened**. This is a derived month-shift study; UCI does not supply exact dates within months. The hypothetical prediction point is after some page browsing but before checkout. `PageValues`, `BounceRates`, `ExitRates` and `Month` are excluded from predictors to reduce target/timing leakage; the preregistered remaining 13 features include observed page counts/durations and session context. Even their exact point-in-time availability cannot be proven from the CSV, so this is research benchmarking, not a prospective e-commerce decision claim.

## Disjoint environments

Fit uses February, March, May, June, July and August (6,608 rows, positive rate 0.1106); September is selection (448, 0.1920). October is split in supplied row order into probability calibration (274, 0.2117) and threshold calibration (275, 0.2073). Confirmation environments are untouched November (2,998, 0.2535) and December (1,727, 0.1251). Exact row-index hashes are in `reports/results/m4_2_shopper_confirmation.json`. The missing January and April months are absent in the source file. Within-October CSV order is only a reproducible disjoint split, not proven chronology.

Five fixed classical expert pipelines fit only on the fit months; source selection chose random forest. The existing metadata router and the simplified two-feature router were fitted only from the earlier three families' source-domain-held-out predictions. The latter uses equal family/domain sample weighting. Sigmoid calibration uses the first October half; confidence/disagreement thresholds use the second. November and December labels were read only for final evaluation after method fitting and thresholds were fixed. The confirmation was run once and no method was retuned afterward.

## Results and limits

| Policy | October Brier | November Brier / log loss | December Brier / log loss |
| --- | ---: | ---: | ---: |
| Source-selected random forest | 0.167811 | 0.190073 / 0.573256 | 0.103842 / 0.342500 |
| Uniform five-expert mean | 0.168469 | 0.189389 / 0.567769 | 0.102527 / 0.339848 |
| Validation-weighted mean | 0.168494 | 0.189411 / 0.567867 | 0.102534 / 0.339834 |
| Source-calibrated random forest | 0.163898 | 0.185863 / 0.555042 | 0.112327 / 0.387048 |
| Original metadata router | 0.174143 | 0.196618 / 0.603259 | 0.102092 / 0.338503 |
| Simplified router | 0.174882 | 0.201542 / 0.628375 | 0.102753 / 0.341490 |

The original router lost to both fixed and weighted policies in November; its small December Brier gain (0.000442 versus weighted) missed the preregistered 0.002 criterion and cannot overcome the failed nested source gate. Source-only calibration helped in November and hurt in December. October-to-November random-forest Brier changed by +0.022262, and October-to-December by -0.063969; these are related seasonal environments, not IID replicates. AUROC, AUPRC, ECE and complete risk–coverage curves are in the JSON.

At the source-set confidence thresholds, static acceptance was 0.7995 in November and 0.7817 in December; uniform and weighted acceptance was 0.8012/0.7759. The disagreement threshold accepted 0.8262/0.8014 but had higher accepted error than static confidence (0.2414/0.1113 versus 0.2282/0.0941). Separately, accepting the lowest-uncertainty 80% of each **unlabeled target batch** gave static accepted errors 0.2280/0.0977. The batch rule attains coverage by using target scores and cannot be equated with an online fixed threshold. No arbitrary-shift risk guarantee or independent-family confidence interval is claimed.

Five model fits took 3.127 seconds in one Windows CPU run; RF alone took 0.393 seconds. Five-model inference took 0.0846 seconds for November and 0.0595 seconds for December; these timings include all five experts and do not represent optimized single-RF inference. Peak process working set was 337,144 KiB. This small, one-site, one-year benchmark does not establish general deployment performance. Reproduce with `python -c "from trusttabx.shopper_confirmation import run_shopper_confirmation,save_shopper_result; save_shopper_result(run_shopper_confirmation(),'reports/results/m4_2_shopper_confirmation.json')"` using the pinned archive and single-thread BLAS.
