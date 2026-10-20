# M1 pinned OpenML-CC18 pilot task manifest

Verified 2026-10-20. The [OpenML-CC18 suite](https://openml.github.io/docs/benchmark/) is suite 99. These task IDs were checked against its 72-task list. OpenML tasks carry the target and official train/test folds; this pilot runs repeat 0, fold 0, sample 0 only. The exact inner and outer row-index SHA-256 digests are in each output JSON.

| Task | Dataset ID/version | Rows / predictors | Target | Dataset MD5 | OpenML licence field | Origin |
| --- | --- | --- | --- | --- | --- | --- |
| [3](https://www.openml.org/t/3) | [kr-vs-kp 3/v1](https://www.openml.org/d/3) | 3,196 / 36 categorical | `class` | `ad6eb32b7492524d4382a40e23cdbb8e` | `Public` | Generated chess endgame positions, attributed by OpenML to Alen Shapiro and UCI. This is a benchmark dataset, not enterprise records. |
| [31](https://www.openml.org/t/31) | [credit-g 31/v1](https://www.openml.org/d/31) | 1,000 / 20 (13 categorical) | `class` | `9a475053fed0c26ee95cd4525e50074c` | `Public` | German Credit from UCI, attributed by OpenML to Dr. Hans Hofmann. Research only; not a lending decision system. |
| [10101](https://www.openml.org/t/10101) | [blood-transfusion-service-center 1464/v1](https://www.openml.org/d/1464) | 748 / 4 numeric | `Class` | `c3242468edab8c2e7a907674122dc851` | `Public` | Blood Transfusion Service Center, Hsin-Chu City, Taiwan; OpenML credits I-Cheng Yeh and colleagues. Research only; not a donor or clinical decision system. |

`Public` is the literal OpenML metadata value, **not** an independently verified commercial-use licence. The code downloads datasets to ignored `data/raw/openml/` and commits no rows. Confirm underlying UCI terms before any redistribution or commercial use. The dataset descriptions do not provide site, person, or chronological IDs sufficient for a validated natural-domain split; this pilot is IID fold evaluation. CC18's selection rules exclude documented grouped/time-stream tasks, but duplicate entities or hidden group structure cannot be ruled out from these metadata alone.

## Protocol

For each task, retrieve `task.get_train_test_split_indices(fold=0, repeat=0, sample=0)` and reject overlapping or incomplete indices. Stratify only the official training fold into 60% fit, 20% selection, 20% conformal calibration, with seed 42. Fit numeric median imputation and scaling, and categorical most-frequent imputation and bounded one-hot encoding, inside each expert's training pipeline. Select minimum selection Brier. Calibrate the fixed winner on calibration labels. Evaluate only that winner on the untouched official test fold. Reject changed task ID, target, dataset ID/version/name/licence/MD5 and datasets over 5,000 rows.

The five-expert option uses one fixed configuration each for logistic regression, HistGradientBoosting, random forest, XGBoost and CatBoost after the same one-hot preprocessing. This is a *pilot comparison*, not equal-compute hyperparameter tuning. CatBoost here does not use its native categorical encoding. Fit and test inference time are recorded; peak memory has not been measured.

## Reproduce

```powershell
cd D:\TrustTab-X
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
.\.venv\Scripts\python.exe -m pip install -e '.[dev,openml]'
.\.venv\Scripts\trusttabx.exe benchmark --output outputs/openml_m1_pilot.json
# Install the optional large packages separately if memory permits:
.\.venv\Scripts\python.exe -m pip install xgboost==3.4.1 catboost==1.2.10
.\.venv\Scripts\trusttabx.exe benchmark --advanced-experts --shift-lab --output outputs/openml_integrated_pilot.json
```

The executed integrated run used Python 3.12.10, NumPy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, OpenML 0.15.1, XGBoost 3.4.1 and CatBoost 1.2.10. See output JSON for per-task split digests, classes, package versions and results. No test labels are used for model choice or calibration. The class-prior stress cohort does use test labels to construct a fixed evaluation subset, and is explicitly separate from deployable adaptation.
