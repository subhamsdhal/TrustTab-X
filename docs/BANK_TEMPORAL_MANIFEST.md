# Bank Marketing temporal benchmark result (2026-10-21)

The [UCI Bank Marketing page](https://archive.ics.uci.edu/dataset/222/bank+marketing) identifies `bank-additional-full.csv` as 41,188 records in date order, May 2008 to November 2010, and gives the dataset a CC BY 4.0 licence. The local ignored archive `data/raw/bank/bank-additional.zip` has SHA-256 `a607b5edab6c6c75ce09c39142a77702c38123bd5aa7ae89a63503bbe17d65cd`. Exact record timestamps and reliable customer identifiers are absent. All data processing preserved file order. No date was inferred from row number or month.

The protocol was locked in `docs/BANK_TEMPORAL_PREREGISTRATION.md` before examining labels. The decision point is before the current telephone call. `duration` and ambiguous `campaign` are excluded, along with economic indicators whose release timing at each call was not established. The 13 predictors and their timing assumptions are listed there. A customer may recur across windows; customer-disjointness cannot be verified.

| Window, half-open rows | N | Positive rate | Brier | Log loss | AUROC | AUPRC | ECE | Conformal coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fit `[0,20594)` | 20,594 | 0.04676 | — | — | — | — | — | — |
| Selection `[20594,26772)` | 6,178 | 0.05293 | — | — | — | — | — | — |
| Calibration `[26772,30891)` | 4,119 | 0.16800 | — | — | — | — | — | — |
| Near future `[30891,35010)` | 4,119 | 0.06846 | 0.063753 | 0.249350 | 0.538815 | 0.073909 | 0.008141 | 0.956300 |
| Final future `[35010,41188)` | 6,178 | 0.38459 | 0.337574 | 1.099454 | 0.551428 | 0.431648 | 0.319628 | 0.787472 |

Source selection chose HistGradientBoosting in both three- and five-expert pools; the separately fitted three-expert pool reproduced its final Brier. The five-expert pool took 4.681 s to fit; a separate three-expert pool took 1.600 s. Selected-model inference was 0.019 s near future and 0.025 s final future. Peak Windows process working set was 362,636 KiB. All are one-run local CPU timings. The positive rate changed substantially across windows, so AUPRC should not be read as a stable ranking comparison. The late-window Brier increase of 0.273822 is descriptive for one future segment. This is temporal evaluation, not IID evaluation, and neither split-conformal coverage under shift nor future generalisation is guaranteed. Exact metrics and reliability bins are in `reports/results/bank_temporal_2008_2010.json`.
