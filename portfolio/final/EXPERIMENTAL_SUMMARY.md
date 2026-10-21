# Experimental summary and decision table

| Study | Fixed/expert Brier | Router or comparison | Interpretation |
|---|---:|---:|---|
| ACS bounded geographic ID → OOD | 0.131644 → 0.145517 | No final-domain router improvement claim | Household-disjoint; modified TableShift-inspired state split. |
| Hospital modified admission-domain ID → OOD | 0.201603 → 0.250504 | No final-domain router improvement claim | Patient-disjoint; not official TableShift. |
| Bank near future → later future | 0.063753 → 0.337574 | No final-domain router improvement claim | Both future; positive prevalence 0.06846 → 0.38459; customer recurrence unverified. |
| Nested source ACS (7 states) | static 0.141614 | router 0.141437; weighted 0.141332 | Correlated source states; weighted mean lower than router. |
| Nested source hospital (4 codes) | static 0.189329 | router 0.189170; weighted 0.188566 | Correlated admission domains; weighted mean lower. |
| Nested source bank (2 forward blocks) | static 0.056951 | router 0.057372; weighted 0.056665 | Router mean worse than both controls. |
| Preregistered shopper November | static 0.190073 | router 0.196618; weighted 0.189411 | Router worse. |
| Preregistered shopper December | static 0.103842 | router 0.102092; weighted 0.102534 | Router gain over weighted 0.000442, below locked 0.002 margin. |

The 13 nested source domains belong to only three original families; November and December are one additional family. Counts and rows do not justify a family-level confidence interval. M4.2 decision **C: no adaptive advantage demonstrated**. The engineering baseline is a source-selected fixed expert. M5.1 decision **B: local demo only**. [Exact claim trace](CLAIM_TRACEABILITY.md) and [figures](figures/manifest.json) identify the tracked result JSON for every plotted quantity.
