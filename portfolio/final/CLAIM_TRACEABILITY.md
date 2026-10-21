# M6 claim traceability

Every primary numeric claim below is drawn from a tracked aggregate JSON. The [verification command](REPRODUCIBILITY.md) checks critical values, figure input hashes and local manuscript links. For the full experiment provenance, use `RESULTS_VERIFICATION.md` and the relevant dataset manifest. The figure script is `scripts/build_m6_figures.py`.

| Manuscript claim | Exact source field or record | Limitation |
|---|---|---|
| ACS fixed expert ID/OOD Brier 0.131644/0.145517 | `reports/results/acs_income_2018_geographic_expanded.json`: `results.five_experts.id_test/ood_test.metrics.brier` | Modified bounded geography; no official TableShift score. |
| Hospital fixed expert ID/OOD Brier 0.201603/0.250504 | `reports/results/tableshift_diabetes_12000.json`: same metric fields | Modified patient-disjoint study. |
| Bank near/later Brier 0.063753/0.337574, ECE 0.008141/0.319628 | `reports/results/bank_temporal_2008_2010.json`: `results.five_experts.near_future/final_future.metrics` | Both periods are future; recurrence and timing uncertain. |
| Nested 13-domain Brier means, router 6/13 wins, source-fixed coverage counts | `reports/results/m4_2_nested_acs.json`, `m4_2_nested_diabetes.json`, `m4_2_nested_bank.json`: `outer_domains.*.policies.*.probability.brier` and `selective.confidence.source_threshold.target_coverage` | Domains within families are correlated. |
| Expert squared-error overlap | `reports/results/m4_1_source_router_diagnosis.json`: `families.*.domains.*.mean_pairwise_squared_error_correlation` | Diagnostic association, not causal evidence. |
| Shopper November/December Brier, fit cost and source-fixed threshold | `reports/results/m4_2_shopper_confirmation.json`: `environments.nov/dec.policies.*`, `fit_seconds_by_expert` | One family, two correlated months, one-run timings. |
| College for-profit static/router Brier | `reports/results/college_scorecard_2025_confirmation.json`: `policies.static/reliability_router.probability.brier` | Already viewed final cohort. |
| M5.1 loopback latency, RSS, startup and probe | `reports/results/m5_1_security_verification.json`, explained in `RESULTS_VERIFICATION.md` | Local smoke, no container/runtime release claim. |
| M4.2 decision C and M5.1 release B | `docs/M4_2_RESEARCH_DECISION.md`, `docs/M5_SECURITY.md`, `docs/M6_PORTFOLIO_HANDOFF.md` | Router research-only; local demo only. |

The citation links in the [manuscript](TrustTab-X-manuscript.md) point to the primary papers or official proceedings. The project does not claim a peer-reviewed publication, officially scored TableShift result, production deployment, or independent company endorsement.
