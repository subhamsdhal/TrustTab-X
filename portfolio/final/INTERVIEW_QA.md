# Technical interview: questions and evidence-backed answers

**What was the decision problem?** Could a source-trained router choose the best expert under new domains without target labels? The locked gate required it to beat both a source-selected fixed expert and a weighted ensemble, including an improvement margin on a new shopper family.

**How did you prevent leakage?** Preprocessing and fitting stayed within source fit folds. Selection, probability calibration and threshold calibration had separate source roles. The nested M4.2 experiment excluded each outer source domain from inner fits and choices. ACS households and hospital patients were split across partitions; bank training respected chronological blocks. Bank customer recurrence and exact contact time are not verifiable from the archive, so leakage is not completely ruled out.

**What does Brier measure?** For binary outcomes it averages \((p-y)^2\), rewarding calibrated probabilities and discrimination together. It is a proper scoring rule. The shift comparisons are descriptive and can also reflect changing outcome prevalence; bank near/later prevalence changed markedly.

**Why did the router fail?** It beat static on 6/13 correlated source domains, but weighted ensemble family means were lower. Expert squared errors were strongly correlated, leaving limited observed selection headroom. In the independent shopper family, November was worse and December's 0.000442 gain over weighted missed the 0.002 margin. The evidence does not prove the router architecture can never help; it does not justify deploying this one.

**Why is a target-ranked risk–coverage curve insufficient for a service policy?** It is computed retrospectively using target outcomes to describe the ranking. The service threshold must be fixed on source calibration rows. The resulting target acceptance may miss the intended 80%; retuning it using the target labels would invalidate the reported gate.

**What does a drift alert mean?** It flags changes in feature or predicted-probability summaries relative to source reference data. It does not establish actual error, fairness harm or whether a particular prediction is wrong. Delayed labels are needed for realized Brier and calibration metrics.

**What would be needed to deploy?** Verified container/runtime build, TLS, managed identity or secure browser session, central rate limits, audit retention and access governance, data-use/privacy review, security approval and application-specific validation. The current service is release B, local demo only.

**What is original here?** The contribution is an auditable experimental protocol, negative routing decision and bounded local engineering integration. The base algorithms, Brier score, calibration methods and public benchmark ideas are existing work; no new mathematical estimator or peer-reviewed publication is claimed.
