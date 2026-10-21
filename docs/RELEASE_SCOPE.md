# TrustTab-X release scope

## A. Research and academic portfolio — complete

The editable manuscript, six figures, their source hashes, aggregate evidence, claim trace,
reproducibility guide, and review package are checked by `scripts/verify_m6.py` and
`scripts/verify_m7.py`. The M4.2 result is **C: no adaptive advantage demonstrated**.
The service therefore uses the source-selected fixed expert. The manuscript has not been
peer reviewed or published; bounded datasets and split limitations remain in the paper.

## B. Container-verified local demonstration — blocked

The intended acceptance evidence is the real Docker Compose run of
`scripts/verify_m5_container.py --runtime` on the private repository's `ubuntu-24.04`
GitHub-hosted runner. It must build the service image and run only a synthetic bundle,
exercise authenticated API and dashboard paths, verify negative security paths, reject
tampered artifacts, restart, and record resource measurements. Static Compose validation
and local FastAPI `TestClient` results alone do not complete this classification.

The hosted run is blocked because account-level included Actions minutes and the
no-charge spending limit could not be confirmed with the available GitHub token.
The Compose port is bound to `127.0.0.1:8000`. Its generated test model is not the real
ACS bundle and cannot validate real-data inference quality. See
[`FINAL_ENGINEERING_VERIFICATION.md`](FINAL_ENGINEERING_VERIFICATION.md) for actual run
status and measurements.

## C. Internal deployment or external production service — out of scope

No environment-specific TLS, managed identity, centralized rate limits, log retention
and access policy, source-data rights/privacy approval, security approval, or operational
owner has been established. The local API key, in-process rate limiter, and JSON audit
metadata are bounded demonstration controls. Do not expose the service publicly or use
it for real decisions. These organizational gates cannot be inferred from an image build
or a passing container smoke. The release classification remains **Local Demo Only**.
