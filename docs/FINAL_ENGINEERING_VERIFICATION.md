# Final engineering verification — M8

This record distinguishes executed checks from pending hosted-container evidence. The
research decision remains **M4.2 C: no adaptive advantage demonstrated**; the service
uses the source-selected fixed expert. Release scope is in [`RELEASE_SCOPE.md`](RELEASE_SCOPE.md).

## Executed on 21 October 2026

- Windows Python 3.12.10: `pytest -q -p no:cacheprovider --basetemp=.pytest_tmp`
  passed **43 tests**. One existing Starlette/httpx deprecation warning remains.
- `ruff check src tests scripts`, `pip check`, `scripts/verify_m6.py`,
  `scripts/verify_m7.py`, and `git diff --check` passed.
- `pip-audit --local` found no known vulnerabilities in auditable installed packages;
  the unpublished local `trusttabx` package was skipped.
- All six tracked PNGs passed image decoding. The manuscript PDF is six A4 pages;
  portfolio copy/hash/link and research-claim checks passed. A local wheel built.
- A real loopback Uvicorn process loaded an ignored **synthetic** fixture under
  temporary operator/admin keys and an approved manifest digest. The bounded
  20-request smoke passed: startup 3.33 s, HTTP p50 20.62 ms, p95 30.97 ms,
  51.59 requests/s, process-tree RSS 183.72 MiB. Its one service error is the
  intentionally rejected unauthenticated request. This is one local run, not a
  deployment performance guarantee.
- Static Compose checks passed. Docker is unavailable on this Windows host.

## GitHub-hosted Docker runtime gate

The dedicated manual workflow `.github/workflows/m5-container.yml` targets
`ubuntu-24.04`, checks out the triggering commit, verifies Docker/Compose, builds
the actual image, and runs `scripts/verify_m5_container.py --runtime`. The harness
generates only a synthetic bundle and temporary credentials, runs the service via
Compose, tests authenticated and denied requests, dashboard, drift, delayed labels,
malformed input, tampered manifest/pipeline, restart, nonroot/read-only/resource
limits, and clean shutdown. It prints only status codes and aggregate timings/resource
data; failure diagnostics redact generated keys and the test feature marker.

**Pending:** account-level included Actions minutes and a no-charge spending limit
could not be read with the current GitHub CLI token (`user` scope required; billing
endpoint returned HTTP 404). No hosted M8 container run is claimed until that
no-cost gate is confirmed and the workflow succeeds. Replace this paragraph with
the exact branch, commit, run URL, Docker/Compose versions, status table, image size,
resource use and conclusion after execution.

## Security and operational boundary

The new negative regression test verifies that rejected sensitive feature values,
labels, operator keys and admin keys do not appear in API errors or JSON audit
records. Existing tests cover startup fail-closed behavior, approved artifact
digest/hash, role authorization, origin/host/body limits, per-process rate limiting,
CSP, and timeouts. The runtime workflow must still confirm Linux behavior.

No public endpoint or production approval exists. TLS, managed identity, centralized
rate limits, audit retention/access policy, source-data rights/privacy review,
security signoff, and an operational owner remain outside this local demonstration.
