# Final engineering verification — M8

**Decision:** COMPLETE — research portfolio and container-verified local demo.
The M4.2 scientific result remains **C: no adaptive advantage demonstrated**;
the service uses the source-selected fixed expert. The engineering release remains
**B: Local Demo Only**. See [`RELEASE_SCOPE.md`](RELEASE_SCOPE.md).

## Local verification, 21 October 2026

- Windows Python 3.12.10: `pytest -q -p no:cacheprovider --basetemp=.pytest_tmp`
  passed **43 tests**; one existing Starlette/httpx deprecation warning remains.
- `ruff check src tests scripts`, `pip check`, `scripts/verify_m6.py`,
  `scripts/verify_m7.py`, and `git diff --check` passed. `pip-audit --local`
  found no known vulnerabilities in auditable installed packages and skipped the
  unpublished local `trusttabx` package.
- All six tracked PNGs passed image decoding. The manuscript PDF is six A4 pages;
  portfolio copy/hash/link and research-claim checks passed. A local wheel built.
- A real loopback Uvicorn process loaded an ignored **synthetic** fixture under
  temporary operator/admin keys and an approved manifest digest. A bounded
  20-request smoke passed: startup 3.33 s, HTTP p50 20.62 ms, p95 30.97 ms,
  51.59 requests/s, process-tree RSS 183.72 MiB. Its one service error was
  the intentionally rejected unauthenticated request.

## Actual local Docker and Compose run

Docker Engine **29.9.1** ran Linux containers on WSL 2; Docker Compose was **5.6.0**.
The engine reported 16 CPUs and 8,141,041,664 bytes (about 7.58 GiB) of memory.
`scripts/verify_m5_container.py --runtime` built the actual service image and ran it
through the checked-in Compose file using an ignored synthetic bundle, generated
temporary keys, and a generated independently supplied manifest digest. The first
run completed the integration checks but exposed a Windows console-decoding error in
captured Docker build output. After explicitly decoding subprocess output as UTF-8
with replacement, the **second complete build/run passed without that error**.

Measured second-run results: image size **680,395,478 bytes**; startup to readiness
**1.734 s**; restart to readiness **1.515 s**; nonroot UID **1000**; memory limit
**2,147,483,648 bytes**, sampled memory use **124.7 MiB**; CPU limit **2 CPUs**;
root filesystem and model mount read-only; host port bound to `127.0.0.1`.
Fifty sequential authenticated loopback predictions returned HTTP 200, with
HTTP p50 **16.0 ms**, p95 **31.0 ms**, and **61.5 requests/s**. These are one-host,
one-run smoke measurements, not capacity or portability guarantees.

The harness recorded 200 for health, authorized model, single and batch prediction,
drift, admin delayed labels, metrics, dashboard/assets, and prediction after restart.
Anonymous/wrong-key calls returned 401, operator access to admin delayed labels
returned 403, and oversized batch, malformed feature and invalid delayed label
returned 422. Deliberate age shift raised a drift alert; the synthetic delayed-label
Brier was 0.04. Both manifest digest mismatch and modified pipeline bytes failed
load. Captured container logs excluded generated keys and the test feature marker.
Compose shut down its unique project; no service container remained.

The manual GitHub workflow was **not used**; no Actions or paid cloud minutes were
needed. Passing this synthetic suite completes a **container-verified local demo**.
It does not validate ACS real-data model quality in Linux or grant production approval.

## Security and operational boundary

The negative regression test verifies rejected sensitive feature and label values,
operator keys and admin keys do not appear in API errors or JSON audit records.
Existing tests cover startup fail-closed behavior, approved artifact digest/hash,
role authorization, origin/host/body limits, per-process rate limiting, CSP, and
timeouts. Docker confirmed the listed Linux container paths and failure modes.

No public endpoint or production approval exists. TLS, managed identity, centralized
rate limits, audit retention/access policy, source-data rights/privacy review,
security signoff, an operational owner, and an immutable base-image digest/scan
remain outside this local demonstration.
