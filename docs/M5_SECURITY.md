# M5.1 local security and audit policy

## Access profiles

The service fails startup unless `TRUSTTABX_API_KEY` and `TRUSTTABX_ADMIN_KEY` are distinct strings of at least 32 characters. Supply randomly generated values through environment variables or a deployment secret store. Never commit them. The operator key permits model metadata, telemetry, prediction, batch prediction and drift inspection. The admin key also permits offline delayed-label evaluation. `/health`, `/docs`, `/openapi.json` and the dashboard shell are public on the bound local interface; they do not provide model data or scoring without a key. The key is sent as `X-TrustTabX-Key`. An invalid or missing key returns 401; an operator attempting an admin operation receives 403.

The browser dashboard asks for an operator key after page load, keeps it only in that tab's JavaScript memory and clears the input; reload clears the key. The key is not embedded in the served HTML or persisted to local storage or cookies. This is suitable for a loopback demo. A shared internal deployment needs TLS plus a managed identity or reverse-proxy session gateway before browser access; this release has neither. The Compose host port is bound to `127.0.0.1` only. Direct `docker run -p` commands must preserve that binding.

## Resource and request controls

- Default maximum body: 262,144 bytes, enforced before JSON parsing even without `Content-Length`. Configurable with `TRUSTTABX_MAX_REQUEST_BYTES` (4 KiB to 1 MiB). Prediction batches remain capped at 100; drift batches at 1000.
- Default processing/body-read timeout: 15 seconds via `TRUSTTABX_REQUEST_TIMEOUT_SECONDS` (1–120). Uvicorn idle keep-alive timeout is 5 seconds. A cancelled synchronous model call may continue briefly in a worker thread; this limit is not a hard CPU kill.
- Default rate: 120 requests/minute per process and credential role, including a shared invalid-credential bucket, via `TRUSTTABX_RATE_LIMIT_PER_MINUTE`. It is an abuse guard for one local process, not a distributed quota. Multiple workers or ingress nodes require a gateway limiter.
- Hosts default to `localhost` and `127.0.0.1`; tests explicitly add `testserver`. Set `TRUSTTABX_ALLOWED_HOSTS` for a reviewed internal hostname. Wildcard hosts are rejected. No cross-origin CORS grants are installed, and browser `Origin` must match the request base URL for authenticated operations. Browser clients should use the same origin.
- Security responses are structured and avoid raw input values. `Cache-Control: no-store`, `X-Content-Type-Options: nosniff` and `Referrer-Policy: no-referrer` are set on application responses. The dashboard serves external local JS/CSS assets with a restrictive same-origin Content Security Policy and no inline scripts/styles. The generated FastAPI `/docs` page may depend on external Swagger assets and is outside that dashboard CSP; disable or self-host it for a locked-down internal deployment.

## Model artifact approval

`TRUSTTABX_TRUST_LOCAL_ARTIFACT=1` is required before pickle deserialization. An independent deployment value, `TRUSTTABX_APPROVED_MANIFEST_SHA256`, must equal the SHA256 of the approved local manifest. The manifest also pins the pipeline SHA256, sklearn version, feature order/types, schema and model version. The operator should compute/record the manifest digest after training in an approved location, then pass it through deployment configuration. A self-contained bundle hash alone is not an authenticity guarantee. Do not approve a bundle received from an untrusted party or run the service with a writable model mount. The Compose mount is read-only. Changing either manifest or pipeline after approval fails startup. The exact model binary and raw ACS data remain ignored by Git.

## Audit and retention

One JSON line per HTTP request is emitted to process stderr with a generated request ID, UTC timestamp, matched route pattern (or `unmatched`), method, status, authenticated role, latency, model version, pipeline hash prefix, first decision ID where applicable, drift alert and validation-error flag. Raw URL paths, feature values, labels, probabilities, API keys and request bodies are not logged. API responses carry `X-Request-ID`; individual prediction responses also carry `decision_id`. Telemetry remains bounded and in memory, resetting on restart. No automatic audit export or persistent store is enabled.

For any internal deployment, the operator must restrict log access, encrypt retained logs, set a documented retention limit (suggested maximum seven days for this demo), handle deletion requests and review permitted metadata with privacy/security owners. This local repository cannot enforce the external log collector's policy. Service drift alerts are unlabeled diagnostics; they do not imply measured accuracy loss. Delayed-label metrics remain a separate admin-only offline operation.

## Scanning and release limits

The Windows project venv was checked with `pip check` and `pip-audit 2.10.1` across 134 installed distributions. Twelve advisories all affected local `pip` 25.0.1; after upgrading to `pip` 26.2.2 the audit found zero known vulnerabilities in auditable packages at scan time. `pip-audit` skipped the local `trusttabx` package because it is not on PyPI; its source is covered by tests and review, not by that vulnerability database. Docker pins the fixed pip version, but image/container vulnerability scanning awaits a Docker runtime. Dependency advisories can change. Authentication and rate limits here do not replace TLS, network policy, centralized identity, durable audit governance or a full security assessment.
