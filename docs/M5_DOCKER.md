# M5.1 Docker verification and local deployment

The image installs only the `m5-service` runtime extra from `constraints/m5.txt`, runs as `appuser`, uses a read-only root filesystem with `/tmp` tmpfs, and has a health check. Compose mounts an approved model bundle read-only and binds `127.0.0.1:8000`, with 2 GiB memory and 2 CPU limits. The base image remains `python:3.12-slim`; a future release should pin and scan an immutable image digest.

Prepare the environment in PowerShell from `D:\TrustTab-X` after training or providing an approved local bundle:

```powershell
$env:TRUSTTABX_MODEL_DIR_HOST = './outputs/m5_acs_model'
$env:TRUSTTABX_APPROVED_MANIFEST_SHA256 = (Get-FileHash -Algorithm SHA256 -LiteralPath outputs/m5_acs_model/manifest.json).Hash.ToLowerInvariant()
$env:TRUSTTABX_API_KEY = .\.venv\Scripts\python.exe -c 'import secrets; print(secrets.token_urlsafe(48))'
$env:TRUSTTABX_ADMIN_KEY = .\.venv\Scripts\python.exe -c 'import secrets; print(secrets.token_urlsafe(48))'
docker compose config --quiet
docker compose up --build -d
docker compose ps
```

The keys remain private environment values; the operator must transfer the operator key to the dashboard using a secure local channel. `docker compose down` stops this service. Compose rejects absent keys/digest. Do not publish port 8000 on a public interface.

For a synthetic, non-sensitive end-to-end verification, install `.[dev,m5-service]` with the pinned constraints and run:

```powershell
.\.venv\Scripts\python.exe scripts/verify_m5_container.py
.\.venv\Scripts\python.exe scripts/verify_m5_container.py --runtime
```

The first command checks configuration without Docker. The second creates an ignored synthetic bundle, validates Compose, builds/runs the image, exercises authenticated and denied API paths, dashboard, drift and delayed labels, malformed input, tampered artifacts, restart, nonroot/read-only/resource limits and clean shutdown, then records startup/restart, image size and container memory. A manual `workflow_dispatch` CI job at `.github/workflows/m5-container.yml` runs the fixture on pinned `ubuntu-24.04`; it is not automatically triggered by a push. Container runtime checks cannot execute on the current Windows host because Docker/Podman/NerdCTL are unavailable. See [final engineering verification](FINAL_ENGINEERING_VERIFICATION.md) for the hosted run status and actual measurements. Passing this local-demo gate does not approve an internal or public deployment.
