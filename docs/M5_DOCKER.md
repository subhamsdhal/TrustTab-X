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

The first command checks configuration without Docker. The second creates an ignored synthetic bundle, validates Compose, builds/runs the image, checks authentication and all main endpoints, confirms nonroot/read-only/resource limits, rejects a mismatched approved digest, restarts the service, measures startup/restart, image size and container memory, then stops only its own uniquely named Compose project. A manual `workflow_dispatch` CI job at `.github/workflows/m5-container.yml` runs the same fixture workflow on an Ubuntu Docker host; it is not automatically triggered by a push. Container runtime checks were **not executed** on the current Windows host because Docker/Podman/NerdCTL are unavailable. No startup/image-size/container-memory claim is made yet.
