# TrustTab-X M8 closure gate

No new research milestone or model feature is authorized. The M4.2 decision is
**C: no adaptive advantage demonstrated**; M5.1 remains **B: Local Demo Only**.

The one remaining M8 execution gate is the synthetic Docker Compose verification
on the private GitHub repository's `ubuntu-24.04` runner. Before dispatching it,
confirm that included Actions minutes remain and that the account cannot incur
additional charges. The current CLI token lacks billing-read scope; do not assume
that a private workflow is free. The local Windows host has no Docker engine.

The reviewed `codex/trusttabx-m8-closure` branch is already private on GitHub
at `cce7567` and was uploaded with `[skip ci]`; no Actions run started. Once the
no-charge gate is confirmed, run CI on a new non-skipped commit and dispatch the
dedicated `.github/workflows/m5-container.yml` workflow on this branch. Inspect
the job's actual Docker
build/Compose/API/security/resource results, and update the M8 evidence and release
classification. If the container fails, fix only demonstrated defects and rerun
within the verified no-charge limit. If billing access or runtime remains unavailable,
record **PARTIALLY COMPLETE — CONTAINER VERIFICATION BLOCKED** with the exact cause.

Internal or public deployment remains out of scope and needs separate environment,
data-rights, privacy, security and operational approvals. Do not create an M9 task.
