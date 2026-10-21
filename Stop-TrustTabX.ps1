$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

Push-Location $PSScriptRoot
try {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw 'Docker CLI is not installed or is absent from PATH.' }
    $envFile = Join-Path $PSScriptRoot '.env'
    if (-not (Test-Path -LiteralPath $envFile)) { throw 'Local .env file is missing; nothing was changed.' }
    try { & docker info --format '{{.ServerVersion}}' 2>$null | Out-Null } catch { $global:LASTEXITCODE = 1 }
    if ($LASTEXITCODE -ne 0) { throw 'Docker Engine is unavailable; TrustTab-X cannot be stopped through Compose.' }
    & docker compose --file (Join-Path $PSScriptRoot 'compose.yaml') --project-name trusttabx-local --env-file $envFile stop trusttabx
    if ($LASTEXITCODE -ne 0) { throw 'Docker Compose could not stop TrustTab-X.' }
    Write-Host 'TrustTab-X stopped. The container, model bundle and local credentials were preserved.'
} catch {
    Write-Error $_.Exception.Message
    exit 1
} finally {
    Pop-Location
}
