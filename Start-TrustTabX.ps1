$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$root = $PSScriptRoot
$envFile = Join-Path $root '.env'
$bundle = Join-Path $root 'outputs\m5_acs_model'
$manifestFile = Join-Path $bundle 'manifest.json'
$pipelineFile = Join-Path $bundle 'pipeline.joblib'
$summaryFile = Join-Path $root 'reports\results\m5_acs_artifact_summary.json'
$url = 'http://127.0.0.1:8000/'
$project = 'trusttabx-local'
$composeFile = Join-Path $root 'compose.yaml'
$requiredNames = @('TRUSTTABX_MODEL_DIR_HOST', 'TRUSTTABX_APPROVED_MANIFEST_SHA256', 'TRUSTTABX_API_KEY', 'TRUSTTABX_ADMIN_KEY')

function New-Secret {
    $bytes = New-Object byte[] 48
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try { $rng.GetBytes($bytes) } finally { $rng.Dispose() }
    return [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
}

function Read-LocalConfig {
    $values = @{}
    foreach ($line in [System.IO.File]::ReadAllLines($envFile)) {
        if ($line -match '^\s*(#|$)') { continue }
        $parts = $line.Split('=', 2)
        if ($parts.Count -ne 2 -or $parts[0] -notin $requiredNames -or $values.ContainsKey($parts[0])) {
            throw 'Invalid or duplicate entry in local .env file.'
        }
        $values[$parts[0]] = $parts[1]
    }
    foreach ($name in $requiredNames) {
        if (-not $values.ContainsKey($name) -or [string]::IsNullOrWhiteSpace($values[$name])) {
            throw "Missing $name in local .env file."
        }
    }
    return $values
}

function Assert-PortAvailable {
    $listener = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
    if ($listener) { throw 'Port 8000 is already in use. Stop the other service before launching TrustTab-X.' }
}

Push-Location $root
try {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw 'Docker CLI is not installed or is absent from PATH.' }
    foreach ($path in @($manifestFile, $pipelineFile, $summaryFile)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required approved model file is missing: $path" }
    }
    $manifest = Get-Content -LiteralPath $manifestFile -Raw | ConvertFrom-Json
    $summary = Get-Content -LiteralPath $summaryFile -Raw | ConvertFrom-Json
    $pipelineDigest = (Get-FileHash -LiteralPath $pipelineFile -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($manifest.pipeline_sha256 -ne $pipelineDigest -or $summary.pipeline_sha256 -ne $pipelineDigest -or
        $manifest.model_version -ne $summary.model_version -or $manifest.schema_version -ne $summary.schema_version -or
        $manifest.selected_expert -ne $summary.selected_expert) {
        throw 'The local model bundle does not match the approved research summary.'
    }
    $manifestDigest = (Get-FileHash -LiteralPath $manifestFile -Algorithm SHA256).Hash.ToLowerInvariant()

    if (-not (Test-Path -LiteralPath $envFile)) {
        & git check-ignore -q .env
        if ($LASTEXITCODE -ne 0) { throw '.env is not Git-ignored; refusing to create credentials.' }
        $operatorKey = New-Secret
        $adminKey = New-Secret
        $contents = @(
            'TRUSTTABX_MODEL_DIR_HOST=./outputs/m5_acs_model'
            "TRUSTTABX_APPROVED_MANIFEST_SHA256=$manifestDigest"
            "TRUSTTABX_API_KEY=$operatorKey"
            "TRUSTTABX_ADMIN_KEY=$adminKey"
        ) -join "`n"
        $temporary = Join-Path $root ('.env.' + [Guid]::NewGuid().ToString('N') + '.tmp')
        try {
            [System.IO.File]::WriteAllText($temporary, $contents + "`n", (New-Object System.Text.UTF8Encoding($false)))
            $acl = Get-Acl -LiteralPath $temporary
            $acl.SetAccessRuleProtection($true, $false)
            $identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
            $rule = New-Object System.Security.AccessControl.FileSystemAccessRule($identity, 'FullControl', 'Allow')
            $acl.AddAccessRule($rule)
            Set-Acl -LiteralPath $temporary -AclObject $acl
            Move-Item -LiteralPath $temporary -Destination $envFile
        } finally {
            if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary }
        }
        Write-Host 'Created private, Git-ignored .env with distinct operator and admin keys.'
    }

    $config = Read-LocalConfig
    if ($config.TRUSTTABX_MODEL_DIR_HOST -ne './outputs/m5_acs_model' -or
        $config.TRUSTTABX_APPROVED_MANIFEST_SHA256 -ne $manifestDigest -or
        $config.TRUSTTABX_API_KEY.Length -lt 32 -or $config.TRUSTTABX_ADMIN_KEY.Length -lt 32 -or
        $config.TRUSTTABX_API_KEY -eq $config.TRUSTTABX_ADMIN_KEY) {
        throw 'Local .env model approval or credential validation failed. No configuration was overwritten.'
    }
    foreach ($name in $requiredNames) { [Environment]::SetEnvironmentVariable($name, $config[$name], 'Process') }

    try { & docker info --format '{{.ServerVersion}}' 2>$null | Out-Null } catch { $global:LASTEXITCODE = 1 }
    if ($LASTEXITCODE -ne 0) {
        if (Get-Process 'Docker Desktop' -ErrorAction SilentlyContinue) {
            throw 'Docker Desktop is running but its Engine is unavailable. Check Docker Desktop status and access.'
        }
        $desktop = Join-Path $env:USERPROFILE 'AppData\Local\Programs\DockerDesktop\Docker Desktop.exe'
        if (-not (Test-Path -LiteralPath $desktop)) {
            $desktop = Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'
        }
        if (-not (Test-Path -LiteralPath $desktop)) { throw 'Docker Engine is unavailable and Docker Desktop could not be located.' }
        Start-Process -FilePath $desktop -WindowStyle Hidden
        $engineDeadline = (Get-Date).AddMinutes(3)
        do {
            Start-Sleep -Seconds 3
            try { & docker info --format '{{.ServerVersion}}' 2>$null | Out-Null } catch { $global:LASTEXITCODE = 1 }
        } until ($LASTEXITCODE -eq 0 -or (Get-Date) -ge $engineDeadline)
        if ($LASTEXITCODE -ne 0) { throw 'Docker Engine did not become ready within three minutes.' }
    }

    & docker compose --file $composeFile --project-name $project --env-file $envFile config --quiet
    if ($LASTEXITCODE -ne 0) { throw 'Compose configuration validation failed.' }
    $containerId = & docker compose --file $composeFile --project-name $project --env-file $envFile ps -a -q trusttabx
    if ($LASTEXITCODE -ne 0) { throw 'Could not inspect the TrustTab-X Compose project.' }
    if (-not $containerId) {
        Assert-PortAvailable
        & docker compose --file $composeFile --project-name $project --env-file $envFile up -d --build trusttabx
    } else {
        $running = & docker inspect --format '{{.State.Running}}' $containerId
        if ($LASTEXITCODE -ne 0) { throw 'Could not inspect the existing TrustTab-X container.' }
        if ($running -ne 'true') { Assert-PortAvailable }
        & docker compose --file $composeFile --project-name $project --env-file $envFile up -d --no-recreate trusttabx
    }
    if ($LASTEXITCODE -ne 0) { throw 'Docker Compose could not start TrustTab-X.' }

    $ready = $false
    $deadline = (Get-Date).AddMinutes(2)
    do {
        try {
            $health = Invoke-RestMethod -Uri ($url + 'health') -TimeoutSec 3
            if ($health.status -eq 'ready') { $ready = $true }
        } catch { }
        if (-not $ready) { Start-Sleep -Seconds 2 }
    } until ($ready -or (Get-Date) -ge $deadline)
    if (-not $ready) { throw 'TrustTab-X did not become healthy within two minutes.' }
    $model = Invoke-RestMethod -Uri ($url + 'model') -Headers @{ 'X-TrustTabX-Key' = $config.TRUSTTABX_API_KEY } -TimeoutSec 10
    if ($model.model_version -ne $manifest.model_version) { throw 'Running model version differs from the approved artifact.' }
    Write-Host "TrustTab-X is ready at $url using model $($model.model_version)."
    Start-Process -FilePath $url
    Write-Host 'Enter the operator key from the local .env file into the dashboard Unlock field.'
} catch {
    Write-Error $_.Exception.Message
    exit 1
} finally {
    Pop-Location
}
