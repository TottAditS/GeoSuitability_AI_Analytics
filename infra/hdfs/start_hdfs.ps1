<#
Start HDFS on demand: open Docker Desktop if it is not running, start the containers, wait until HDFS is ready.
The data from the last session is still there (it is only lost after `docker compose down`).

Run:  powershell -ExecutionPolicy Bypass -File infra/hdfs/start_hdfs.ps1
#>
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$ErrorActionPreference = "Continue"   # docker prints progress to stderr

$desktop = Join-Path $env:LOCALAPPDATA "Programs\DockerDesktop"
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    $bin = Join-Path $desktop "resources\bin"
    if (Test-Path $bin) { $env:Path += ";$bin" }
}
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "docker not found: install Docker Desktop (WSL 2 backend) first, see infra/hdfs/README.md"
}

docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Starting Docker Desktop..."
    Start-Process (Join-Path $desktop "Docker Desktop.exe")
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 3
        docker info *> $null
        if ($LASTEXITCODE -eq 0) { break }
    }
    if ($LASTEXITCODE -ne 0) { throw "Docker did not start within 3 minutes" }
}

Push-Location $here
try {
    docker compose up -d
    if ($LASTEXITCODE -ne 0) { throw "docker compose up failed" }

    Write-Host "Waiting for HDFS (up to about 2 minutes)..."
    $ready = $false
    for ($i = 0; $i -lt 40; $i++) {
        $out = (docker compose exec -T namenode hdfs dfsadmin -safemode get 2>&1 | Out-String)
        if ($out -match "Safe mode is (OFF|ON)") { $ready = $true; break }
        Start-Sleep -Seconds 3
    }
    if (-not $ready) {
        docker compose logs --tail 40 namenode
        throw "the NameNode did not start; see the log above"
    }
    # Safe mode lasts until the DataNode reports in.
    docker compose exec -T namenode hdfs dfsadmin -safemode wait 2>&1 | Out-Null
    Write-Host "HDFS is ready. NameNode UI: http://localhost:9870"
} finally {
    Pop-Location
}
