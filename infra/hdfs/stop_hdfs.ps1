<#
Stop HDFS, then close Docker Desktop and the WSL VM so they stop using RAM. The HDFS data is kept.
Add -KeepDocker to leave Docker Desktop open.

Run:  powershell -ExecutionPolicy Bypass -File infra/hdfs/stop_hdfs.ps1 [-KeepDocker]
#>
param([switch]$KeepDocker)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$ErrorActionPreference = "Continue"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    $bin = Join-Path $env:LOCALAPPDATA "Programs\DockerDesktop\resources\bin"
    if (Test-Path $bin) { $env:Path += ";$bin" }
}

docker info *> $null
if ($LASTEXITCODE -eq 0) {
    Push-Location $here
    try { docker compose stop -t 5 } finally { Pop-Location }
}

if (-not $KeepDocker) {
    # The containers are already stopped, so closing the app is safe and much faster than `docker desktop stop`.
    Get-Process "Docker Desktop", "com.docker.backend", "com.docker.build", "docker-sandbox*" -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue
    wsl --shutdown
    Write-Host "Docker Desktop closed."
}
