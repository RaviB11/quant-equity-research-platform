<#
.SYNOPSIS
  One-time setup on Windows: virtual environment, dependencies, .env, and a demo run.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
#>
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment in .venv"
    py -3 -m venv .venv
}
$Python = Join-Path $Repo ".venv\Scripts\python.exe"
& $Python -m pip install --upgrade pip
& $Python -m pip install -e ".[dev]"

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example. Add API keys there if you use FMP or Alpha Vantage."
}

Write-Host "Running the offline demo to check the install"
& $Python -m eqscreen.pipeline --mode demo
if ($LASTEXITCODE -ne 0) { throw "Demo run failed (exit $LASTEXITCODE)" }
Write-Host "Done. Open outputs\demo_synthetic\report.html"
