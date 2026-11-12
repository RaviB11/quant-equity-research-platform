<#
.SYNOPSIS
  Runs the pipeline and writes a timestamped log. Task Scheduler calls this script.
.PARAMETER Mode
  live (default) fetches data; demo runs offline on synthetic data.
.PARAMETER FromStage
  Optional: start from a later stage, e.g. quality, to recompute without refetching.
#>
param(
    [ValidateSet("live", "demo")][string]$Mode = "live",
    [string]$FromStage = ""
)
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Python = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Error "Virtual environment not found at $Python. Run scripts\setup.ps1 first."
    exit 1
}
New-Item -ItemType Directory -Force -Path (Join-Path $Repo "logs") | Out-Null
$Log = Join-Path $Repo ("logs\scheduled_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

$PyArgs = @("-m", "eqscreen.pipeline", "--mode", $Mode)
if ($FromStage) { $PyArgs += @("--from-stage", $FromStage) }

# Continue (not Stop) so Python's stderr lines are logged instead of aborting the script;
# the exit code below decides success.
$ErrorActionPreference = "Continue"
& $Python @PyArgs *>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $Log
$Code = $LASTEXITCODE
if ($Code -ne 0) {
    Write-Error "Pipeline failed with exit code $Code. See $Log"
    exit $Code
}
Write-Host "Pipeline finished. Log: $Log"
