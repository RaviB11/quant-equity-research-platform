<#
.SYNOPSIS
  Registers a monthly Windows scheduled task that runs scripts\run_monthly.ps1.
.DESCRIPTION
  New-ScheduledTaskTrigger has no monthly option, so this uses the Task Scheduler
  COM API, which supports monthly triggers directly and avoids schtasks.exe quoting
  problems with paths that contain spaces. The task runs as the current user while
  logged on, and StartWhenAvailable catches up if the PC was off at the scheduled time.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\register_monthly_task.ps1 -Day 2 -Time 06:00
.EXAMPLE
  # Remove it again
  Unregister-ScheduledTask -TaskName EquityScreenerMonthly -Confirm:$false
#>
param(
    [ValidateRange(1, 28)][int]$Day = 2,
    [ValidatePattern("^\d{2}:\d{2}$")][string]$Time = "06:00",
    [string]$TaskName = "EquityScreenerMonthly"
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Script = Join-Path $Repo "scripts\run_monthly.ps1"
$Hour, $Minute = $Time.Split(":") | ForEach-Object { [int]$_ }

$Service = New-Object -ComObject Schedule.Service
$Service.Connect()
$Task = $Service.NewTask(0)
$Task.RegistrationInfo.Description = "Equity screener: fetch data, recompute factors, backtest, report"
$Task.Settings.StartWhenAvailable = $true
$Task.Settings.ExecutionTimeLimit = "PT3H"
$Task.Settings.DisallowStartIfOnBatteries = $false

$Trigger = $Task.Triggers.Create(4)                       # 4 = TASK_TRIGGER_MONTHLY
$Trigger.DaysOfMonth = [int][math]::Pow(2, $Day - 1)     # bit mask: day 1 = 1, day 2 = 2, day 3 = 4 ...
$Trigger.MonthsOfYear = 0xFFF                             # every month
$Trigger.StartBoundary = (Get-Date -Hour $Hour -Minute $Minute -Second 0).ToString("yyyy-MM-dd'T'HH:mm:ss")

$Action = $Task.Actions.Create(0)                         # 0 = TASK_ACTION_EXEC
$Action.Path = "powershell.exe"
$Action.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$Script`""
$Action.WorkingDirectory = $Repo

# 6 = TASK_CREATE_OR_UPDATE, 3 = TASK_LOGON_INTERACTIVE_TOKEN (current user, no stored password)
$Service.GetFolder("\").RegisterTaskDefinition($TaskName, $Task, 6, $null, $null, 3) | Out-Null
Write-Host "Registered '$TaskName': day $Day of every month at $Time"
Write-Host "Test it now with: Start-ScheduledTask -TaskName $TaskName"
