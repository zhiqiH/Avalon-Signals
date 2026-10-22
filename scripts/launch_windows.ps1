<#
.SYNOPSIS
    Start training runs on Windows that keep running after the SSH session closes.

.DESCRIPTION
    Windows OpenSSH ends every process of a session when the connection drops, and there is no
    nohup or tmux. Processes created through WMI (Win32_Process.Create) are spawned by the WMI
    service instead of the SSH session, so they survive disconnects. Each seed runs as its own
    process and writes its console output to logs\<config>_seed<k>_<timestamp>.log; metrics and
    checkpoints go to runs\ as usual.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\launch_windows.ps1 -Seeds 0,1,2

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\launch_windows.ps1 -Config configs\pilot_silent.yaml -Seeds 0 -ExtraArgs "--total-games 512"
#>
param(
    [string]$Config = "configs\baseline_silent.yaml",
    [string]$Seeds = "0",          # comma-separated, e.g. "0,1,2"
    [string]$Device = "cuda",
    [int]$Threads = 2,             # torch CPU threads per run; keep low when running seeds in parallel
    [string]$ExtraArgs = ""        # passed through to scripts\train.py
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Virtual environment not found: $python" }
if (-not (Test-Path (Join-Path $repo $Config))) { throw "Config not found: $Config" }

$logDir = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$configName = [IO.Path]::GetFileNameWithoutExtension($Config)

foreach ($seed in ($Seeds -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })) {
    $log = Join-Path $logDir "${configName}_seed${seed}_${stamp}.log"
    $train = "`"$python`" -u scripts\train.py --config $Config --seed $seed --device $Device --threads $Threads $ExtraArgs"
    # cmd /s /c strips exactly one outer pair of quotes, so quoted paths inside survive.
    $commandLine = "cmd.exe /s /c `"$train > `"$log`" 2>&1`""
    $result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create `
        -Arguments @{ CommandLine = $commandLine; CurrentDirectory = $repo }
    if ($result.ReturnValue -ne 0) { throw "Failed to start seed $seed (Win32_Process.Create returned $($result.ReturnValue))" }
    Write-Output "seed $seed started: PID $($result.ProcessId), log $log"
}

Write-Output ""
Write-Output "Follow a log:   Get-Content <log> -Wait -Tail 5"
Write-Output "List runs:      Get-CimInstance Win32_Process -Filter `"Name='python.exe'`" | Select-Object ProcessId, CommandLine"
Write-Output "Stop one run:   taskkill /PID <PID above> /T /F"
