<#
.SYNOPSIS
    Start training (or any other Python job) on Windows so that it keeps running after the SSH session closes.

.DESCRIPTION
    Windows OpenSSH ends every process of a session when the connection drops, and there is no
    nohup or tmux. Processes created through WMI (Win32_Process.Create) are spawned by the WMI
    service instead of the SSH session, so they survive disconnects. Each job runs as its own
    process and writes its console output to logs\<name>_<timestamp>.log.

    Default mode starts scripts\train.py once per seed. With -Raw, starts exactly one job running
    `python -u <Raw>` (e.g. a best-response run) and names its log after -Name.

    Python comes from the conda environment named -EnvName (default avalon-signals), looked up in
    the usual conda locations, or from -Python if given. The environment does not need to be
    activated: its folders are put on PATH for the started process.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\launch_windows.ps1 -Config configs\baseline_eq.yaml -Seeds 0,1,2 -Device cpu -Threads 4

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\launch_windows.ps1 -Name br_seed0_evil -Threads 4 -Raw "scripts\best_response.py --checkpoint runs\baseline_eq\seed0\latest.pt --side evil"
#>
param(
    [string]$Config = "configs\baseline_silent.yaml",
    [string]$Seeds = "0",          # comma-separated, e.g. "0,1,2"
    [string]$Device = "cuda",
    [int]$Threads = 2,             # torch CPU threads per run; keep low when running seeds in parallel
    [string]$ExtraArgs = "",       # passed through to scripts\train.py
    [string]$Raw = "",             # run `python -u <Raw>` once instead of train.py per seed
    [string]$Name = "job",         # log name for -Raw jobs
    [string]$EnvName = "avalon-signals",
    [string]$Python = ""           # explicit python.exe; overrides the conda lookup
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
if (-not $Python) {
    $roots = @("$HOME\anaconda3", "$HOME\miniconda3", "$HOME\miniforge3", "$env:ProgramData\anaconda3", "$env:ProgramData\miniconda3")
    $Python = $roots | ForEach-Object { Join-Path $_ "envs\$EnvName\python.exe" } | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $Python -or -not (Test-Path $Python)) { throw "Python for environment '$EnvName' not found; pass -Python <path to python.exe>" }
$envDir = Split-Path -Parent $Python
Write-Output "python: $Python"

$logDir = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
# Same effect as activating the conda env; PYTHONUTF8 keeps file and console IO in UTF-8; OMP/MKL threads follow -Threads.
# Quoted `set "VAR=value"` keeps the space before && out of the value.
$setup = "set `"PATH=$envDir;$envDir\Library\bin;$envDir\Scripts;%PATH%`" && set `"PYTHONUTF8=1`" && set `"OMP_NUM_THREADS=$Threads`""

function Start-Detached([string]$job, [string]$log, [string]$label) {
    # cmd /s /c strips exactly one outer pair of quotes, so quoted paths inside survive.
    $commandLine = "cmd.exe /s /c `"$setup && $job > `"$log`" 2>&1`""
    $result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create `
        -Arguments @{ CommandLine = $commandLine; CurrentDirectory = $repo }
    if ($result.ReturnValue -ne 0) { throw "Failed to start $label (Win32_Process.Create returned $($result.ReturnValue))" }
    Write-Output "$label started: PID $($result.ProcessId), log $log"
}

if ($Raw) {
    Start-Detached "`"$Python`" -u $Raw" (Join-Path $logDir "${Name}_${stamp}.log") $Name
} else {
    if (-not (Test-Path (Join-Path $repo $Config))) { throw "Config not found: $Config" }
    $configName = [IO.Path]::GetFileNameWithoutExtension($Config)
    foreach ($seed in ($Seeds -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })) {
        $train = "`"$Python`" -u scripts\train.py --config $Config --seed $seed --device $Device --threads $Threads $ExtraArgs"
        Start-Detached $train (Join-Path $logDir "${configName}_seed${seed}_${stamp}.log") "seed $seed"
    }
}

Write-Output ""
Write-Output "Follow a log:   Get-Content <log> -Wait -Tail 5"
Write-Output "List jobs:      Get-CimInstance Win32_Process -Filter `"Name='python.exe'`" | Select-Object ProcessId, CommandLine"
Write-Output "Stop one job:   taskkill /PID <PID above> /T /F"
