# install.ps1's speech engine check must give up with a clear error, not hang, when an import
# never returns (reported: a PC where the install sat silent after `uv tool install`).
# A fake `av` module that sleeps stands in for whatever holds the real one.
$ErrorActionPreference = 'Stop'
$src = Get-Content (Join-Path $PSScriptRoot '..\..\install.ps1') -Raw
Invoke-Expression $src.Substring(0, $src.IndexOf('# Note: no `exit`'))  # functions only, no install

$py = Join-Path (& uv tool dir --color never | Select-Object -Last 1).Trim() 'vlc-subsync\Scripts\python.exe'
$fake = Join-Path $env:RUNNER_TEMP 'fake-av'
New-Item -ItemType Directory -Force $fake | Out-Null
Set-Content (Join-Path $fake 'av.py') 'import time; time.sleep(3600)'

if ($null -ne (Test-Engine $py)) { throw 'the engine check fails on a good install' }
$env:PYTHONPATH = $fake
$t0 = Get-Date
try {
    Test-Engine $py -TimeoutSec 30
    throw 'the engine check returned instead of failing'
} catch {
    $msg = "$_"
}
$secs = ((Get-Date) - $t0).TotalSeconds
Write-Host "after $([int]$secs) s: $msg"
if ($msg -notmatch '\(av\) did not finish') { throw "unexpected error: $msg" }
if ($secs -gt 60) { throw 'the timeout was not honoured' }
