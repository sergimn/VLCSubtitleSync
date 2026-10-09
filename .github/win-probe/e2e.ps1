# Real VLC as configured by the installer (vlcrc in %APPDATA%\vlc): play a file whose sidecar
# subtitle is 3.2 s late and wait for the helper to be started by VLC and the synced track added.
param([string]$Fix, [int]$TimeoutSec = 900)
$vlc = @("$env:ProgramFiles\VideoLAN\VLC\vlc.exe", "${env:ProgramFiles(x86)}\VideoLAN\VLC\vlc.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
Write-Host "== vlc: $vlc"
$m = Join-Path $env:TEMP 'e2e media'
New-Item -ItemType Directory -Force $m | Out-Null
Copy-Item (Join-Path $Fix 'en_dialogue.mkv') $m
Copy-Item (Join-Path $Fix 'en_dialogue.offset_plus_3_2.srt') (Join-Path $m 'en_dialogue.srt')
$log = Join-Path $env:TEMP 'vlc.log'
Remove-Item $log -ErrorAction SilentlyContinue
Write-Host '== vlcrc subsync lines:'; Select-String -Path "$env:APPDATA\vlc\vlcrc" -Pattern '^(lua-intf|extraintf)=' | ForEach-Object { $_.Line }
Write-Host '== helper running before VLC:'; Get-Process vlc-subsync-daemon -ErrorAction SilentlyContinue
$vargs = @('-I','dummy','--dummy-quiet','--file-logging',"--logfile=$log",'--log-verbose=2',
  '--vout','dummy','--aout','dummy','--sub-track','0','--input-repeat','200',
  '--no-metadata-network-access',"`"$m\en_dialogue.mkv`"")
$p = Start-Process $vlc -ArgumentList $vargs -PassThru
$deadline = (Get-Date).AddSeconds($TimeoutSec); $ok = $false; $sawHelper = $false
while ((Get-Date) -lt $deadline) {
  Start-Sleep 3
  if (Get-Process vlc-subsync-daemon -ErrorAction SilentlyContinue) { $sawHelper = $true }
  $text = if (Test-Path $log) { Get-Content $log -Raw -ErrorAction SilentlyContinue } else { '' }
  if ($text -match '\[subsync\] synced track es=') { $ok = $true; break }
  if ($p.HasExited) { Write-Host "== VLC exited early ($($p.ExitCode))"; break }
}
Write-Host "== helper process seen while VLC ran: $sawHelper"
Write-Host '== VLC log, subsync lines:'
if (Test-Path $log) { Select-String -Path $log -Pattern 'subsync' | Select-Object -Last 60 | ForEach-Object { $_.Line } }
Write-Host '== out dir:'; Get-ChildItem "$env:APPDATA\vlc\subsync\out" -ErrorAction SilentlyContinue
Get-ChildItem "$env:APPDATA\vlc\subsync\out\*.srt" -ErrorAction SilentlyContinue | Select-Object -First 1 | ForEach-Object { Get-Content $_ -TotalCount 12 }
if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
Write-Host '== waiting for the helper to quit after VLC'
$gone = $false
foreach ($i in 1..60) { if (-not (Get-Process vlc-subsync-daemon -ErrorAction SilentlyContinue)) { $gone = $true; break }; Start-Sleep 3 }
Write-Host "== helper exited after VLC: $gone"
Write-Host '== helper logs:'
Get-ChildItem "$env:LOCALAPPDATA\vlc-subsync" -Recurse -Filter *.log -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "-- $($_.FullName)"; Get-Content $_ -Tail 60 }
if ($ok) { 'E2E_OK' } else { 'E2E_FAILED'; exit 1 }
