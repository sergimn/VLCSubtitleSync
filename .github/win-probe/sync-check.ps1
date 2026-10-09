# Run the installed CLI on a fixture: proves Python + faster-whisper + ctranslate2 + PyAV load.
param([string]$Fix)
$exe = Join-Path $env:USERPROFILE '.local\bin\vlc-subsync.exe'
Write-Host "== $exe exists: $(Test-Path $exe)"
& $exe --version
& $exe doctor
Write-Host "== doctor exit $LASTEXITCODE"
$out = Join-Path $env:TEMP 'out.srt'
$t = Measure-Command { & $exe sync (Join-Path $Fix 'en_dialogue.mkv') --sub-file (Join-Path $Fix 'en_dialogue.offset_plus_3_2.srt') -o $out | Out-Host }
Write-Host "== sync exit $LASTEXITCODE after $([int]$t.TotalSeconds) s"
if (Test-Path $out) {
  Write-Host '== synced (first cues):'; Get-Content $out -TotalCount 12
  Write-Host '== truth (first cues):'; Get-Content (Join-Path $Fix 'en_dialogue.truth.srt') -TotalCount 12
  Write-Host '== input (first cues):'; Get-Content (Join-Path $Fix 'en_dialogue.offset_plus_3_2.srt') -TotalCount 12
  'SYNC_OK'
} else { 'SYNC_FAILED' }
