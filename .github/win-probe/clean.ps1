# Runs inside a fresh Windows Server Core container: no Python, no VC++ redist, no ffmpeg.
$ErrorActionPreference = 'Continue'
Write-Host "== OS: $([Environment]::OSVersion.VersionString)  arch: $env:PROCESSOR_ARCHITECTURE"
foreach ($d in 'msvcp140.dll','vcruntime140.dll','vcruntime140_1.dll','msvcp140_1.dll','vcomp140.dll') {
  Write-Host "== System32\$d present: $(Test-Path (Join-Path $env:WINDIR "System32\$d"))"
}
foreach ($c in 'python','py','uv','ffmpeg','vlc') { Write-Host "== $c on PATH: $([bool](Get-Command $c -ErrorAction SilentlyContinue))" }
& C:\probe\install-release.ps1
& C:\probe\sync-check.ps1 -Fix C:\fix
