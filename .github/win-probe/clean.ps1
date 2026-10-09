# Runs inside a fresh Windows Server Core container: no Python, no VC++ redist, no ffmpeg.
$ErrorActionPreference = 'Continue'
Write-Host "== OS: $([Environment]::OSVersion.VersionString)  arch: $env:PROCESSOR_ARCHITECTURE"
foreach ($d in 'msvcp140.dll','vcruntime140.dll','vcruntime140_1.dll','msvcp140_1.dll','vcomp140.dll') {
  Write-Host "== System32\$d present: $(Test-Path (Join-Path $env:WINDIR "System32\$d"))"
}
foreach ($c in 'python','py','uv','ffmpeg','vlc') { Write-Host "== $c on PATH: $([bool](Get-Command $c -ErrorAction SilentlyContinue))" }
& C:\probe\install-release.ps1
& C:\probe\imports.ps1
& C:\probe\sync-check.ps1 -Fix C:\fix
Write-Host '############ installing the Microsoft Visual C++ 2015-2022 x64 redistributable'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Invoke-WebRequest -UseBasicParsing https://aka.ms/vs/17/release/vc_redist.x64.exe -OutFile C:\vc_redist.x64.exe
$p = Start-Process C:\vc_redist.x64.exe -ArgumentList '/install','/quiet','/norestart' -Wait -PassThru
Write-Host "== vc_redist exit $($p.ExitCode); msvcp140 now: $(Test-Path (Join-Path $env:WINDIR 'System32\msvcp140.dll'))"
& C:\probe\imports.ps1
$exe = Join-Path $env:USERPROFILE '.local\bin\vlc-subsync.exe'
& $exe download-models
& C:\probe\sync-check.ps1 -Fix C:\fix
