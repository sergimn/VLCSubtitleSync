# Runs inside a fresh Windows Server Core container (no Python, no uv, no Visual C++
# runtime): install.cmd from the checkout must leave a SubSync whose speech engine loads.
# Usage (from ci.yml): docker run -v <checkout>:C:\src ... powershell -File C:\src\.github\ci\clean-windows-install.ps1
$ErrorActionPreference = 'Stop'
$sys = Join-Path $env:WINDIR 'System32'

if (Test-Path (Join-Path $sys 'msvcp140.dll')) {
    throw 'this container already has the Visual C++ runtime; the test would prove nothing'
}
# Server Core lacks Video for Windows (avicap32/msvfw32), which every desktop Windows has and
# PyAV's FFmpeg links. The job copies the runner's into C:\sys so this stands in for a fresh PC.
foreach ($d in Get-ChildItem C:\sys -Filter *.dll) { Copy-Item $d.FullName $sys }

$env:VLC_SUBSYNC_SOURCE = 'C:\src'
cmd /c "C:\src\install.cmd --no-model < NUL"
if ($LASTEXITCODE -ne 0) { throw "install.cmd failed ($LASTEXITCODE)" }
if (-not (Test-Path (Join-Path $sys 'msvcp140.dll'))) { throw 'the Visual C++ runtime was not installed' }

$exe = Join-Path $env:USERPROFILE '.local\bin\vlc-subsync.exe'
& $exe doctor
if ($LASTEXITCODE -ne 0) { throw 'vlc-subsync doctor reported problems' }
