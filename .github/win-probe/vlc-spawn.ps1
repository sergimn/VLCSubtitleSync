$ErrorActionPreference = 'Stop'
$u = 'https://download.videolan.org/pub/videolan/vlc/3.0.21/win64/vlc-3.0.21-win64.zip'
Invoke-WebRequest -UseBasicParsing $u -OutFile vlc.zip
Expand-Archive vlc.zip "$env:RUNNER_TEMP\vlczip"
New-Item -ItemType Directory -Force "$env:ProgramFiles\VideoLAN" | Out-Null
Move-Item "$env:RUNNER_TEMP\vlczip\vlc-3.0.21" "$env:ProgramFiles\VideoLAN\VLC"
$vlc = "$env:ProgramFiles\VideoLAN\VLC\vlc.exe"

Remove-Item Env:PSModulePath
$env:VLC_SUBSYNC_SOURCE = (Get-Location).Path
& powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 --no-model
if ($LASTEXITCODE -ne 0) { throw "install failed $LASTEXITCODE" }

Write-Host "uv tool dir: $(& "$env:USERPROFILE\.local\bin\uv.exe" tool dir)"
$launcher = "$env:APPDATA\vlc\subsync\launcher"
Get-Content $launcher
if (-not ((Get-Content $launcher) -match '^exe=.*pythonw\.exe$')) { throw 'launcher does not use pythonw.exe' }

$p = Start-Process $vlc -ArgumentList '-I', 'dummy', '--dummy-quiet', 'vlc://pause:25', 'vlc://quit' -PassThru
$seen = $false
for ($i = 0; $i -lt 20; $i++) {
    Start-Sleep 1
    $h = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'vlcsubsync' }
    if ($h) { $seen = $true; $h | Format-List ProcessId, Name, CommandLine | Out-String | Write-Host; break }
}
& "$env:USERPROFILE\.local\bin\vlc-subsync.exe" doctor
if (-not $seen) { throw 'VLC did not start the helper' }
$p.WaitForExit()
Write-Host "VLC exited"
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep 1
    if (-not (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'vlcsubsync' })) {
        Write-Host "helper exited $i s after VLC"; exit 0
    }
}
throw 'helper still running 60 s after VLC exited'
