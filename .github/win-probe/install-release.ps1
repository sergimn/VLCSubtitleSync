# Download the published SubSync zip and run its install.cmd as a user would (double-click).
param([string]$Version = '1.0.0')
$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$zip = Join-Path $env:TEMP "SubSync-$Version.zip"
Invoke-WebRequest -UseBasicParsing "https://github.com/sergimn/VLCSubtitleSync/releases/download/v$Version/SubSync-$Version.zip" -OutFile $zip
$dir = Join-Path $env:TEMP 'subsync-zip'
Expand-Archive $zip $dir -Force
$cmd = Join-Path $dir "SubSync-$Version\install.cmd"
Write-Host "== running $cmd"
$t = Measure-Command { cmd /c "`"$cmd`" < NUL" | Out-Host }
Write-Host "== install.cmd exit code $LASTEXITCODE after $([int]$t.TotalSeconds) s"
