@echo off
rem SubSync (vlc-subsync) installer for Windows: double-click to install.
rem Pass --uninstall (or -Uninstall) to remove. Other arguments go to "vlc-subsync setup".
setlocal
rem From a PowerShell 7 prompt, its PSModulePath makes Windows PowerShell load PowerShell 7's
rem modules, which fail ("module could not be loaded"). Let powershell.exe use its defaults.
set "PSModulePath="
set "PS1=%~dp0install.ps1"
if exist "%PS1%" goto run
set "PS1=%TEMP%\vlc-subsync-install.ps1"
echo Downloading the SubSync installer...
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -UseBasicParsing -Uri 'https://raw.githubusercontent.com/sergimn/VLCSubtitleSync/main/install.ps1' -OutFile (Join-Path $env:TEMP 'vlc-subsync-install.ps1')"
if errorlevel 1 goto dlfail

:run
set "ARGS=%*"
if /i "%~1"=="--uninstall" set "ARGS=-Uninstall"
powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%" %ARGS%
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (echo Finished successfully.) else (echo Failed with exit code %RC%.)
pause
exit /b %RC%

:dlfail
echo Could not download install.ps1. Check your internet connection.
pause
exit /b 1
