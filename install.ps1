# SubSync (vlc-subsync) installer for Windows.
#
#   powershell -ExecutionPolicy Bypass -File install.ps1 [-Uninstall] [setup options]
#   irm https://raw.githubusercontent.com/sergimn/VLCSubtitleSync/main/install.ps1 | iex
#
# Environment:
#   VLC_SUBSYNC_SOURCE     package source (default: GitHub main branch archive);
#                          may be a local checkout directory for testing
#   VLC_SUBSYNC_PYTHON     Python version (default 3.12)
#   VLC_SUBSYNC_MIN_AGE_DAYS  only install dependency releases at least this many
#                          days old (default 14, supply-chain protection; 0 disables)
#   VLC_SUBSYNC_UNINSTALL  set to 1 to uninstall (for `irm | iex`, which takes no args)

param(
    [switch]$Uninstall,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$SetupArgs
)

$ErrorActionPreference = 'Stop'
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}

$DefaultSource = 'https://github.com/sergimn/VLCSubtitleSync/archive/refs/heads/main.zip'
$Source = if ($env:VLC_SUBSYNC_SOURCE) { $env:VLC_SUBSYNC_SOURCE } else { $DefaultSource }
$PyVer = if ($env:VLC_SUBSYNC_PYTHON) { $env:VLC_SUBSYNC_PYTHON } else { '3.12' }
$MinAgeDays = if ($null -ne $env:VLC_SUBSYNC_MIN_AGE_DAYS -and $env:VLC_SUBSYNC_MIN_AGE_DAYS -ne '') { $env:VLC_SUBSYNC_MIN_AGE_DAYS } else { '14' }
if ($env:VLC_SUBSYNC_UNINSTALL -eq '1') { $Uninstall = $true }
if (-not $SetupArgs) { $SetupArgs = @() }

function Say($msg) { Write-Host "==> $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "warning: $msg" -ForegroundColor Yellow }
function Fail($msg) { throw $msg }

function Find-Uv {
    $cmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidates = @(
        (Join-Path $env:USERPROFILE '.local\bin\uv.exe'),
        (Join-Path $env:USERPROFILE '.cargo\bin\uv.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\uv\uv.exe')
    )
    foreach ($c in $candidates) { if ($c -and (Test-Path $c)) { return $c } }
    return $null
}

function Get-ToolBinDir($uv) {
    $dir = (& $uv tool dir --bin --color never) | Select-Object -Last 1
    if ($LASTEXITCODE -ne 0 -or -not $dir) { return $null }
    return $dir.Trim()
}

function Invoke-Main {
    $uv = Find-Uv

    if ($Uninstall) {
        Say 'Uninstalling SubSync'
        if (-not $uv) { Fail 'uv not found; nothing to uninstall?' }
        $bin = Get-ToolBinDir $uv
        $exe = if ($bin) { Join-Path $bin 'vlc-subsync.exe' } else { $null }
        if ($exe -and (Test-Path $exe)) {
            & $exe uninstall @SetupArgs
            if ($LASTEXITCODE -ne 0) { Warn 'vlc-subsync uninstall reported problems' }
        } else {
            Warn 'vlc-subsync not found; skipping VLC cleanup'
        }
        & $uv tool uninstall vlc-subsync
        if ($LASTEXITCODE -ne 0) { Warn 'uv tool uninstall failed' }
        Say 'SubSync has been removed.'
        return
    }

    Say 'Installing SubSync (automatic subtitle sync for VLC)'

    if (-not $uv) {
        Say 'Installing uv (Python package manager from astral.sh)'
        try {
            Invoke-RestMethod -Uri 'https://astral.sh/uv/install.ps1' | Invoke-Expression
        } catch {
            Fail "uv installation failed: $_"
        }
        $uv = Find-Uv
        if (-not $uv) { Fail 'uv was installed but cannot be found; open a new window and re-run' }
    }
    Say "Using uv: $uv"

    $spec = "vlc-subsync @ $Source"
    if (Test-Path -LiteralPath $Source) {
        $abs = (Resolve-Path -LiteralPath $Source).Path
        $spec = 'vlc-subsync @ ' + ([System.Uri]$abs).AbsoluteUri
    }

    # Supply-chain protection: ignore dependency releases newer than $MinAgeDays days,
    # so a freshly published malicious version can't reach users before it's noticed.
    # uv records the cutoff in the tool receipt, so 'uv tool upgrade' keeps honouring it.
    if ($MinAgeDays -notmatch '^[0-9]+$') { Fail 'VLC_SUBSYNC_MIN_AGE_DAYS must be a whole number of days' }
    $installArgs = @('tool', 'install', '--force', '--python', $PyVer)
    if ([int]$MinAgeDays -gt 0) {
        $cutoff = (Get-Date).ToUniversalTime().AddDays(-[int]$MinAgeDays).ToString("yyyy-MM-dd'T'HH:mm:ss'Z'", [Globalization.CultureInfo]::InvariantCulture)
        Say "Using only dependency releases published before $cutoff ($MinAgeDays days ago)"
        $installArgs += @('--exclude-newer', $cutoff)
    }

    Say "Installing vlc-subsync with Python $PyVer (this downloads ~200 MB the first time)"
    & $uv @installArgs $spec
    if ($LASTEXITCODE -ne 0) { Fail 'package installation failed' }

    $bin = Get-ToolBinDir $uv
    if (-not $bin) { Fail 'cannot determine the uv tool bin directory' }
    $exe = Join-Path $bin 'vlc-subsync.exe'
    if (-not (Test-Path $exe)) { Fail "vlc-subsync.exe not found in $bin" }

    Say 'Configuring VLC'
    & $exe setup @SetupArgs
    if ($LASTEXITCODE -ne 0) { Fail "vlc-subsync setup failed (run `"$exe doctor`" for details)" }

    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (-not ($userPath -split ';' | Where-Object { $_.TrimEnd('\') -ieq $bin.TrimEnd('\') })) {
        Warn "$bin is not on your PATH; run 'uv tool update-shell' to use vlc-subsync from a terminal"
    }

    Write-Host ''
    Write-Host 'SubSync is installed. Restart VLC, open a video and choose a subtitle track.' -ForegroundColor Green
}

# Note: no `exit` inside Invoke-Main: under `irm | iex` it would close the user's window.
$rc = 0
try {
    Invoke-Main
} catch {
    Write-Host "error: $_" -ForegroundColor Red
    $rc = 1
}
if ($PSCommandPath) { exit $rc }
