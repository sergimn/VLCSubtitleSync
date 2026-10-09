# SubSync (vlc-subsync) installer for Windows.
#
#   powershell -ExecutionPolicy Bypass -File install.ps1 [-Uninstall] [setup options]
#   irm https://github.com/sergimn/VLCSubtitleSync/releases/latest/download/install.ps1 | iex
#
# Environment:
#   VLC_SUBSYNC_SOURCE     package source (default: GitHub main branch archive);
#                          may be a local checkout directory for testing
#   VLC_SUBSYNC_PYTHON     Python version or uv Python request (default 3.12; x64 Python
#                          on Windows on ARM, where the speech engine has no ARM64 build)
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
$VcRedistUrl = 'https://aka.ms/vs/17/release/vc_redist.x64.exe'
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

# The Python of the SubSync tool environment. SubSync's own commands run through it, not
# through vlc-subsync.exe: uv makes that launcher on install, so Smart App Control and
# App Control policies know nothing about it and block it, while python.exe is the same
# file on every PC.
function Get-ToolPython($uv) {
    $dir = (& $uv tool dir --color never) | Select-Object -Last 1
    if ($LASTEXITCODE -ne 0 -or -not $dir) { return $null }
    $py = Join-Path $dir.Trim() 'vlc-subsync\Scripts\python.exe'
    if (Test-Path $py) { return $py }
    return $null
}

function Test-Arm64 {
    if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64' -or $env:PROCESSOR_ARCHITEW6432 -eq 'ARM64') { return $true }
    try {
        return [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString() -eq 'Arm64'
    } catch { return $false }
}

# The speech engine (ctranslate2, onnxruntime) needs the Microsoft Visual C++ runtime, which
# a fresh Windows does not always have. Returns $null when everything imports, else the error.
# The first start of the new Python and its DLLs can be slow (antivirus scanning them), so
# say that it is still working, and give up with a clear error rather than hang silently.
function Test-Engine($py, [int]$TimeoutSec = 300) {
    $tmp = Join-Path $env:TEMP "vlc-subsync-check-$PID"
    $code = '-c "import sys; [print(m, flush=True) or __import__(m) for m in sys.argv[1:]]"'
    $p = Start-Process -FilePath $py -ArgumentList "$code av ctranslate2 onnxruntime faster_whisper" `
        -NoNewWindow -PassThru -RedirectStandardOutput "$tmp.out" -RedirectStandardError "$tmp.err"
    $null = $p.Handle  # without this, ExitCode can stay empty after the process exits
    $t0 = Get-Date
    $told = $false
    while (-not $p.WaitForExit(2000)) {
        $secs = ((Get-Date) - $t0).TotalSeconds
        if (-not $told -and $secs -ge 20) {
            Say 'Still checking; the first start can take a few minutes while antivirus scans the new files'
            $told = $true
        }
        if ($secs -ge $TimeoutSec) {
            $module = Get-Content "$tmp.out" -ErrorAction SilentlyContinue | Select-Object -Last 1
            Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
            Remove-Item "$tmp.out", "$tmp.err" -ErrorAction SilentlyContinue
            Fail ("loading the speech engine ($module) did not finish in $([int]($TimeoutSec / 60)) minutes. " +
                'Something on this PC, often antivirus, is holding it; allow it or wait for its scan, then run this installer again')
        }
    }
    $last = Get-Content "$tmp.err" -ErrorAction SilentlyContinue | Where-Object { $_.Trim() } | Select-Object -Last 1
    Remove-Item "$tmp.out", "$tmp.err" -ErrorAction SilentlyContinue
    if ($p.ExitCode -eq 0) { return $null }
    if (-not $last) { return "python exited with code $($p.ExitCode)" }
    return $last.Trim()
}

function Install-VcRuntime {
    $file = Join-Path $env:TEMP 'vc_redist.x64.exe'
    Invoke-WebRequest -UseBasicParsing -Uri $VcRedistUrl -OutFile $file
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $admin = ([Security.Principal.WindowsPrincipal]$id).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    $sp = @{ FilePath = $file; ArgumentList = '/install /quiet /norestart'; Wait = $true; PassThru = $true }
    if (-not $admin) { $sp.Verb = 'RunAs' }  # UAC prompt; throws if the user declines
    $p = Start-Process @sp
    Remove-Item $file -ErrorAction SilentlyContinue
    return $p.ExitCode
}

function Invoke-Main {
    $uv = Find-Uv

    if ($Uninstall) {
        Say 'Uninstalling SubSync'
        if (-not $uv) { Fail 'uv not found; nothing to uninstall?' }
        $py = Get-ToolPython $uv
        if ($py) {
            & $py -m vlcsubsync.cli uninstall @SetupArgs
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

    if (-not $env:VLC_SUBSYNC_PYTHON -and (Test-Arm64)) {
        # ctranslate2 publishes no Windows ARM64 wheels; x64 Python runs under emulation.
        $PyVer = "cpython-$PyVer-windows-x86_64-none"
        Say 'Windows on ARM: using x64 Python (the speech engine has no ARM64 build)'
    }

    $spec = "vlc-subsync @ $Source"
    if (Test-Path -LiteralPath $Source) {
        $abs = (Resolve-Path -LiteralPath $Source).Path
        $spec = 'vlc-subsync @ ' + ([System.Uri]$abs).AbsoluteUri
    }

    Say "Installing vlc-subsync with Python $PyVer (this downloads ~200 MB the first time)"
    & $uv tool install --force --python $PyVer $spec
    if ($LASTEXITCODE -ne 0) { Fail 'package installation failed' }

    $bin = Get-ToolBinDir $uv
    if (-not $bin) { Fail 'cannot determine the uv tool bin directory' }
    $exe = Join-Path $bin 'vlc-subsync.exe'
    if (-not (Test-Path $exe)) { Fail "vlc-subsync.exe not found in $bin" }

    $py = Get-ToolPython $uv
    if (-not $py) { Fail 'cannot find the Python that SubSync was installed with' }
    Say 'Checking that the speech engine loads'
    $err = Test-Engine $py
    if ($err) {
        Say "The speech engine cannot load yet ($err)"
        Say 'Installing the Microsoft Visual C++ runtime it needs (Windows may ask for permission)'
        try { $code = Install-VcRuntime } catch {
            Fail "could not install the Microsoft Visual C++ runtime ($_). Install it from $VcRedistUrl, then run this installer again"
        }
        if ($code -notin 0, 1638, 3010) { Warn "the Visual C++ runtime installer exited with code $code" }
        $err = Test-Engine $py
        if ($err) { Fail "the speech engine still cannot load: $err" }
    }

    Say 'Configuring VLC'
    & $py -m vlcsubsync.cli setup @SetupArgs
    if ($LASTEXITCODE -ne 0) { Fail "vlc-subsync setup failed (run `"$py`" -m vlcsubsync.cli doctor for details)" }

    $blocked = $false
    try { & $exe --version | Out-Null } catch { $blocked = $true }
    if ($blocked) {
        Warn ('Windows (Smart App Control or an App Control policy) blocks the vlc-subsync command. ' +
            'SubSync still works in VLC. To run its commands, use: & "' + $py + '" -m vlcsubsync.cli doctor')
    }

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
