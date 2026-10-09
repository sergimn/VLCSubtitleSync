# Mimic the report: 32-bit Windows PowerShell, elevated, cwd System32, irm | iex of the latest release.
$ErrorActionPreference = 'Continue'
"PS arch: $env:PROCESSOR_ARCHITECTURE / W6432: $env:PROCESSOR_ARCHITEW6432  ps: $([Environment]::Is64BitProcess)"
$log = Join-Path $env:RUNNER_TEMP 'x86-install.log'
$ps86 = "$env:WINDIR\SysWOW64\WindowsPowerShell\v1.0\powershell.exe"
$p = Start-Process $ps86 -WorkingDirectory "$env:WINDIR\System32" -ArgumentList '-NoProfile','-Command',"powershell -c `"irm https://github.com/sergimn/VLCSubtitleSync/releases/latest/download/install.ps1 | iex`" *>&1 | Tee-Object -FilePath '$log'" -PassThru -NoNewWindow
$t0 = Get-Date
while (-not $p.HasExited -and ((Get-Date) - $t0).TotalSeconds -lt 600) {
  Start-Sleep 15
  $s = [int]((Get-Date) - $t0).TotalSeconds
  Write-Host "---- t=${s}s processes:"
  Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|uv|vlc-subsync|vc_redist|powershell' } |
    ForEach-Object { Write-Host ("  {0} {1} cpu={2}" -f $_.ProcessId, $_.CommandLine, $_.KernelModeTime) }
}
"exited: $($p.HasExited) code: $(if ($p.HasExited) { $p.ExitCode })"
if (-not $p.HasExited) {
  Write-Host '---- HUNG; python stacks via py-spy would go here; dumping child tree'
  Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' } | ForEach-Object {
    $id = $_.ProcessId
    Write-Host "python $id threads: $((Get-Process -Id $id).Threads.Count)"
  }
}
