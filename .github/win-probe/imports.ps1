$py = Join-Path $env:APPDATA 'uv\tools\vlc-subsync\Scripts\python.exe'
foreach ($m in 'numpy','av','ctranslate2','onnxruntime','tokenizers','faster_whisper') {
  $r = & $py -c "import $m; print('ok')" 2>&1 | Select-Object -Last 1
  Write-Host "== import ${m}: $r"
}
