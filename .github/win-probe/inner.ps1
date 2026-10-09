"inner: 64-bit process: $([Environment]::Is64BitProcess)  W6432=$env:PROCESSOR_ARCHITEW6432"
powershell -c "irm https://github.com/sergimn/VLCSubtitleSync/releases/latest/download/install.ps1 | iex"
"inner: child exited $LASTEXITCODE"
