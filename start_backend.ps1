$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Backend = Join-Path $Root "backend"
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Error "Virtual environment was not found at $Python. Run: python -m venv .venv"
}

Set-Location $Backend
& $Python -m uvicorn app:app --host 127.0.0.1 --port 8000
