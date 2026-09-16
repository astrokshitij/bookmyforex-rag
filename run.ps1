# BookMyForex Internal Support RAG Application Launcher
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "   BookMyForex Internal Support RAG Application        " -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

# Locate Python
$pythonExe = "python"
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    $fallback = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
    if (Test-Path $fallback) {
        $pythonExe = $fallback
    } else {
        Write-Error "Python was not found. Please ensure Python is installed and in your PATH."
        exit 1
    }
}

Write-Host "[1/3] Using Python: $pythonExe" -ForegroundColor Green

# Setup virtual environment
if (-not (Test-Path "venv")) {
    Write-Host "[2/3] Creating virtual environment (venv)..." -ForegroundColor Yellow
    & $pythonExe -m venv venv
}

# Activate virtual environment
$venvPython = ".\venv\Scripts\python.exe"
$venvPip = ".\venv\Scripts\pip.exe"

Write-Host "[2/3] Installing dependencies..." -ForegroundColor Yellow
& $venvPip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org -r requirements.txt --quiet

Write-Host "`n[3/3] Starting BookMyForex RAG server at http://127.0.0.1:8000 ..." -ForegroundColor Green
Write-Host "Opening web browser..." -ForegroundColor Cyan
Start-Process "http://127.0.0.1:8000"

& $venvPython -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
