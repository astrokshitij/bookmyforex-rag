@echo off
setlocal enabledelayedexpansion
title BookMyForex Internal RAG Assistant

echo =======================================================
echo    BookMyForex Internal Support RAG Application
echo =======================================================
echo.

:: Detect Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
        set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    ) else (
        echo [ERROR] Python not found. Please ensure Python is installed and in your PATH.
        pause
        exit /b 1
    )
) else (
    set "PYTHON_EXE=python"
)

echo [1/3] Using Python: !PYTHON_EXE!

:: Check virtual environment
if not exist "venv" (
    echo [2/3] Creating virtual environment (venv)...
    "!PYTHON_EXE!" -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

call venv\Scripts\activate.bat

echo [2/3] Installing/verifying dependencies from requirements.txt...
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org -r requirements.txt --quiet

echo.
echo [3/3] Starting BookMyForex RAG server at http://127.0.0.1:8000 ...
echo.
echo Open your browser and navigate to: http://127.0.0.1:8000
echo.
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

pause
