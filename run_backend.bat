@echo off
title DocuMind Backend Server
cd /d "%~dp0\backend"

echo [1/3] Checking Python environment...
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    if exist "C:\Python312\python.exe" (
        "C:\Python312\python.exe" -m venv .venv
    ) else (
        python -m venv .venv
    )
    call .venv\Scripts\activate.bat
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

if not exist ".env" (
    echo Creating .env file from .env.example...
    copy .env.example .env
)

echo.
echo ========================================================
echo   DocuMind Backend starting on http://127.0.0.1:8000
echo   API Docs available at http://127.0.0.1:8000/docs
echo ========================================================
echo.

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
pause
