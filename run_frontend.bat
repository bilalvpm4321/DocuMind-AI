@echo off
title DocuMind Frontend Server
cd /d "%~dp0\frontend"

echo [1/2] Checking node_modules...
if not exist "node_modules\" (
    echo Installing frontend dependencies...
    call npm install
)

echo.
echo ========================================================
echo   DocuMind Frontend starting on http://localhost:5173
echo ========================================================
echo.

npm run dev
pause
