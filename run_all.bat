@echo off
title DocuMind AI Launcher
echo ========================================================
echo   Launching DocuMind AI (Backend + Frontend)
echo ========================================================
echo.

start "DocuMind Backend" "%~dp0run_backend.bat"
timeout /t 2 /nobreak >nul
start "DocuMind Frontend" "%~dp0run_frontend.bat"

echo Both servers are starting up!
echo - Backend:  http://127.0.0.1:8000
echo - Frontend: http://localhost:5173
echo.
