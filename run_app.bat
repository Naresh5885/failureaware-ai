@echo off
title FailureAware AI — Server Startup Script
echo ===================================================================
echo 🚀 Launching FailureAware AI — Hybrid Multi-Agent Platform
echo ===================================================================
echo.
echo [1/2] Checking Python environment...
python --version
echo.
echo [2/2] Starting FastAPI Web Server at http://localhost:8000 ...
python app/api.py
pause
