@echo off
setlocal enabledelayedexpansion

title VentureIQ - AI-Powered Startup Intelligence Platform

echo ================================================================================
echo   VENTUREIQ - AI-POWERED STARTUP EVALUATION & RISK INTELLIGENCE PLATFORM
echo   Target Architecture: Next.js + React + FastAPI + Scikit-Learn + RAG Engine
echo ================================================================================
echo.

:: 1. Check Python
echo [*] Checking Python environment...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [!] ERROR: Python is not detected in your PATH.
    echo     Please install Python 3.11 or 3.12 from https://www.python.org/
    echo     and ensure 'Add Python to PATH' is checked during setup.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do set PYTHON_VER=%%v
echo     Detected: %PYTHON_VER%

:: 2. Check Node.js and NPM
echo [*] Checking Node.js environment...
node --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [!] ERROR: Node.js is not detected in your PATH.
    echo     Please install Node.js (LTS version) from https://nodejs.org/
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('node --version 2^>^&1') do set NODE_VER=%%v
echo     Detected Node.js: %NODE_VER%

:: 3. Setup Python Virtual Environment
set VENV_DIR=.venv
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [*] Creating Python virtual environment in %VENV_DIR%...
    python -m venv %VENV_DIR%
    if %ERRORLEVEL% NEQ 0 (
        echo [!] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

echo [*] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

:: 4. Verify & Install Python Dependencies
echo [*] Verifying Python dependencies...
python -c "import fastapi, uvicorn, pydantic, sklearn, joblib, pandas, numpy, reportlab" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [*] Installing required Python backend dependencies...
    python -m pip install --upgrade pip
    pip install fastapi uvicorn pydantic python-multipart httpx scikit-learn pandas numpy joblib reportlab requests
)

:: 5. Ensure Machine Learning Model & RAG Index are built
if not exist "models\model.joblib" (
    echo [*] Training initial Gradient Boosting model...
    python train_model.py
)

if not exist "models\rag_index\manifest.json" (
    echo [*] Building RAG knowledge base index...
    python scripts\build_index.py
)

:: 6. Setup Frontend
echo [*] Checking frontend dependencies...
if not exist "frontend\node_modules" (
    echo [*] Installing Next.js frontend dependencies (first launch only)...
    cd frontend
    call npm install
    cd ..
)

:: 7. Launch Services
echo.
echo ================================================================================
echo   STARTING VENTUREIQ APPLICATION SERVICES
echo   - Backend REST API: http://127.0.0.1:8000
echo   - Interactive UI:   http://localhost:3000
echo ================================================================================
echo.

echo [*] Starting FastAPI Backend on port 8000...
start "VentureIQ Backend (FastAPI)" cmd /k "call %VENV_DIR%\Scripts\activate.bat && uvicorn api:app --host 127.0.0.1 --port 8000 --reload"

echo [*] Starting Next.js Frontend on port 3000...
start "VentureIQ Frontend (Next.js)" cmd /k "cd frontend && npm run dev"

:: 8. Wait and Launch Browser
echo [*] Waiting 4 seconds for services to initialize...
timeout /t 4 /nobreak >nul

echo [*] Opening VentureIQ in your default browser...
start http://localhost:3000

echo.
echo [OK] VentureIQ is now running!
echo      - To view API Documentation: http://127.0.0.1:8000/docs
echo      - To view Dashboard UI:     http://localhost:3000
echo.
echo Keep the backend and frontend terminal windows open while using VentureIQ.
echo Press any key to exit this launcher window.
pause >nul
