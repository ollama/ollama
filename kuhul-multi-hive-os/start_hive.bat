@echo off
REM K'uhul Multi Hive OS - Windows Startup Script

echo ============================================
echo    K'UHUL MULTI HIVE OS - STARTUP
echo ============================================
echo.

echo [1/5] Checking Ollama installation...
where ollama >nul 2>nul
if %errorlevel% neq 0 (
    echo WARNING: Ollama not found. Please install it first:
    echo https://ollama.ai
    pause
    exit /b 1
)
echo OK: Ollama found
echo.

echo [2/5] Checking Ollama service...
curl -s http://localhost:11434/api/tags >nul 2>nul
if %errorlevel% neq 0 (
    echo WARNING: Ollama not running. Please start it manually:
    echo   ollama serve
    pause
)
echo OK: Ollama is running
echo.

echo [3/5] Checking required models...
echo    Checking qwen2.5:latest...
echo    Checking llama3.2:latest...
echo    Checking mistral:latest...
echo NOTE: Run 'ollama pull MODEL_NAME' if models are missing
echo.

echo [4/5] Installing Python dependencies...
cd backend
pip install -q -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo OK: Python dependencies installed
echo.

echo [5/5] Starting K'uhul Hive Server...
echo.
echo ============================================
echo    K'UHUL HIVE IS NOW ONLINE
echo ============================================
echo.
echo Backend API: http://localhost:8000
echo Frontend:    Open frontend/index.html in your browser
echo API Docs:    http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the hive
echo.

python kuhul_server.py
