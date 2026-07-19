@echo off
setlocal
chcp 65001 >nul
set "USB_DIR=%~dp0"
cd /d "%USB_DIR%"

echo [啟動] 啟動 AI 工作流...
powershell -NoProfile -ExecutionPolicy Bypass -File "%USB_DIR%Start-All-AI.ps1"
if not "%errorlevel%"=="0" (
  echo [錯誤] 啟動失敗。
  pause
  exit /b 1
)
exit /b 0
