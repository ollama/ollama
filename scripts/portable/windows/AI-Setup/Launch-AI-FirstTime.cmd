@echo off
setlocal
chcp 65001 >nul
cls

set "USB_DIR=%~dp0"
cd /d "%USB_DIR%"

echo ============================================================
echo      🚀 隨身碟 AI 工作流 - 跨電腦一鍵部署與自動化啟動 🚀
echo ============================================================

net session >nul 2>&1
if not "%errorlevel%"=="0" (
  echo [系統] 需要系統管理員權限，正在重新啟動...
  powershell -NoProfile -Command "Start-Process -FilePath '%ComSpec%' -ArgumentList '/c ""%~f0""' -Verb RunAs"
  exit /b 0
)

set "FLAG_FILE=%USB_DIR%.initialized"
if exist "%FLAG_FILE%" (
  echo [狀態] 已完成初始化，直接啟動 AI。
  call "%USB_DIR%Launch-AI.cmd"
  exit /b %errorlevel%
)

echo [狀態] 首次初始化中...
echo [安全提醒] 目前以 ExecutionPolicy RemoteSigned 執行本地腳本，請只在可信任隨身碟上使用。
powershell -NoProfile -ExecutionPolicy RemoteSigned -File "%USB_DIR%Install-AI-Tools.ps1"
if not "%errorlevel%"=="0" (
  echo [錯誤] 初始化失敗，請檢查訊息後重試。
  pause
  exit /b 1
)

type nul > "%FLAG_FILE%"
echo [完成] 初始化完成，已寫入 %FLAG_FILE%
call "%USB_DIR%Launch-AI.cmd"
exit /b %errorlevel%
