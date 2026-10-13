@echo off
chcp 65001 >nul
title Portable AI - 啟動 ComfyUI (繪圖環境)

:: 自動鎖定當前隨身碟路徑
set "BASE_DIR=%~dp0"
echo ==========================================
echo       正在啟動 Portable ComfyUI
echo ==========================================
echo [系統] 目前隨身碟路徑: %BASE_DIR%

:: 鎖定 ComfyUI 官方免安裝版資料夾路徑
set "COMFY_DIR=%BASE_DIR%ComfyUI_windows_portable"

if not exist "%COMFY_DIR%\run_nvidia_gpu.bat" (
    echo.
    echo [錯誤] 找不到 ComfyUI 執行檔！
    echo 系統預期路徑: %COMFY_DIR%
    echo.
    echo 請確認您日後有用電腦將 ComfyUI 官方免安裝版解壓縮，
    echo 並且整個資料夾名稱維持為 ComfyUI_windows_portable，放在與此腳本同一層目錄。
    echo.
    pause
    exit /b
)

echo.
echo 找到 ComfyUI，正在啟動影像生成伺服器...
cd /d "%COMFY_DIR%"

:: 呼叫官方的 NVIDIA 啟動腳本
call run_nvidia_gpu.bat

pause
