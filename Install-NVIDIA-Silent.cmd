@echo off
chcp 65001 >nul
title 網咖專用 - NVIDIA 驅動靜默安裝

:: 1. 請求管理員權限 (安裝驅動必備)
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"
if '%errorlevel%' NEQ '0' (
    echo 正在請求管理員權限...
    echo Set UAC = CreateObject^("Shell.Application"^) > "%temp%\getadmin.vbs"
    echo UAC.ShellExecute "%~s0", "", "", "runas", 1 >> "%temp%\getadmin.vbs"
    "%temp%\getadmin.vbs"
    del "%temp%\getadmin.vbs"
    exit /B
)
pushd "%CD%"
CD /D "%~dp0"

echo ==========================================
echo   正在背景更新 NVIDIA 驅動 (繞過還原卡)
echo ==========================================
echo.
echo [提示] 系統將動態載入驅動，螢幕可能會閃爍或短暫黑屏，請勿關閉視窗。

:: 2. 自動尋找 env\drivers 目錄下的 NVIDIA 安裝檔
set "DRIVER_DIR=%~dp0env\drivers"
set "DRIVER_EXE="

for %%f in ("%DRIVER_DIR%\*.exe") do (
    set "DRIVER_EXE=%%f"
    goto :Install
)

:Install
if "%DRIVER_EXE%"=="" (
    echo [錯誤] 找不到驅動程式！請確保安裝檔已放置於 %DRIVER_DIR%
    echo.
    pause
    exit /b
)

echo 找到安裝檔: %DRIVER_EXE%
echo 正在執行靜默安裝 (預估需 3-5 分鐘)...

:: 3. 核心指令：-s (靜默模式) 與 -noreboot (禁止重開機)
start /wait "" "%DRIVER_EXE%" -s -noreboot

echo.
echo ✅ 安裝完成！新版 CUDA 驅動已在當前環境生效。
echo 🚀 您現在可以點擊 [Launch-Portable-AI.cmd] 啟動 Ollama 服務了！
echo ⚠️ (註：網咖電腦重開機後，此驅動將被還原卡清除，不留痕跡。)
echo.
pause
