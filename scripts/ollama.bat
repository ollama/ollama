@echo off
setlocal DisableDelayedExpansion
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0ollama.ps1" %*
exit /b %ERRORLEVEL%
