@echo off
set "OLLAMA_ROOT=%~dp0"
set "ENV_ROOT=%OLLAMA_ROOT%env"
set "OLLAMA_MODELS=%OLLAMA_ROOT%models"
set "UV_CACHE_DIR=%ENV_ROOT%\cache\uv"
set "PIP_CACHE_DIR=%ENV_ROOT%\cache\pip"
set "HF_HOME=%ENV_ROOT%\cache\huggingface"
set "TORCH_HOME=%ENV_ROOT%\cache\torch"
set "TEMP=%ENV_ROOT%\tmp"
set "TMP=%ENV_ROOT%\tmp"
set "PATH=%OLLAMA_ROOT%;%ENV_ROOT%\git\cmd;%ENV_ROOT%\uv;%PATH%"
for /d %%D in ("%ENV_ROOT%\cmake\cmake-*") do set "PATH=%%D\bin;%PATH%"
for /d %%D in ("%ENV_ROOT%\python\cpython-*") do set "PATH=%%D;%PATH%"
cd /d "%OLLAMA_ROOT%"
echo Portable AI shell ready.
echo Models: %OLLAMA_MODELS%
cmd.exe /k
