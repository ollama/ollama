@echo off
setlocal

:: Define a function to print in red color
:print_red
echo %~1
exit /b

:: Check if 'go' is installed
where go >nul 2>nul
if errorlevel 1 (
    color 0C
    call :print_red "Error: 'go' is not installed."
    exit /b
)

:: Check if 'cmake' is installed
where cmake >nul 2>nul
if errorlevel 1 (
    color 0C
    call :print_red "Error: 'cmake' is not installed."
    exit /b
)

:: If both are installed, execute the following commands:
go generate ./... 
if errorlevel 1 (
    color 0C
    call :print_red "Error: Failed to build ollama."
    exit /b
)

go build -ldflags '-linkmode external -extldflags "-static"' .
if errorlevel 1 (
    color 0C
    call :print_red "Error: Failed to build ollama."
    exit /b
)

echo Successfully build ollama!
endlocal
exit /b
