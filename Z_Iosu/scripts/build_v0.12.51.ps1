# ============================================================================
# BUILD OLLAMA v0.12.51 - SCRIPT COMPLETAMENTE AUTOMATIZADO
# ============================================================================
# Versión: 0.12.51
# Compilador: llvm-mingw-20240619-ucrt-x86_64 (clang 18.1.8)
# TODO EN UN SOLO SCRIPT - SIN DEPENDENCIAS EXTERNAS
# ============================================================================

$ErrorActionPreference = "Stop"
$VERSION = "0.12.51"

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host " OLLAMA v$VERSION - COMPILACION COMPLETAMENTE AUTOMATIZADA" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# ----------------------------------------------------------------------------
# PASO 1: Configurar entorno
# ----------------------------------------------------------------------------
Write-Host "[1/7] Configurando entorno de compilacion..." -ForegroundColor Yellow

# Configurar llvm-mingw
$llvmPath = "C:\llvm-mingw\llvm-mingw-20240619-ucrt-x86_64\bin"
if (-not (Test-Path $llvmPath)) {
    Write-Host "ERROR: llvm-mingw no encontrado en $llvmPath" -ForegroundColor Red
    Write-Host "Instala llvm-mingw primero" -ForegroundColor Red
    exit 1
}
$env:Path = "$llvmPath;" + $env:Path
$env:CC = "$llvmPath\gcc.exe"
$env:CXX = "$llvmPath\g++.exe"
$env:CGO_ENABLED = "1"
$env:VERSION = $VERSION

# Verificar compilador
$gccVersion = & gcc --version 2>&1 | Select-String "clang version"
if ($gccVersion -match "18.1.8") {
    Write-Host "  Compilador: $gccVersion" -ForegroundColor Green
} else {
    Write-Host "ERROR: Versión de clang incorrecta" -ForegroundColor Red
    exit 1
}

# Configurar ccache
& ccache -o cache_dir=".ccache" 2>&1 | Out-Null
Write-Host "  ccache configurado" -ForegroundColor Green

# ----------------------------------------------------------------------------
# PASO 2: Verificar cache de dependencias
# ----------------------------------------------------------------------------
Write-Host "[2/7] Verificando dependencias en cache..." -ForegroundColor Yellow
$needsBuildDeps = $false

if (-not (Test-Path "dist\lib\ollama\ggml-base.dll")) {
    Write-Host "  Dependencias no encontradas, se compilaran" -ForegroundColor Yellow
    $needsBuildDeps = $true
} else {
    $dllCount = (Get-ChildItem "dist\lib\ollama\*.dll" -ErrorAction SilentlyContinue).Count
    Write-Host "  DLLs en cache: $dllCount archivos" -ForegroundColor Green
}

# ----------------------------------------------------------------------------
# PASO 3: Compilar dependencias si es necesario
# ----------------------------------------------------------------------------
if ($needsBuildDeps) {
    Write-Host "[3/7] Compilando dependencias nativas (CPU + CUDA)..." -ForegroundColor Yellow
    Write-Host "  Llamando a build_windows.ps1..." -ForegroundColor Yellow
    & powershell -ExecutionPolicy Bypass -File "Z_Iosu\scripts\build_windows.ps1" buildCPU buildCUDA13
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Fallo en compilacion de dependencias" -ForegroundColor Red
        exit 1
    }
    Write-Host "  Dependencias compiladas correctamente" -ForegroundColor Green
} else {
    Write-Host "[3/7] OMITIDO: Dependencias ya compiladas" -ForegroundColor Green
}

# ----------------------------------------------------------------------------
# PASO 4: Compilar ollama.exe
# ----------------------------------------------------------------------------
Write-Host "[4/7] Compilando ollama.exe..." -ForegroundColor Yellow

if (-not (Test-Path "dist\windows-amd64")) {
    New-Item -ItemType Directory -Path "dist\windows-amd64" -Force | Out-Null
}

$ollamaExePath = "dist\windows-amd64\ollama.exe"
if (Test-Path $ollamaExePath) {
    $existingSize = [math]::Round((Get-Item $ollamaExePath).Length/1MB, 2)
    Write-Host "  ollama.exe ya existe ($existingSize MB), recompilando..." -ForegroundColor Yellow
}

& go build -a -trimpath -ldflags "-s -w -X=github.com/ollama/ollama/version.Version=$VERSION -X=github.com/ollama/ollama/server.mode=release" .

if (Test-Path "ollama.exe") {
    $size = [math]::Round((Get-Item "ollama.exe").Length/1MB, 2)
    Copy-Item "ollama.exe" $ollamaExePath -Force
    Remove-Item "ollama.exe" -Force
    Write-Host "  ollama.exe compilado: $size MB" -ForegroundColor Green
} else {
    Write-Host "ERROR: ollama.exe no se compilo" -ForegroundColor Red
    exit 1
}

# ----------------------------------------------------------------------------
# PASO 5: Compilar app (GUI)
# ----------------------------------------------------------------------------
Write-Host "[5/7] Compilando ollama app (GUI)..." -ForegroundColor Yellow

Set-Location "app"
& go build -trimpath -ldflags "-s -w -H windowsgui -X=github.com/ollama/ollama/version.Version=$VERSION -X=github.com/ollama/ollama/server.mode=release" -o "..\dist\ollama app.exe" .
Set-Location ".."

if (Test-Path "dist\ollama app.exe") {
    $appSize = [math]::Round((Get-Item "dist\ollama app.exe").Length/1MB, 2)
    Write-Host "  app compilada: $appSize MB" -ForegroundColor Green
} else {
    Write-Host "ERROR: App no se compilo" -ForegroundColor Red
    exit 1
}

# ----------------------------------------------------------------------------
# PASO 6: Preparar archivos para instalador
# ----------------------------------------------------------------------------
Write-Host "[6/7] Preparando archivos para instalador..." -ForegroundColor Yellow

if (-not (Test-Path "dist\ollama_welcome.ps1")) {
    Copy-Item "app\ollama_welcome.ps1" "dist\ollama_welcome.ps1" -Force
    Write-Host "  ollama_welcome.ps1 copiado" -ForegroundColor Green
}

# ----------------------------------------------------------------------------
# PASO 7: Crear instalador
# ----------------------------------------------------------------------------
Write-Host "[7/7] Creando instalador con Inno Setup..." -ForegroundColor Yellow
& powershell -ExecutionPolicy Bypass -File "Z_Iosu\scripts\build_windows.ps1" buildInstaller
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Fallo al crear instalador" -ForegroundColor Red
    exit 1
}
Write-Host "  Instalador creado correctamente" -ForegroundColor Green

# ----------------------------------------------------------------------------
# VERIFICACIÓN FINAL
# ----------------------------------------------------------------------------
Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host " VERIFICACION DE ARCHIVOS GENERADOS" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

$files = Get-ChildItem "dist" -Recurse -Include "*.exe", "*.dll" | Select-Object FullName, @{N = "Size (MB)"; E = { [math]::Round($_.Length / 1MB, 2) } }
$files | Format-Table -AutoSize

# Verificar versión
Write-Host ""
Write-Host "Verificando version de ollama.exe..." -ForegroundColor Yellow
$ollamaVersion = & ".\dist\windows-amd64\ollama.exe" --version 2>&1
Write-Host $ollamaVersion -ForegroundColor Green

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Green
Write-Host " COMPILACION COMPLETADA EXITOSAMENTE" -ForegroundColor Green
Write-Host "============================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Instalador: dist\OllamaSetup.exe" -ForegroundColor Cyan
Write-Host "CLI: dist\windows-amd64\ollama.exe" -ForegroundColor Cyan
Write-Host "App: dist\ollama app.exe" -ForegroundColor Cyan
Write-Host ""
