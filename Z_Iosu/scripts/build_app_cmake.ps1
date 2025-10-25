#!powershell
# Compilar Ollama App usando MSVC 2022 (como GitHub Actions)
# Uso: powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\build_app_cmake.ps1

$ErrorActionPreference = "Stop"

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "  Compilando Ollama App con MSVC 2022 (estilo GitHub Actions)" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

# Configurar entorno
$SRC_DIR = $PWD
$VERSION = if ($env:VERSION) { $env:VERSION } else { "0.12.5" }
$ARCH = "amd64"
$APP_DIR = "${SRC_DIR}\app"
$DIST_DIR = "${SRC_DIR}\dist\windows-${ARCH}"

Write-Host "`n[1/4] Verificando herramientas..." -ForegroundColor Yellow

# Verificar rc.exe (de MSVC, NO windres)
$rcPath = (Get-Command rc.exe -ErrorAction SilentlyContinue).Source
if (-not $rcPath) {
    Write-Warning "rc.exe no encontrado en PATH, necesitamos cargar MSVC primero"
}

Write-Host "`n[2/4] Inicializando entorno Visual Studio 2022..." -ForegroundColor Yellow

# Cargar entorno MSVC 2022 (igual que GitHub Actions con Enterprise)
$MSVC_INSTALL = (Get-CimInstance MSFT_VSInstance -Namespace root/cimv2/vs | Where-Object { $_.InstallLocation -like "*2022*" })[0].InstallLocation
if (-not $MSVC_INSTALL) {
    Write-Error "Visual Studio 2022 no encontrado"
    exit 1
}

Write-Host "  → Cargando DevShell..." -ForegroundColor Gray
$vsDevShellModule = "${MSVC_INSTALL}\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
Import-Module $vsDevShellModule
Enter-VsDevShell -VsInstallPath $MSVC_INSTALL -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -no_logo'

Write-Host "  ✓ MSVC 2022: $MSVC_INSTALL" -ForegroundColor Green

Write-Host "`n[3/4] Generando recursos y compilando..." -ForegroundColor Yellow

# Crear directorio de salida
New-Item -ItemType Directory -Force -Path $DIST_DIR | Out-Null

# Ir a directorio app
Push-Location $APP_DIR

# Limpiar compilacion anterior
Remove-Item "ollama.syso" -ErrorAction SilentlyContinue
Remove-Item "ollama.res" -ErrorAction SilentlyContinue
Remove-Item "${DIST_DIR}\windows-${ARCH}-app.exe" -ErrorAction SilentlyContinue

# Generar ollama.syso con rc.exe de MSVC (NO windres)
Write-Host "  -> Generando recursos con rc.exe (MSVC)..." -ForegroundColor Gray
& rc.exe /fo ollama.res ollama.rc
if ($LASTEXITCODE -ne 0) {
    Write-Error "rc.exe fallo"
    exit 1
}

# Convertir .res a .syso para Go
Write-Host "  -> Convirtiendo .res a .syso..." -ForegroundColor Gray
Copy-Item "ollama.res" "ollama.syso" -Force

Write-Host "  OK ollama.syso generado con MSVC" -ForegroundColor Green

# CRÍTICO: Limpiar variables de entorno de llvm-mingw para usar MSVC puro
Write-Host "  → Limpiando entorno para usar MSVC..." -ForegroundColor Gray
$env:CGO_ENABLED = "1"
$env:CC = ""
$env:CXX = ""
Remove-Item env:\CGO_CFLAGS -ErrorAction SilentlyContinue
Remove-Item env:\CGO_CXXFLAGS -ErrorAction SilentlyContinue

# Compilar con Go usando MSVC (como GitHub Actions)
Write-Host "  → Compilando con Go + MSVC..." -ForegroundColor Gray
& go build `
    -trimpath `
    -ldflags "-s -w -H windowsgui -X=github.com/ollama/ollama/version.Version=$VERSION -X=github.com/ollama/ollama/server.mode=release" `
    -o "${DIST_DIR}\windows-${ARCH}-app.exe" `
    .

$buildResult = $LASTEXITCODE
Pop-Location

if ($buildResult -ne 0) {
    Write-Error "Compilación falló con código de salida: $buildResult"
    exit 1
}

Write-Host "`n[4/4] Verificando resultado..." -ForegroundColor Yellow

$appPath = "${DIST_DIR}\windows-${ARCH}-app.exe"
if (Test-Path $appPath) {
    $appSize = [math]::Round((Get-Item $appPath).Length / 1MB, 2)
    Write-Host "  OK App compilada exitosamente: $appSize MB" -ForegroundColor Green
    Write-Host "`n  Archivo: $appPath" -ForegroundColor Cyan
    
    # Verificar que NO se uso llvm-mingw
    Write-Host "`n  Verificando compilador usado..." -ForegroundColor Yellow
    $dllCheck = & dumpbin /dependents $appPath 2>$null | Select-String -Pattern "msvcrt|libgcc|libstdc"
    if ($dllCheck) {
        Write-Host "  Dependencias detectadas:" -ForegroundColor Gray
        $dllCheck | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }
    }
} else {
    Write-Error "App no se genero correctamente"
    exit 1
}

Write-Host "`n=====================================================================" -ForegroundColor Cyan
Write-Host "  OK COMPILACION EXITOSA CON MSVC 2022" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Cyan
