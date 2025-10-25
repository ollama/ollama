<#
.SYNOPSIS
  Instalar Vulkan SDK para compilacion con soporte Vulkan

.DESCRIPTION
  Descarga e instala Vulkan SDK 1.4.321.1 (version usada por Ollama oficial)
  Configura variables de entorno necesarias
  Solo modifica archivos en Z_Iosu/ para configuracion

.PARAMETER SkipDownload
  Omitir descarga si ya existe el instalador

.PARAMETER InstallPath
  Ruta personalizada de instalacion (por defecto C:\VulkanSDK)

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\install-vulkan-sdk.ps1
#>
[CmdletBinding()]
param(
    [switch]$SkipDownload,
    [string]$InstallPath = "C:\VulkanSDK"
)

$ErrorActionPreference = 'Stop'

Write-Host "INSTALADOR VULKAN SDK PARA OLLAMA" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

# Configuracion
$VulkanVersion = "1.4.321.1"
$VulkanUrl = "https://sdk.lunarg.com/sdk/download/$VulkanVersion/windows/vulkansdk-windows-X64-$VulkanVersion.exe"
$InstallerPath = Join-Path $env:TEMP "vulkan-installer-$VulkanVersion.exe"
$ConfigScript = Join-Path (Split-Path $PSScriptRoot -Parent) "config\vulkan-env.ps1"

Write-Host "Version Vulkan SDK: $VulkanVersion" -ForegroundColor Yellow
Write-Host "Instalacion en: $InstallPath" -ForegroundColor Yellow
Write-Host "Configuracion en: $ConfigScript" -ForegroundColor Yellow

# Paso 1: Verificar si ya esta instalado
if (Test-Path $InstallPath) {
    $existingVersion = Get-ChildItem $InstallPath -Directory | Select-Object -First 1 -ExpandProperty Name
    if ($existingVersion) {
        Write-Host "[INFO] Vulkan SDK ya instalado: $existingVersion" -ForegroundColor Green
        $useExisting = Read-Host "¿Usar instalacion existente? (s/N)"
        if ($useExisting -eq "s" -or $useExisting -eq "S") {
            $InstallPath = Join-Path $InstallPath $existingVersion
            Write-Host "[OK] Usando instalacion existente: $InstallPath" -ForegroundColor Green
            $skipInstallation = $true
        }
    }
}

# Inicializar variable de control
$skipInstallation = $false

# Paso 2: Descargar instalador
if (-not $skipInstallation -and (-not $SkipDownload -or -not (Test-Path $InstallerPath))) {
    Write-Host "[INFO] Descargando Vulkan SDK..." -ForegroundColor Yellow
    try {
        Invoke-WebRequest -Uri $VulkanUrl -OutFile $InstallerPath -UseBasicParsing
        Write-Host "[OK] Descarga completada: $InstallerPath" -ForegroundColor Green
    } catch {
        Write-Host "[ERROR] Fallo al descargar: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[INFO] Usando instalador existente: $InstallerPath" -ForegroundColor Yellow
}

# Verificar archivo descargado
if (-not (Test-Path $InstallerPath)) {
    Write-Host "[ERROR] Instalador no encontrado: $InstallerPath" -ForegroundColor Red
    exit 1
}

$fileSize = (Get-Item $InstallerPath).Length / 1MB
Write-Host "[INFO] Tamaño del instalador: $([math]::Round($fileSize, 1)) MB" -ForegroundColor Gray

# Paso 3: Verificar permisos de administrador
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "[WARN] Se necesitan permisos de administrador para instalar Vulkan SDK" -ForegroundColor Yellow
    Write-Host "[INFO] Reiniciando script como administrador..." -ForegroundColor Cyan
    
    # Reiniciar como administrador
    $scriptPath = $MyInvocation.MyCommand.Path
    $argumentList = "-ExecutionPolicy Bypass -File `"$scriptPath`""
    if ($SkipDownload) { $argumentList += " -SkipDownload" }
    if ($InstallPath -ne "C:\VulkanSDK") { $argumentList += " -InstallPath `"$InstallPath`"" }
    
    Start-Process PowerShell -ArgumentList $argumentList -Verb RunAs -Wait
    
    # Verificar si la instalacion fue exitosa
    if (Test-Path $InstallPath) {
        Write-Host "[OK] Vulkan SDK instalado por el proceso elevado" -ForegroundColor Green
        $skipInstallation = $true
    } else {
        Write-Host "[ERROR] Instalacion como administrador fallo" -ForegroundColor Red
        exit 1
    }
}

# Paso 4: Instalar SDK (con permisos de administrador)
if (-not $skipInstallation) {
    Write-Host "[INFO] Iniciando instalacion Vulkan SDK como administrador..." -ForegroundColor Yellow
    Write-Host "       Esto puede tomar varios minutos..." -ForegroundColor Gray

    try {
    # Instalacion silenciosa
    $installArgs = @(
        "-c",      # Componentes por defecto
        "--am",    # Acepta EULA
        "--al",    # Acepta licencias
        "in"       # Instalacion
    )
    
    Write-Host "[INFO] Ejecutando: $InstallerPath $($installArgs -join ' ')" -ForegroundColor Gray
    
    $process = Start-Process -FilePath $InstallerPath -ArgumentList $installArgs -NoNewWindow -Wait -PassThru
    
    if ($process.ExitCode -eq 0) {
        Write-Host "[OK] Vulkan SDK instalado exitosamente" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Instalacion fallo con codigo: $($process.ExitCode)" -ForegroundColor Red
        Write-Host "[INFO] Intentando instalacion interactiva..." -ForegroundColor Yellow
        
        # Fallback: instalacion interactiva
        Write-Host "[INFO] Abriendo instalador interactivo..." -ForegroundColor Cyan
        Write-Host "       Por favor, instala manualmente y presiona Enter cuando termine" -ForegroundColor Yellow
        
        Start-Process -FilePath $InstallerPath -Wait
        Read-Host "Presiona Enter cuando la instalacion termine"
    }
} catch {
    Write-Host "[ERROR] Error durante instalacion: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "[INFO] Intentando instalacion manual..." -ForegroundColor Yellow
    
    # Fallback: instalacion manual
    Write-Host "[INFO] Abriendo instalador para instalacion manual..." -ForegroundColor Cyan
    Start-Process -FilePath $InstallerPath
    Read-Host "Por favor, instala Vulkan SDK manualmente y presiona Enter cuando termine"
    }
}

# Paso 5: Encontrar ruta de instalacion
if (-not (Test-Path $InstallPath)) {
    # Buscar instalacion automaticamente
    $possiblePaths = @(
        "C:\VulkanSDK\$VulkanVersion",
        "C:\VulkanSDK\*"
    )
    
    foreach ($path in $possiblePaths) {
        $resolved = Resolve-Path $path -ErrorAction SilentlyContinue
        if ($resolved) {
            $InstallPath = $resolved.Path | Select-Object -First 1
            break
        }
    }
}

if (-not (Test-Path $InstallPath)) {
    Write-Host "[ERROR] No se pudo encontrar la instalacion de Vulkan SDK" -ForegroundColor Red
    Write-Host "        Buscar manualmente en C:\VulkanSDK\" -ForegroundColor Yellow
    exit 1
}

Write-Host "[OK] Vulkan SDK encontrado: $InstallPath" -ForegroundColor Green

# Paso 6: Crear script de configuracion de entorno
$configDir = Split-Path $ConfigScript -Parent
if (-not (Test-Path $configDir)) {
    New-Item -ItemType Directory -Path $configDir -Force | Out-Null
}

$envConfig = @"
# Configuracion de entorno Vulkan SDK para Ollama
# Generado automaticamente el $(Get-Date)

# Ruta del SDK
`$env:VULKAN_SDK = "$InstallPath"

# Añadir al PATH
`$vulkanBin = Join-Path `$env:VULKAN_SDK "bin"
if (`$env:PATH -notlike "*`$vulkanBin*") {
    `$env:PATH = "`$vulkanBin;" + `$env:PATH
}

# Variables CMake
`$env:CMAKE_PREFIX_PATH = `$env:VULKAN_SDK

# Verificar instalacion
`$vulkaninfo = Join-Path `$vulkanBin "vulkaninfo.exe"
if (Test-Path `$vulkaninfo) {
    Write-Host "[OK] Vulkan SDK configurado: `$env:VULKAN_SDK" -ForegroundColor Green
} else {
    Write-Host "[WARN] vulkaninfo.exe no encontrado en `$vulkanBin" -ForegroundColor Yellow
}

# Exportar para uso en otros scripts
Export-ModuleMember -Variable * -Function *
"@

Set-Content -Path $ConfigScript -Value $envConfig -Encoding UTF8
Write-Host "[OK] Configuracion guardada: $ConfigScript" -ForegroundColor Green

# Paso 7: Aplicar configuracion en sesion actual
& $ConfigScript

# Paso 8: Verificacion final
Write-Host ""
Write-Host "VERIFICACION FINAL" -ForegroundColor Cyan
Write-Host "=================" -ForegroundColor Cyan

$vulkanBin = Join-Path $InstallPath "bin"
$vulkaninfo = Join-Path $vulkanBin "vulkaninfo.exe"

if (Test-Path $vulkaninfo) {
    Write-Host "[OK] vulkaninfo.exe encontrado" -ForegroundColor Green
    Write-Host "    Ruta: $vulkaninfo" -ForegroundColor Gray
    
    try {
        $version = & $vulkaninfo --version 2>$null | Select-Object -First 1
        if ($version) {
            Write-Host "[OK] Version: $version" -ForegroundColor Green
        }
    } catch {
        Write-Host "[WARN] No se pudo obtener version" -ForegroundColor Yellow
    }
} else {
    Write-Host "[ERROR] vulkaninfo.exe no encontrado" -ForegroundColor Red
}

# Verificar variables de entorno
Write-Host "[INFO] VULKAN_SDK = $env:VULKAN_SDK" -ForegroundColor Gray
Write-Host "[INFO] PATH contiene Vulkan: $($env:PATH -like "*vulkan*")" -ForegroundColor Gray

Write-Host ""
Write-Host "INSTALACION COMPLETADA" -ForegroundColor Green
Write-Host "======================" -ForegroundColor Green
Write-Host "Para usar en futuras sesiones, ejecuta:" -ForegroundColor Yellow
Write-Host "    . $ConfigScript" -ForegroundColor Cyan
Write-Host ""
Write-Host "Para compilar con Vulkan:" -ForegroundColor Yellow  
Write-Host "    powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\build-with-vulkan.ps1" -ForegroundColor Cyan

# Limpiar instalador temporal
if (Test-Path $InstallerPath) {
    Remove-Item $InstallerPath -Force -ErrorAction SilentlyContinue
    Write-Host "[INFO] Instalador temporal eliminado" -ForegroundColor Gray
}