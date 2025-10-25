<#
.SYNOPSIS
  Verificar estado de Vulkan SIN modificar archivos originales

.DESCRIPTION
  Este script revisa el estado actual del soporte Vulkan en el repo
  sin tocar ningun archivo fuera de Z_Iosu/. Solo reporta que falta.

.PARAMETER ShowDetails
  Mostrar informacion detallada sobre cada verificacion

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\check-vulkan-status.ps1 -ShowDetails
#>
[CmdletBinding()]
param(
    [switch]$ShowDetails
)

$ErrorActionPreference = 'Continue'

Write-Host "VERIFICACION DE ESTADO VULKAN (SIN MODIFICAR ARCHIVOS)" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

# Funcion para mostrar resultado
function Show-Check($name, $result, $details = "") {
    $icon = if ($result) { "[OK]" } else { "[FALTA]" }
    Write-Host "$icon $name" -ForegroundColor $(if ($result) { "Green" } else { "Red" })
    if ($ShowDetails -and $details) {
        Write-Host "   -> $details" -ForegroundColor Gray
    }
}

# 1. Verificar directorio ggml-vulkan
$vulkanDir = "ml\backend\ggml\ggml\src\ggml-vulkan"
$hasVulkanDir = Test-Path $vulkanDir
Show-Check "Directorio ggml-vulkan/" $hasVulkanDir $(if ($hasVulkanDir) { "Encontrado en $vulkanDir" } else { "No encontrado - necesario del commit 2aba569" })

# 2. Verificar archivos Vulkan criticos
$vulkanFiles = @(
    "ml\backend\ggml\ggml\include\ggml-vulkan.h",
    "ml\backend\ggml\ggml\src\ggml-vulkan\ggml-vulkan.cpp"
)

$vulkanFilesExist = 0
foreach ($file in $vulkanFiles) {
    $exists = Test-Path $file
    Show-Check "Archivo $(Split-Path $file -Leaf)" $exists $(if ($exists) { $file } else { "Falta - necesario merge" })
    if ($exists) { $vulkanFilesExist++ }
}

# 3. Verificar Vulkan SDK
# Priorizar la variable de entorno VULKAN_SDK si está configurada
$vulkanSdkPaths = @(
    "$env:VULKAN_SDK",
    "C:\VulkanSDK"
)

$vulkanSdkFound = $false
$vulkanSdkPath = ""
foreach ($path in $vulkanSdkPaths) {
    if ($path -and (Test-Path $path)) {
        $vulkanSdkFound = $true
        $vulkanSdkPath = $path
        break
    }
}

Show-Check "Vulkan SDK instalado" $vulkanSdkFound $(if ($vulkanSdkFound) { "En $vulkanSdkPath" } else { "No encontrado - usar install-vulkan-sdk.ps1" })

# 4. Verificar variable VULKAN_SDK
$vulkanSdkVar = [bool]$env:VULKAN_SDK
Show-Check "Variable VULKAN_SDK" $vulkanSdkVar $(if ($vulkanSdkVar) { $env:VULKAN_SDK } else { "No configurada" })

# 5. Verificar herramientas Vulkan
$hasVulkanTools = $false
$toolDetails = "No disponibles"

# Candidatos de rutas a bin
$binCandidates = @()
if ($env:VULKAN_SDK) {
    $binCandidates += (Join-Path $env:VULKAN_SDK "bin")
}
if (Test-Path "C:\VulkanSDK") {
    $versionDirs = Get-ChildItem "C:\VulkanSDK" -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending
    foreach ($d in $versionDirs) {
        $binCandidates += (Join-Path $d.FullName "bin")
    }
}

# Comprobar vulkaninfo por rutas conocidas y por PATH
$vulkaninfoCmd = $null
foreach ($binPath in $binCandidates | Get-Unique) {
    if (Test-Path (Join-Path $binPath "vulkaninfo.exe")) {
        $vulkaninfoCmd = Join-Path $binPath "vulkaninfo.exe"
        break
    }
}
if (-not $vulkaninfoCmd) {
    $vulkaninfoCmd = Get-Command vulkaninfo.exe -ErrorAction SilentlyContinue
}

if ($vulkaninfoCmd) {
    $hasVulkanTools = $true
    $toolDetails = "vulkaninfo.exe disponible en $($vulkaninfoCmd | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue)"
}
Show-Check "Herramientas Vulkan" $hasVulkanTools $toolDetails

# 6. Verificar commit actual
try {
    $currentCommit = git rev-parse --short HEAD 2>$null
    $targetCommit = "2aba569"
    $hasTargetCommit = $false
    git cat-file -e "$targetCommit" 2>$null
    if ($LASTEXITCODE -eq 0) { $hasTargetCommit = $true }
    Show-Check "Commit Vulkan disponible" $hasTargetCommit $(if ($hasTargetCommit) { "Commit $targetCommit accesible" } else { "Commit $targetCommit no encontrado (puede no ser necesario si ya se hizo merge)" })
} catch {
    Show-Check "Git repository" $false "Error accediendo a git"
}

# 7. Verificar CMakeLists.txt para Vulkan
$cmakeFile = "CMakeLists.txt"
$hasCMakeVulkan = $false
if (Test-Path $cmakeFile) {
    $cmakeContent = Get-Content $cmakeFile -Raw
    $hasCMakeVulkan = $cmakeContent -match "find_package\(Vulkan\)" -or $cmakeContent -match "GGML_USE_VULKAN"
}
Show-Check "CMake configurado para Vulkan" $hasCMakeVulkan $(if ($hasCMakeVulkan) { "find_package(Vulkan) encontrado" } else { "Configuracion faltante" })

# RESUMEN
Write-Host ""
Write-Host "RESUMEN DE ESTADO" -ForegroundColor Yellow
Write-Host "=================" -ForegroundColor Yellow

$totalChecks = 7
$passedChecks = @($hasVulkanDir, ($vulkanFilesExist -eq $vulkanFiles.Count), $vulkanSdkFound, $vulkanSdkVar, $hasVulkanTools, $hasTargetCommit, $hasCMakeVulkan) | Where-Object { $_ } | Measure-Object | Select-Object -ExpandProperty Count

Write-Host "Estado: $passedChecks/$totalChecks verificaciones pasadas" -ForegroundColor $(if ($passedChecks -eq $totalChecks) { "Green" } else { "Yellow" })

if ($passedChecks -eq $totalChecks) {
    Write-Host "VULKAN LISTO PARA USAR" -ForegroundColor Green
    Write-Host "Puedes compilar con: powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\build-with-vulkan.ps1"
} elseif ($vulkanFilesExist -eq 0) {
    Write-Host "IMPLEMENTACION VULKAN FALTANTE" -ForegroundColor Red
    Write-Host "Necesitas: powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\integrate-vulkan.ps1"
} elseif (-not $vulkanSdkFound) {
    Write-Host "VULKAN SDK FALTANTE" -ForegroundColor Red
    Write-Host "Necesitas: powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\install-vulkan-sdk.ps1"
} else {
    Write-Host "CONFIGURACION INCOMPLETA" -ForegroundColor Yellow
    Write-Host "Revisar configuracion y variables de entorno"
}

Write-Host ""
Write-Host "Todos los archivos permanecen sin modificar" -ForegroundColor Green
Write-Host "Consultar Z_Iosu\VULKAN_IMPLEMENTATION_PLAN.md para proximos pasos" -ForegroundColor Cyan