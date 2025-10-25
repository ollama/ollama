# Integración Vulkan en build_windows.ps1

## Situación Actual
✅ **Vulkan completamente implementado** (merge del upstream completado)
✅ **Script build_windows.ps1 funcional** para CPU + CUDA
⏳ **Falta integrar Vulkan** en el proceso de compilación existente

## Modificación Necesaria en Z_Iosu/scripts/build_windows.ps1

### Añadir función buildVulkan

Agregar esta función al script existente:

```powershell
function buildVulkan {
    Write-Host "================== Vulkan Backend Build ==================" -ForegroundColor Cyan
    
    # Verificar Vulkan SDK
    if (-not $env:VULKAN_SDK) {
        Write-Host "VULKAN_SDK no encontrado. Instalando..." -ForegroundColor Yellow
        & (Join-Path $PSScriptRoot "install-vulkan-sdk.ps1")
        if (-not $env:VULKAN_SDK) {
            Write-Host "ERROR: No se pudo configurar Vulkan SDK" -ForegroundColor Red
            return $false
        }
    }
    
    Write-Host "Vulkan SDK: $env:VULKAN_SDK" -ForegroundColor Green
    
    # Configurar CMake para Vulkan
    $cmakeArgs = @(
        "-B", "build"
        "-DGGML_USE_VULKAN=ON"
        "-DCMAKE_BUILD_TYPE=Release"
        "-A", "x64"
    )
    
    Write-Host "Configurando CMake con Vulkan..." -ForegroundColor Yellow
    cmake @cmakeArgs
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR en configuración CMake Vulkan" -ForegroundColor Red
        return $false
    }
    
    # Compilar bibliotecas Vulkan
    Write-Host "Compilando backend Vulkan..." -ForegroundColor Yellow
    cmake --build build --config Release --target ggml-vulkan
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Backend Vulkan compilado exitosamente" -ForegroundColor Green
        return $true
    } else {
        Write-Host "❌ Error compilando backend Vulkan" -ForegroundColor Red
        return $false
    }
}
```

### Modificar función gatherDependencies

Actualizar para incluir DLLs de Vulkan:

```powershell
# En la función gatherDependencies, añadir después de CUDA:

# Vulkan DLLs (si están disponibles)
if (Test-Path "build\lib\ggml-vulkan.dll") {
    Copy-Item "build\lib\ggml-vulkan.dll" "$dllPath\" -Force
    Write-Host "✅ ggml-vulkan.dll copiado" -ForegroundColor Green
}
```

## Comando Actualizado

### Original (sin Vulkan):
```powershell
$env:VERSION = "0.12.59"
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\build_windows.ps1 buildCPU buildCUDA13 gatherDependencies buildOllama buildApp buildInstaller
```

### Nuevo (con Vulkan):
```powershell
$env:VERSION = "0.12.59"
powershell -ExecutionPolicy Bypass -File Z_Iosu\scripts\build_windows.ps1 buildCPU buildCUDA13 buildVulkan gatherDependencies buildOllama buildApp buildInstaller
```

## Beneficios de Añadir Vulkan

✅ **Soporte multi-GPU** (AMD + Intel + NVIDIA)
✅ **Mejor rendimiento** en GPUs no-CUDA
✅ **Compatibilidad ampliada** con hardware diverso
✅ **Fallback robusto** si CUDA falla

## Instalación Automática Vulkan SDK

El script detectará automáticamente si falta Vulkan SDK y ejecutará:
```powershell
Z_Iosu\scripts\install-vulkan-sdk.ps1
```

## Resultado Final

La compilación generará:
```
dist\windows-amd64\lib\ollama\
├── ggml-cuda.dll        # NVIDIA GPUs
├── ggml-vulkan.dll      # AMD/Intel/NVIDIA GPUs (universal)
├── ggml-cpu-*.dll       # CPU backends
└── [otros archivos]
```

Ollama automáticamente elegirá el mejor backend disponible en runtime.