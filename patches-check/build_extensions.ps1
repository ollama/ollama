# ============================================================================
# Build script for OLLaMA ROCm 7.x + Vulkan aggressive extensions (Windows)
# Target: gfx1201 (RX 9070 XT)
# ============================================================================
$ErrorActionPreference = "Stop"

$OLLAMA_ROOT = if ($env:OLLAMA_ROOT) { $env:OLLAMA_ROOT } else { Get-Location }
$BUILD_DIR = Join-Path $OLLAMA_ROOT "build_ext"
$HIP_PATH = if ($env:HIP_PATH) { $env:HIP_PATH } else { "C:\Program Files\AMD\ROCm\7.1" }
$VULKAN_SDK = if ($env:VULKAN_SDK) { $env:VULKAN_SDK } else { "C:\VulkanSDK\1.4.335.0" }
$CLANG = Join-Path $HIP_PATH "bin\clang++.exe"

New-Item -ItemType Directory -Force -Path $BUILD_DIR | Out-Null

Write-Host "=== Building HIP Extensions (libggml_hip_ext.dll) ==="
& $CLANG -O3 -ffast-math -shared `
    -I"$HIP_PATH\include" -I"$HIP_PATH\include\rocblas" `
    -D__HIP_PLATFORM_AMD__ -DGGML_HIP_WAVE32=1 `
    -mllvm -amdgpu-inline-threshold=10000 `
    -o "$BUILD_DIR\libggml_hip_ext.dll" `
    ggml_hip_ext.cpp `
    -L"$HIP_PATH\lib" -lamdhip64 -lrocblas

Write-Host "=== Building Vulkan Extensions (libggml_vulkan_ext.dll) ==="
g++ -O3 -shared `
    -I"$VULKAN_SDK\Include" `
    -o "$BUILD_DIR\libggml_vulkan_ext.dll" `
    ggml_vulkan_ext.cpp `
    -L"$VULKAN_SDK\Lib" -lvulkan-1

Write-Host "=== Building TurboQuant (libggml_turboquant.dll) ==="
& $CLANG -O3 -ffast-math -shared `
    -I"$HIP_PATH\include" `
    -D__HIP_PLATFORM_AMD__ `
    -o "$BUILD_DIR\libggml_turboquant.dll" `
    ggml_turboquant.cpp `
    -L"$HIP_PATH\lib" -lamdhip64

Write-Host "=== Installing ==="
$OLLAMA_LIB_DIR = Join-Path $OLLAMA_ROOT "dist\windows-amd64\lib"
if (Test-Path $OLLAMA_LIB_DIR) {
    Copy-Item "$BUILD_DIR\libggml_hip_ext.dll" $OLLAMA_LIB_DIR
    Copy-Item "$BUILD_DIR\libggml_vulkan_ext.dll" $OLLAMA_LIB_DIR
    Copy-Item "$BUILD_DIR\libggml_turboquant.dll" $OLLAMA_LIB_DIR
    Write-Host "Installed to $OLLAMA_LIB_DIR"
} else {
    Write-Host "OLLaMA lib dir not found. Copy manually:"
    Write-Host "  copy $BUILD_DIR\*.dll <ollama_install>\lib\"
}

Write-Host "=== Done ==="
Write-Host "Set PATH=$BUILD_DIR;%PATH% before running ollama"