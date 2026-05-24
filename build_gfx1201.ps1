param([switch]$Clean, [switch]$Installer, [switch]$FastMath)
$ErrorActionPreference = "Continue"
. .\setup_env.ps1
if ($Clean -and (Test-Path "build")) { Remove-Item -Recurse -Force "build" }
. .\apply_rocwmma_fix.ps1
# HIP flags: only HIP-safe flags here; gfx1201 LLVM flags are applied via target_compile_options in CMakeLists.txt
$hipFlags = "-parallel-jobs=4 -isystem C:/llvm/compiler/include -w"
# CXX flags: do NOT use -ffinite-math-only (part of -ffast-math) — ggml cpu vec.h requires NaN/Inf support
$cxxFlags = "-isystem C:/llvm/compiler/include"
if ($FastMath) {
    # GPU: full fast-math is safe
    $hipFlags += " -ffast-math"
    # CPU: fast-math but preserve finite-math-only=no for ggml cpu compatibility
    $cxxFlags += " -ffast-math -fno-finite-math-only"
}
cmake -B build -G Ninja -Wno-dev `
    -DCMAKE_C_COMPILER="C:/llvm/compiler/bin/clang.exe" `
    -DCMAKE_CXX_COMPILER="C:/llvm/compiler/bin/clang++.exe" `
    -DCMAKE_HIP_COMPILER="C:/llvm/compiler/bin/clang++.exe" `
    -DAMDGPU_TARGETS=gfx1201 `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_PREFIX_PATH="C:/Program Files/AMD/ROCm/7.1" `
    -DGGML_CCACHE=OFF `
    -DCMAKE_C_COMPILER_LAUNCHER="" `
    -DCMAKE_CXX_COMPILER_LAUNCHER="" `
    -DCMAKE_HIP_COMPILER_LAUNCHER="" `
    -DGGML_HIP_ROCWMMA_FATTN=OFF `
    -DGGML_HIP_GFX12_WMMA=ON `
    -DGGML_HIP_MTP=OFF `
    -DGGML_HIP_TURBOQUANT=ON `
    -DCMAKE_HIP_FLAGS="$hipFlags" `
    -DCMAKE_CXX_FLAGS="$cxxFlags" `
    -S .
if ($LASTEXITCODE -ne 0) { Write-Error "Configure failed"; exit 1 }
cmake --build build --config Release
if ($LASTEXITCODE -ne 0) { Write-Error "Build failed"; exit 1 }
Write-Host "Build complete!" -ForegroundColor Green
if ($Installer) { powershell -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1 }
