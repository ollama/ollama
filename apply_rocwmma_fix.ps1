$ErrorActionPreference = "Stop"
function Write-Success($text) { Write-Host "[OK] $text" -ForegroundColor Green }
Write-Host "=== Applying ALL upstream fixes ===" -ForegroundColor Cyan

$vendorHip = "ml\backend\ggml\ggml\src\ggml-cuda\vendors\hip.h"
if (Test-Path $vendorHip) {
    $content = Get-Content $vendorHip -Raw
    if ($content -notmatch 'GGML_HIP_WARP_MASK') {
        $header = @"
#ifdef GGML_HIP_ROCWMMA_FATTN
#define GGML_HIP_WARP_MASK 0xFFFFFFFFFFFFFFFFULL
#define __shfl_sync(mask, var, laneMask, width) __shfl(var, laneMask, width)
#define __shfl_xor_sync(mask, var, laneMask, width) __shfl_xor(var, laneMask, width)
#else
#define GGML_HIP_WARP_MASK 0xFFFFFFFF
#endif

#ifdef __AMDGCN_WAVEFRONT_SIZE
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-pragma"
#endif

"@
        Set-Content $vendorHip ($header + $content) -NoNewline
        Write-Success "Patched vendors/hip.h"
    }
}

$cudaDirs = @("ml\backend\ggml\ggml\src\ggml-cuda", "ml\backend\ggml\ggml\src\ggml-hip")
$count = 0
foreach ($dir in $cudaDirs) {
    if (-not (Test-Path $dir)) { continue }
    Get-ChildItem -Path $dir -Recurse -Include "*.cu", "*.cuh" | ForEach-Object {
        $c = Get-Content $_.FullName -Raw
        if ($c -match '0xFFFFFFFF|0xffffffff') {
            $c = $c -replace '0xFFFFFFFF', 'GGML_HIP_WARP_MASK'
            $c = $c -replace '0xffffffff', 'GGML_HIP_WARP_MASK'
            Set-Content $_.FullName $c -NoNewline
            $count++
        }
    }
}
Write-Success "Fixed $count CUDA files"

$hipCMake = "ml\backend\ggml\ggml\src\ggml-hip\CMakeLists.txt"
if (Test-Path $hipCMake) {
    $c = Get-Content $hipCMake -Raw
    if ($c -match 'CHECK_INCLUDE_FILE_CXX.*rocwmma' -and $c -notmatch 'ROCWMMA_SEARCH_PATHS') {
        $fix = @(
            'set(ROCWMMA_SEARCH_PATHS "/opt/rocm/include" "/usr/include" "$ENV{ROCM_PATH}/include" "$ENV{HIP_PATH}/include" "C:/Program Files/AMD/ROCm/7.1/include" "C:/Program Files/AMD/ROCm/7.2/include" "C:/opt/rocm/include")'
            'set(FOUND_ROCWMMA FALSE)'
            'foreach(PATH ${ROCWMMA_SEARCH_PATHS})'
            '    if(EXISTS "${PATH}/rocwmma/rocwmma.hpp")'
            '        set(FOUND_ROCWMMA TRUE)'
            '        include_directories(${PATH})'
            '        message(STATUS "Found rocWMMA at: ${PATH}")'
            '        break()'
            '    endif()'
            'endforeach()'
        ) -join "`n"
        $c = $c -replace 'CHECK_INCLUDE_FILE_CXX\("rocwmma/rocwmma.hpp" FOUND_ROCWMMA\)', $fix
        Set-Content $hipCMake $c -NoNewline
        Write-Success "Patched ggml-hip/CMakeLists.txt"
    }
}

Write-Host "`nALL fixes applied! Build with:" -ForegroundColor Green
Write-Host "  cmake -B build -G Ninja -DAMDGPU_TARGETS=gfx1201 -DGGML_HIP_ROCWMMA_FATTN=ON ..." -ForegroundColor Yellow
