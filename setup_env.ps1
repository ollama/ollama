$ErrorActionPreference = "Stop"
$rocmPaths = @("C:\opt\rocm", "C:\Program Files\AMD\ROCm\7.2", "C:\Program Files\AMD\ROCm\7.1", "C:\Program Files\AMD\ROCm\6.2", "C:\Program Files\AMD\ROCm\6.1")
$ROCM_PATH = $null; foreach ($p in $rocmPaths) { if (Test-Path "$p\bin\hipconfig.exe") { $ROCM_PATH = $p; break } }
if (-not $ROCM_PATH) { Write-Error "No ROCm found"; exit 1 }
$env:PATH = "$ROCM_PATH\bin;$env:PATH"
$env:HIP_PATH = $ROCM_PATH; $env:ROCM_PATH = $ROCM_PATH; $env:CMAKE_PREFIX_PATH = $ROCM_PATH
$env:AMDGPU_TARGETS = "gfx1201"
$env:GGML_HIP_ROCWMMA_FATTN = "1"
$env:GGML_HIP_MTP = "0"
$env:GGML_HIP_TURBOQUANT = "0"
$env:HIP_STREAM_PER_THREAD = "1"
Write-Host "ROCm env ready: $ROCM_PATH" -ForegroundColor Green
