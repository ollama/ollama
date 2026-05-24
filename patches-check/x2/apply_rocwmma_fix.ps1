$ErrorActionPreference = "Stop"
function Write-Success($text) { Write-Host "[OK] $text" -ForegroundColor Green }
function Write-Warn($text) { Write-Host "[WARN] $text" -ForegroundColor Yellow }
function Write-Info($text) { Write-Host "[INFO] $text" -ForegroundColor Cyan }

Write-Info "=== gfx12 WMMA Fix Applier (PowerShell) ==="
Write-Info "This replaces fragile regex patching with unified diffs."
Write-Info ""

$repoRoot = $PSScriptRoot
if (-not $repoRoot) { $repoRoot = Get-Location }

# ── Step 1: Verify required files ────────────────────────────────────────────
$requiredFiles = @(
    "fattn-wmma-gfx12-fixed.cuh",
    "fattn.cu.patch",
    "build_gfx1201.ps1"
)

foreach ($file in $requiredFiles) {
    if (-not (Test-Path "$repoRoot\$file")) {
        Write-Error "Missing required file: $file`nDownload all patch files from release assets."
        exit 1
    }
}
Write-Success "All required patch files found"

# ── Step 2: Replace kernel ───────────────────────────────────────────────────
$kernelSrc = "$repoRoot\ml\backend\ggml\ggml\src\ggml-cuda\fattn-wmma-gfx12.cuh"
$kernelFix = "$repoRoot\fattn-wmma-gfx12-fixed.cuh"

if (Test-Path $kernelSrc) {
    Copy-Item $kernelSrc "$kernelSrc.bak" -Force
    Write-Success "Backed up original fattn-wmma-gfx12.cuh"
} else {
    New-Item -ItemType Directory -Path (Split-Path $kernelSrc) -Force | Out-Null
}

Copy-Item $kernelFix $kernelSrc -Force
Write-Success "Installed fixed fattn-wmma-gfx12.cuh (v2.0)"

# ── Step 3: Patch fattn.cu ───────────────────────────────────────────────────
$fattnCu = "$repoRoot\ml\backend\ggml\ggml\src\ggml-cuda\fattn.cu"
$fattnPatch = "$repoRoot\fattn.cu.patch"

if (Test-Path $fattnCu) {
    Copy-Item $fattnCu "$fattnCu.bak" -Force
    Write-Success "Backed up original fattn.cu"

    $gitApply = Get-Command git -ErrorAction SilentlyContinue
    if ($gitApply) {
        Push-Location $repoRoot
        $result = git apply --check $fattnPatch 2>&1
        if ($LASTEXITCODE -eq 0) {
            git apply $fattnPatch
            Write-Success "Applied fattn.cu.patch via git apply"
        } else {
            Write-Warn "git apply failed — see manual merge instructions below"
            Show-ManualMerge
        }
        Pop-Location
    } else {
        Show-ManualMerge
    }
} else {
    Write-Error "fattn.cu not found at: $fattnCu"
    exit 1
}

# ── Step 4: Verify dispatch wiring ───────────────────────────────────────────
$fattnContent = Get-Content $fattnCu -Raw
$hasInclude = $fattnContent -match 'fattn-wmma-gfx12\.cuh'
$hasLaunch = $fattnContent -match 'launch_flash_attn_ext_gfx12'
$hasDispatch = $fattnContent -match 'GGML_HIP_GFX12_WMMA'

if ($hasInclude) { Write-Success "gfx12 include verified" } 
else { Write-Warn "MISSING: gfx12 include" }
if ($hasLaunch) { Write-Success "gfx12 launcher call verified" } 
else { Write-Warn "MISSING: gfx12 launcher call" }
if ($hasDispatch) { Write-Success "gfx12 dispatch path verified" } 
else { Write-Warn "MISSING: gfx12 dispatch path" }

# ── Step 5: Fix build script flags ───────────────────────────────────────────
$buildScript = "$repoRoot\build_gfx1201.ps1"
if (Test-Path $buildScript) {
    $bc = Get-Content $buildScript -Raw
    $modified = $false

    if ($bc -match '-DGGML_HIP_GFX12_WMMA\s*=\s*OFF') {
        $bc = $bc -replace '-DGGML_HIP_GFX12_WMMA\s*=\s*OFF', '-DGGML_HIP_GFX12_WMMA=ON'
        $modified = $true
    }
    if ($bc -match '-DGGML_HIP_ROCWMMA_FATTN\s*=\s*ON') {
        $bc = $bc -replace '-DGGML_HIP_ROCWMMA_FATTN\s*=\s*ON', '-DGGML_HIP_ROCWMMA_FATTN=OFF'
        $modified = $true
    }
    if ($modified) {
        Set-Content $buildScript $bc -NoNewline
        Write-Success "Fixed CMake flags in build_gfx1201.ps1"
    }
}

Write-Info ""
Write-Info "=== Fixes applied! ==="
Write-Info "Run: .\build_gfx1201.ps1"
Write-Info "Verify: $env:OLLAMA_DEBUG=1; .\build\bin\ollama.exe run llama3.1 'test' --verbose"

function Show-ManualMerge {
    Write-Info "MANUAL MERGE REQUIRED — apply these changes to fattn.cu:"
    Write-Info "  1. Add '#include \"fattn-wmma-gfx12.cuh\"' with other includes"
    Write-Info "  2. Add gfx12 fast-path in ggml_cuda_get_best_fattn_kernel():"
    Write-Info "     if (cc >= 12000 && cc < 13000 && K->ne[1] % FATTN_KQ_STRIDE == 0)"
    Write-Info "         return BEST_FATTN_KERNEL_WMMA_F16;"
    Write-Info "  3. Add launch_flash_attn_ext_gfx12() in WMMA_F16 case"
    Write-Info "See fattn.cu.patch for exact diff."
}
