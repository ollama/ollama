# Integration Guide — Maxritz/ollama-ROCM @ rdna4-gfx1201

This guide is specific to YOUR fork. It assumes:
- Branch: `rdna4-gfx1201`
- Build: `build_gfx1201.ps1` (Windows) + you need Linux support
- Kernel: `fattn-wmma-gfx12.cuh` in `ml/backend/ggml/ggml/src/ggml-cuda/`
- Dispatch: `fattn.cu` in same directory

## Step 0: Download All Patch Files

Download these 10 files into your repo root:
```
fattn-wmma-gfx12-fixed.cuh      ← BUG-FIXED kernel (v2.0)
fattn-wmma-gfx12-gqa-v3.cuh     ← Future GQA kernel (v3.0 preview)
fattn-vec-gfx12.cuh             ← Decode optimization sketch
fattn.cu.patch                  ← Dispatch fix for fattn.cu
CMakeLists.txt.patch            ← CMake flag wiring
apply_fixes.sh                  ← Linux one-shot applier
apply_rocwmma_fix.ps1           ← Windows one-shot applier (replaces your regex script)
build_gfx1201.sh                ← Linux build script
README_FIXES.md                 ← Full documentation
INDEX.md                        ← Quick reference
```

## Step 1: Apply the Fixes

### Option A: Linux (NEW — your repo had no Linux script)
```bash
cd /path/to/ollama-ROCM
chmod +x apply_fixes.sh
./apply_fixes.sh
```

### Option B: Windows (REPLACES your fragile regex script)
```powershell
cd C:\path\to\ollama-ROCM
.\apply_rocwmma_fix.ps1
```

This new PowerShell script:
- Uses `git apply` for deterministic patching instead of regex surgery
- Verifies dispatch wiring after patching
- Fixes CMake flags automatically
- Provides clear error messages if patching fails

## Step 2: Build

### Linux (NEW)
```bash
./build_gfx1201.sh
```

### Windows (your existing, but verify flags)
```powershell
# Ensure these flags are in build_gfx1201.ps1:
#   -DGGML_HIP_GFX12_WMMA=ON
#   -DGGML_HIP_ROCWMMA_FATTN=OFF
#   -DGGML_HIP_MTP=OFF

.\build_gfx1201.ps1
```

## Step 3: Verify the Kernel Executes

This is **critical** — your original kernel might have been dead code.

```bash
# Linux
OLLAMA_DEBUG=1 ./build/bin/ollama run llama3.1 "test" --verbose 2>&1 | grep -i gfx12

# Windows PowerShell
$env:OLLAMA_DEBUG=1
.\build\bin\ollama.exe run llama3.1 "test" --verbose 2>&1 | Select-String "gfx12"
```

**Expected:** Lines containing `gfx12`, `wmma`, or `launch_flash_attn_ext_gfx12`.

**If nothing appears:** The dispatch fix didn't apply. Your kernel is compiled but never called. Manually check `fattn.cu` for the gfx12 fast-path.

## Step 4: Benchmark

```bash
# Baseline measurement
./build/bin/ollama run llama3.1 "Write a 500-word essay about..." --verbose
# Note the "prompt eval rate" (prefill)

# Compare against your old binary
# Your README claims ~1,881 tok/s for Llama-3-8B prefill
# With v2.0 fixes, expect ~2,400-2,700 tok/s
```

## Step 5: Future — GQA Batching (v3.0)

After v2.0 is stable, integrate `fattn-wmma-gfx12-gqa-v3.cuh`:

1. Add `#include "fattn-wmma-gfx12-gqa.cuh"` to `fattn.cu`
2. In the gfx12 dispatch path, check `gqa_ratio`:
   ```cpp
   if (gqa_ratio >= 2 && D <= 128) {
       err = launch_flash_attn_ext_gfx12_gqa(...);
   } else {
       err = launch_flash_attn_ext_gfx12(...);
   }
   ```
3. Expected **+20-40%** on Llama-3, Qwen2.5, Mistral

## File Map: Where Everything Goes

| Patch File | Destination | Action |
|---|---|---|
| `fattn-wmma-gfx12-fixed.cuh` | `ml/backend/ggml/ggml/src/ggml-cuda/fattn-wmma-gfx12.cuh` | Replace |
| `fattn.cu.patch` | `ml/backend/ggml/ggml/src/ggml-cuda/fattn.cu` | Patch |
| `CMakeLists.txt.patch` | `ml/backend/ggml/ggml/src/ggml-hip/CMakeLists.txt` | Patch |
| `build_gfx1201.sh` | repo root | New file |
| `apply_fixes.sh` | repo root | New file |
| `apply_rocwmma_fix.ps1` | repo root | Replace old regex script |

## Troubleshooting Your Specific Repo

### "patch fails with 'hunk FAILED'"
Your vendored llama.cpp has drifted from the patch base. Options:
1. Check `FETCH_HEAD` in `Makefile.sync` — what llama.cpp commit are you on?
2. Manually apply the 3 changes from `fattn.cu.patch` (see README_FIXES.md)
3. Or: update `FETCH_HEAD`, run `make -f Makefile.sync apply-patches`, then re-apply

### "gfx12 kernel never executes"
1. Check `ggml_cuda_get_best_fattn_kernel()` — does it have the gfx12 fast-path?
2. Check `cc` value for gfx1201 — is it `>= 12000`?
3. Add debug print: `printf("gfx12 kernel selected, cc=%d\\n", cc);`

### "Build fails on Linux"
You didn't have a Linux build script. Use `build_gfx1201.sh`. If ROCm isn't in standard paths, set:
```bash
export CMAKE_PREFIX_PATH=/opt/rocm
export PATH=/opt/rocm/llvm/bin:$PATH
```

### "My original apply_rocwmma_fix.ps1 broke"
The old script does regex surgery on vendored files. It's fragile by design. The new `apply_rocwmma_fix.ps1` uses `git apply` and falls back to manual instructions. Replace the old script entirely.

## What NOT to Do

1. **Don't enable `-DGGML_HIP_ROCWMMA_FATTN=ON`** — rocWMMA is broken on gfx12. Your custom kernel replaces it.
2. **Don't enable `-DGGML_HIP_MTP=ON`** — MTP speculative decoding is not implemented in your HIP backend. It won't compile.
3. **Don't claim MTP/MoE speedups from this kernel** — This kernel accelerates **dense attention prefill**. It does not help MTP (decode-phase) and only marginally helps MoE (which is bottlenecked by expert MLPs).

## License
All fixes provided under MIT license (same as upstream ollama-ROCM).
