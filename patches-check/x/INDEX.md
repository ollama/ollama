# gfx12 WMMA Flash Attention — Complete Fix Pack

## Files Overview

| # | File | Purpose | Action |
|---|------|---------|--------|
| 1 | `fattn-wmma-gfx12-fixed.cuh` | **Bug-fixed kernel v2.0** | Replace existing file |
| 2 | `fattn.cu.patch` | Dispatch logic fix | Patch fattn.cu |
| 3 | `build_gfx1201.sh` | Linux build script | New file |
| 4 | `apply_fixes.sh` | One-shot applier | Run once |
| 5 | `README_FIXES.md` | Full documentation | Read first |
| 6 | `fattn-wmma-gfx12-gqa-v3.cuh` | GQA-optimized kernel (future) | Integrate after v2.0 |
| 7 | `CMakeLists.txt.patch` | CMake flag wiring | Patch CMakeLists.txt |

## Quick Start (3 Steps)

```bash
# 1. Download all files into your ollama-ROCM repo root
cd /path/to/ollama-ROCM

# 2. Run the applier
chmod +x apply_fixes.sh
./apply_fixes.sh

# 3. Build
./build_gfx1201.sh   # Linux
# or
.\build_gfx1201.ps1  # Windows
```

## What Each Fix Does

### v2.0 (Immediate — apply now)
- **Register-accumulated O**: Eliminates s_O shared memory entirely. No bank conflicts.
- **Padded K/V shared memory**: `HEAD_DIM + 2` stride eliminates s_V bank conflicts.
- **All-threads masking**: 32 threads instead of 16 for causal/padding masks.
- **s_rescale array**: Decouples softmax rescale from O update, enabling register-based rescale.
- **Dispatch fix**: Ensures gfx12 kernel is actually selected at runtime (not dead code).

### v3.0 (Future — after v2.0 is stable)
- **GQA batching**: Processes 2 query heads per block, sharing KV cache loads.
- Expected **+20-40%** on Llama-3, Qwen2.5, Mistral (all GQA models).

## Verification

After building, run:
```bash
OLLAMA_DEBUG=1 ./build/bin/ollama run llama3.1 "test" --verbose 2>&1 | grep -i gfx12
```

You should see references to gfx12 WMMA in the debug output. If not, the dispatch fix didn't apply correctly.

## Support

- Original kernel author: Maxritz / ollama-ROCM
- Bug analysis & fixes: Generated analysis
- License: Same as upstream (MIT)
