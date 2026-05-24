#!/bin/bash
# build_gfx1201.sh — Linux build script for RDNA4 gfx1201
# Place in repo root, chmod +x, and run.

set -euo pipefail

# ── Environment ──────────────────────────────────────────────────────────────
export HSA_OVERRIDE_GFX_VERSION=12.0.1
export AMDGPU_TARGETS=gfx1201
export HIP_VISIBLE_DEVICES=0

# Optional: point to custom LLVM if not using system ROCm
# export PATH="/opt/rocm/llvm/bin:$PATH"
# export CMAKE_PREFIX_PATH="/opt/rocm"

# ── CMake configure ──────────────────────────────────────────────────────────
# NOTE: -DGGML_HIP_MTP=OFF because MTP speculative decoding is NOT implemented
# in the HIP backend yet. Set to ON only after implementing MTP kernels.
#
# NOTE: -DGGML_HIP_ROCWMMA_FATTN=OFF because rocWMMA is broken on gfx12.
# Your custom fattn-wmma-gfx12.cuh replaces it.
#
# NOTE: -DGGML_HIP_GFX12_WMMA=ON enables the native gfx12 WMMA path.

cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DAMDGPU_TARGETS=gfx1201 \
    -DGGML_HIP=ON \
    -DGGML_HIP_GFX12_WMMA=ON \
    -DGGML_HIP_ROCWMMA_FATTN=OFF \
    -DGGML_HIP_MTP=OFF \
    -DGGML_HIP_TURBOQUANT=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=OFF \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_HIP_COMPILER=clang++ \
    -DGGML_CCACHE=OFF \
    -S .

# ── Build ────────────────────────────────────────────────────────────────────
cmake --build build --config Release -j$(nproc)

echo "Build complete! Binary: ./build/bin/ollama"
echo ""
echo "To benchmark:"
echo "  export OLLAMA_FLASH_ATTENTION=1"
echo "  ./build/bin/ollama run devstral:latest 'What is the capital of France?' --verbose"
