#!/bin/bash
# OLLaMA ROCm 7.x Maximum Aggression Launcher
# Apply before running ollama serve or ollama run

export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_NOPRUNE=1

# ROCm/HIP Runtime Tuning
export HSA_ENABLE_SDMA=1
export HSA_SDMA_PERF_MODE=1
export HSA_USE_HOST_ALLOC=0
export HIP_FORCE_DEV_KERNARG=1
export AMD_DIRECT_DISPATCH=1
export AMD_SHADER_CACHE=1
export AMD_SHADER_CACHE_PATH=/tmp/amd_shader_cache
export GPU_MAX_HW_QUEUES=8
export HIP_VISIBLE_DEVICES=0

# Memory & Allocation Aggression
export HSA_XNACK=0
export HIP_HOST_COHERENT=0
export HSA_USERPTR_FOR_PAGED_MEM=0
export GPU_SINGLE_ALLOC_PERCENT=80
export GPU_MAX_ALLOC_PERCENT=95
export GPU_ENABLE_ALLOC_CACHE=1

# Compiler/JIT Aggression (for runtime compiled kernels)
export HIPCC_COMPILE_FLAGS_APPEND="-O3 -ffast-math -funroll-loops -finline-functions -march=native -fomit-frame-pointer -DNDEBUG -flto"
export AMDGPU_TARGETS="gfx1200;gfx1201;gfx1100;gfx1101;gfx1030"

# GGML/LLaMA.cpp Backend Tuning
export GGML_HIP_GRAPHS=1
export GGML_HIP_UMA=0
export GGML_CUDA_FORCE_CUBLAS=0
export GGML_CUDA_NO_VMM=0

# Vulkan Workarounds (if using Vulkan backend for anything)
export MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS=1
export MVK_CONFIG_PREFILL_METAL_COMMAND_BUFFERS=1

# Threading
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export GOTOBLAS_MAIN_FREE=1

# Scheduler Aggression
export OLLAMA_SCHED_SPREAD=1
export OLLAMA_KEEP_ALIVE=-1

# Create shader cache dir
mkdir -p /tmp/amd_shader_cache

# Run with taskset on isolated cores if possible (Linux only)
if [ "$(uname -s)" = "Linux" ] && command -v taskset &> /dev/null; then
    OMP_NUM_THREADS=$(nproc)
    exec taskset -c 0-$(($(nproc)-1)) ollama "$@"
else
    OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
    exec ollama "$@"
fi
