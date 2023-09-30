#!/bin/sh
set -x

# Set ROCM path from environment variable or default to /opt/rocm
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
CMAKE_GGUF_ARGS=""
CMAKE_GGML_ARGS=""

# Check for CUDA and generate for GGUF and GGML
_=$(which nvcc)
if [ $? -eq 0 ]; then
    echo "Building with CUDA"
    CMAKE_GGUF_ARGS+="-DLLAMA_CUBLAS=on "
    CMAKE_GGML_ARGS+="-DLLAMA_CUBLAS=on "
elif [ -d "$ROCM_PATH" ]; then
    echo "Building with ROCm"
    CMAKE_GGUF_ARGS+="-DLLAMA_HIPBLAS=on -DCMAKE_C_COMPILER=$ROCM_PATH/llvm/bin/clang -DCMAKE_CXX_COMPILER=$ROCM_PATH/llvm/bin/clang++ "
fi

if [ -n "$CMAKE_GGUF_ARGS" ]; then
    cmake -S gguf clean
    cmake -S gguf -B gguf/build/cuda -DLLAMA_K_QUANTS=on -DLLAMA_ACCELERATE=on $CMAKE_GGUF_ARGS
    cmake --build gguf/build/cuda --target server --config Release
else
    echo "No llama.cpp supported GPU found. Skipping cuda build for cuda/gguf."
fi

if [ -n "$CMAKE_GGML_ARGS" ]; then
    cmake -S ggml clean
    cmake -S ggml -B ggml/build/cuda -DLLAMA_K_QUANTS=on -DLLAMA_ACCELERATE=on $CMAKE_GGML_ARGS
    cmake --build ggml/build/cuda --target server --config Release
else
    echo "No GGML-era llama.cpp supported GPU found. Skipping cuda build for cuda/ggml."
fi
