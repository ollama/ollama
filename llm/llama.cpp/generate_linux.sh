#!/bin/bash
# only do GPU builds if nvcc is installed
if which nvcc > /dev/null; then
  # get CUDA version information
  CUDA_MAJOR_VERSION=$(nvcc --version | grep -o 'cuda_[0-9]*\.[0-9]*' | awk -F'_' '{print $2}' | awk -F'.' '{print $1}')
  
  cmake -S ggml -B ggml/build/cuda-$CUDA_MAJOR_VERSION -DLLAMA_CUBLAS=on -DLLAMA_ACCELERATE=on -DLLAMA_K_QUANTS=on
  cmake --build ggml/build/cuda-$CUDA_MAJOR_VERSION --target server --config Release
  cmake -S gguf -B gguf/build/cuda-$CUDA_MAJOR_VERSION -DLLAMA_CUBLAS=on -DLLAMA_ACCELERATE=on -DLLAMA_K_QUANTS=on
  cmake --build gguf/build/cuda-$CUDA_MAJOR_VERSION --target server --config Release
else
  echo "Warning: nvcc is not installed, skipping CUDA builds."
fi