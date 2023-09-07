#!/bin/bash

# only do gpu builds if nvcc is installed
if which nvcc > /dev/null; then
  cmake --fresh -S ggml -B ggml/build/gpu -DLLAMA_CUBLAS=on -DLLAMA_ACCELERATE=on -DLLAMA_K_QUANTS=on
  cmake --build ggml/build/gpu --target server --config Release

  cmake --fresh -S gguf -B gguf/build/gpu -DLLAMA_CUBLAS=on -DLLAMA_ACCELERATE=on -DLLAMA_K_QUANTS=on
  cmake --build gguf/build/gpu --target server --config Release
else
  echo "Warning: nvcc is not installed, skipping gpu builds."
fi
