#!/bin/sh
set -x

# Set ROCM path from environment variable or default to /opt/rocm.
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
CLBlast_DIR="${CLBlast_DIR:-/usr/lib/cmake/CLBlast}"

# Accumulate CMake options in per-target variables.
CMAKE_GGUF_ARGS=""
CMAKE_GGML_ARGS=""

# Check for CUDA and generate for GGUF and GGML
_=$(which nvcc)
if [ $? -eq 0 ]; then
	# We found nvcc. Proceed with CUDA!"
	echo "Building with CUDA"
	# We probably don't need CLBlast here, so disable it in case presence is determined automagically.
	CMAKE_GGUF_ARGS+="-DLLAMA_CUBLAS=on -DLLAMA_CLBLAST=off"
	CMAKE_GGML_ARGS+="-DLLAMA_CUBLAS=on -DLLAMA_CLBLAST=off"
elif [ -d "$ROCM_PATH" ]; then
	# We did not find nvcc, but we found the ROCm SDK. Build for ROCm/HIP.
	echo "Building with ROCm"
	# Can't compile gguf with CLBlast and ROCm due to library conflicts. Turn off CLBlast explicitly.
	CMAKE_GGUF_ARGS+="-DLLAMA_HIPBLAS=on -DCMAKE_C_COMPILER=$ROCM_PATH/llvm/bin/clang -DCMAKE_CXX_COMPILER=$ROCM_PATH/llvm/bin/clang++ -DLLAMA_CLBLAST=off"
	if [ -d "$CLBlast_DIR" ]; then
		# ROCm doesn't work for old llama.cpp, so we can fall back to CLBlast here if it is present.
		CMAKE_GGML_ARGS+="-DLLAMA_CLBLAST=on -DCLBlast_DIR=$CLBlast_DIR"
	fi
elif [ -d "$CLBlast_DIR" ]; then
	# No ROCm, No CUDA, so try CLBlast for both binaries.
	CMAKE_GGUF_ARGS+="-DLLAMA_CLBLAST=on -DCLBlast_DIR=$CLBlast_DIR"
	CMAKE_GGML_ARGS+="-DLLAMA_CLBLAST=on -DCLBlast_DIR=$CLBlast_DIR"
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
