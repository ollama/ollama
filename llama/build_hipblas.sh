#!/bin/bash

archs=(
    gfx900
    gfx940
    gfx941
    gfx942
    gfx1010
    gfx1012
    gfx1030
    gfx1100
    gfx1101
    gfx1102
)

linux_archs=(
    gfx906:xnack-
    gfx908:xnack-
    gfx90a:xnack+
    gfx90a:xnack-
)

os="$(uname -s)"

additional_flags=""

if [[ "$os" == "Windows_NT" || "$os" == "MINGW64_NT"* ]]; then
    output="ggml-hipblas.dll"
    additional_flags=" -Xclang --dependent-lib=msvcrt"
else
    output="libggml-hipblas.so"
    archs+=("${linux_archs[@]}")
fi

for arch in "${archs[@]}"; do
    additional_flags+=" --offload-arch=$arch"
done

# Create an array of all source files, expanding globs
sources=(
    $(echo ggml-cuda/template-instances/fattn-wmma*.cu)
    $(echo ggml-cuda/template-instances/mmq*.cu)
    $(echo ggml-cuda/template-instances/fattn-vec*q4_0-q4_0.cu)
    $(echo ggml-cuda/template-instances/fattn-vec*q8_0-q8_0.cu)
    $(echo ggml-cuda/template-instances/fattn-vec*f16-f16.cu)
    ggml-cuda.cu
    $(echo ggml-cuda/*.cu)
    ggml.c
    ggml-backend.c
    ggml-alloc.c
    ggml-quants.c
    sgemm.cpp
)

# Function to compile a single source file
compile_source() {
    src="$1"
    hipcc -c -O3 -DGGML_USE_CUDA -DGGML_BUILD=1 -DGGML_SHARED=1 -DGGML_CUDA_DMMV_X=32 -DGGML_CUDA_MMV_Y=1 \
          -DGGML_SCHED_MAX_COPIES=4 -DGGML_USE_HIPBLAS -DGGML_USE_LLAMAFILE -DHIP_FAST_MATH -DNDEBUG \
          -DK_QUANTS_PER_ITERATION=2 -D_CRT_SECURE_NO_WARNINGS -DCMAKE_POSITION_INDEPENDENT_CODE=on \
          -D_GNU_SOURCE -Wno-expansion-to-defined -Wno-invalid-noreturn -Wno-ignored-attributes -Wno-pass-failed \
          -Wno-deprecated-declarations -Wno-unused-result -I. \
          $additional_flags -o "${src%.cu}.o" "$src"
}

# Function to handle Ctrl+C
cleanup() {
    echo "Terminating all background processes..."
    kill 0
}

# Set trap to handle SIGINT (Ctrl+C)
trap cleanup SIGINT

# Limit the number of concurrent jobs
max_jobs=$(nproc)
job_count=0

for src in "${sources[@]}"; do
    echo "$src"
    compile_source "$src" &
    job_count=$((job_count + 1))
    if [[ $job_count -ge $max_jobs ]]; then
        wait -n
        job_count=$((job_count - 1))
    fi
done

wait

# Link all object files into a shared library
echo "Linking object files..."
hipcc -v -shared -o $output *.o ggml-cuda/*.o ggml-cuda/template-instances/*.o -lhipblas -lamdhip64 -lrocblas

# Clean up object files after linking
rm -f *.o ggml-cuda/*.o ggml-cuda/template-instances/*.o