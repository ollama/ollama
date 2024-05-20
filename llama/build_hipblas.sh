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
    additional_flags="-Xclang --dependent-lib=msvcrt -Wl,/subsystem:console"
else
    output="libggml-hipblas.so"
    archs+=("${linux_archs[@]}")
fi

offload_arch_flags=""
for arch in "${archs[@]}"; do
    offload_arch_flags+=" --offload-arch=$arch"
done

hipcc \
    -parallel-jobs=12 \
    -O3 \
    -DGGML_USE_CUDA \
    -DGGML_BUILD=1 \
    -DGGML_SHARED=1 \
    -DGGML_CUDA_DMMV_X=32 \
    -DGGML_CUDA_MMV_Y=1 \
    -DGGML_SCHED_MAX_COPIES=4 \
    -DGGML_USE_HIPBLAS \
    -DGGML_USE_LLAMAFILE \
    -DHIP_FAST_MATH \
    -DNDEBUG \
    -DK_QUANTS_PER_ITERATION=2 \
    -D_CRT_SECURE_NO_WARNINGS \
    -DCMAKE_POSITION_INDEPENDENT_CODE=on \
    -D_GNU_SOURCE \
    $offload_arch_flags \
    $additional_flags \
    -Wno-expansion-to-defined \
    -Wno-invalid-noreturn \
    -Wno-ignored-attributes \
    -Wno-pass-failed \
    -Wno-deprecated-declarations \
    -I. \
    -lhipblas -lamdhip64 -lrocblas \
    -shared \
    -o ggml-hipblas.dll \
    ggml-cuda.cu ggml-cuda/*.cu ggml.c ggml-backend.c ggml-alloc.c ggml-quants.c sgemm.cpp

    # -D_DLL \
    # -D_MT \
    # -D_XOPEN_SOURCE=600 \
