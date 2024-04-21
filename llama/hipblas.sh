hipcc \
    -parallel-jobs=12 \
    -O3 \
    --offload-arch=gfx1100 \
    -DGGML_USE_CUDA \
    -DGGML_BUILD=1 \
    -DGGML_SHARED=1 \
    -DGGML_CUDA_DMMV_X=32 \
    -DGGML_CUDA_MMV_Y=1 \
    -DGGML_SCHED_MAX_COPIES=4 \
    -DGGML_USE_HIPBLAS \
    -DGGML_USE_LLAMAFILE \
    -DK_QUANTS_PER_ITERATION=2 \
    -D_CRT_SECURE_NO_WARNINGS \
    -Xclang --dependent-lib=msvcrt -Wl,/subsystem:console \
    -Wno-expansion-to-defined \
    -Wno-invalid-noreturn \
    -Wno-ignored-attributes \
    -Wno-pass-failed \
    -Wno-deprecated-declarations \
    -I. \
    -lhipblas -lamdhip64 -lrocblas \
    -shared \
    -o ggml-hipblas.dll \
    ggml-cuda.cu ggml-cuda/*.cu ggml.c ggml-backend.c ggml-alloc.c ggml-quants.c

#   --offload-arch=gfx900 \
#     --offload-arch=gfx906:xnack- \
#     --offload-arch=gfx908:xnack- \
#     --offload-arch=gfx90a:xnack+ \
#     --offload-arch=gfx90a:xnack- \
#     --offload-arch=gfx940 \
#     --offload-arch=gfx941 \
#     --offload-arch=gfx942 \
#     --offload-arch=gfx1010 \
#     --offload-arch=gfx1012 \
#     --offload-arch=gfx1030 \
#     --offload-arch=gfx1100 \
#     --offload-arch=gfx1101 \
#     --offload-arch=gfx1102 \

# hipcc \
#     -v \
#     -parallel-jobs=12 \
#     --offload-arch=gfx900 \
#     --offload-arch=gfx940 \
#     --offload-arch=gfx941 \
#     --offload-arch=gfx942 \
#     --offload-arch=gfx1010 \
#     --offload-arch=gfx1012 \
#     --offload-arch=gfx1030 \
#     --offload-arch=gfx1100 \
#     --offload-arch=gfx1101 \
#     --offload-arch=gfx1102 \
#     -DLLAMA_BUILD \
#     -DLLAMA_SHARED \
#     -DK_QUANTS_PER_ITERATION=2 \
#     -DGGML_USE_CUDA=1 \
#     -DGGML_USE_HIPBLAS \
#     -DGGML_CUDA_DMMV_X=32 \
#     -DGGML_CUDA_MMV_Y=1 \
#     -DGGML_CUDA_FORCE_DMMV \
#     -DNDEBUG \
#     -D_DLL \
#     -D_MT \
#     -D_XOPEN_SOURCE=600 \
#     -D_CRT_SECURE_NO_WARNINGS \
#     -Xclang \
#     --dependent-lib=msvcrt \
#     -std=gnu++14 \
#     -Wmissing-declarations \
#     -Wmissing-noreturn \
#     -Wall \
#     -Wextra \
#     -Wpedantic \
#     -Wcast-qual \
#     -Wno-unused-function \
#     -Wunreachable-code-break \
#     -Wunreachable-code-return \
#     -Wmissing-prototypes \
#     -Wextra-semi \
#     -Wno-cast-qual \
#     -I. \
#     -lhipblas -lamdhip64 -lrocblas \
#     -Wno-expansion-to-defined \
#     -Wno-invalid-noreturn \
#     -Wno-deprecated-declarations \
#     -mllvm \
#     -x hip \
#     -shared -o ggml-cuda.dll \
#     ggml-cuda.cu ggml-cuda/*.cu ggml.c ggml-backend.c ggml-alloc.c ggml-quants.c 
