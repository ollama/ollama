#!/bin/bash

os="$(uname -s)"

if [[ "$os" == "Windows_NT" || "$os" == "MINGW64_NT"* ]]; then
    output="ggml-cuda.dll"
else
    output="libggml-cuda.so"
fi

nvcc \
    -t $(nproc) \
    --generate-code=arch=compute_50,code=[compute_50,sm_50] \
    --generate-code=arch=compute_52,code=[compute_52,sm_52] \
    --generate-code=arch=compute_61,code=[compute_61,sm_61] \
    --generate-code=arch=compute_70,code=[compute_70,sm_70] \
    --generate-code=arch=compute_75,code=[compute_75,sm_75] \
    --generate-code=arch=compute_80,code=[compute_80,sm_80] \
    -DGGML_CUDA_DMMV_X=32 \
    -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 \
    -DGGML_CUDA_MMV_Y=1 \
    -DGGML_USE_CUDA=1 \
    -DGGML_SHARED=1 \
    -DGGML_BUILD=1 \
    -DGGML_USE_LLAMAFILE \
    -D_GNU_SOURCE \
    -DCMAKE_POSITION_INDEPENDENT_CODE=on \
    -Wno-deprecated-gpu-targets \
    --forward-unknown-to-host-compiler \
    -use_fast_math \
    -link \
    -shared \
    -I. \
    -lcuda -lcublas -lcudart -lcublasLt \
    -O3 \
    -o $output \
    ggml-cuda.cu \
    ggml-cuda/*.cu \
    ggml-cuda/template-instances/fattn-wmma*.cu \
    ggml-cuda/template-instances/mmq*.cu \
    ggml-cuda/template-instances/fattn-vec*q4_0-q4_0.cu \
    ggml-cuda/template-instances/fattn-vec*q8_0-q8_0.cu \
    ggml-cuda/template-instances/fattn-vec*f16-f16.cu \
    ggml.c ggml-backend.c ggml-alloc.c ggml-quants.c sgemm.cpp

#   -DGGML_CUDA_USE_GRAPHS=1 
#   -DGGML_CUDA_FA_ALL_QUANTS=1