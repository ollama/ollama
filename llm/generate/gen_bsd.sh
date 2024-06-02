#!/bin/sh
# This script is intended to run inside the go generate
# working directory must be ./llm/generate/

set -ex
set -o pipefail
echo "Starting BSD generate script"
. $(dirname $0)/gen_common.sh
init_vars
git_module_setup
apply_patches

COMMON_BSD_DEFS="-DCMAKE_SYSTEM_NAME=$(uname -s)"
CMAKE_TARGETS="--target llama --target ggml"

case "${GOARCH}" in
  "amd64")
    COMMON_CPU_DEFS="${COMMON_BSD_DEFS} -DCMAKE_SYSTEM_PROCESSOR=${ARCH}"

    # Static build for linking into the Go binary
    init_vars
    CMAKE_DEFS="${COMMON_CPU_DEFS} -DBUILD_SHARED_LIBS=off -DLLAMA_ACCELERATE=off -DLLAMA_AVX=off -DLLAMA_AVX2=off -DLLAMA_AVX512=off -DLLAMA_FMA=off -DLLAMA_F16C=off ${CMAKE_DEFS}"
    BUILD_DIR="../build/bsd/${ARCH}_static"
    echo "Building static library"
    build

    init_vars
    CMAKE_DEFS="${COMMON_CPU_DEFS} -DLLAMA_AVX=off -DLLAMA_AVX2=off -DLLAMA_AVX512=off -DLLAMA_FMA=off -DLLAMA_F16C=off ${CMAKE_DEFS}"
    BUILD_DIR="../build/bsd/${ARCH}/cpu"
    echo "Building LCD CPU"
    build
    compress

    init_vars
    CMAKE_DEFS="${COMMON_CPU_DEFS} -DLLAMA_AVX=on -DLLAMA_AVX2=off -DLLAMA_AVX512=off -DLLAMA_FMA=off -DLLAMA_F16C=off ${CMAKE_DEFS}"
    BUILD_DIR="../build/bsd/${ARCH}/cpu_avx"
    echo "Building AVX CPU"
    build
    compress

    init_vars
    CMAKE_DEFS="${COMMON_CPU_DEFS} -DLLAMA_AVX=on -DLLAMA_AVX2=on -DLLAMA_AVX512=off -DLLAMA_FMA=on -DLLAMA_F16C=on ${CMAKE_DEFS}"
    BUILD_DIR="../build/bsd/${ARCH}/cpu_avx2"
    echo "Building AVX2 CPU"
    build
    compress

    init_vars
    CMAKE_DEFS="${COMMON_CPU_DEFS} -DLLAMA_VULKAN=on ${CMAKE_DEFS}"
    BUILD_DIR="../build/bsd/${ARCH}/vulkan"
    echo "Building Vulkan GPU"
    build
    compress
    ;;

  *)
    echo "GOARCH must be set"
    echo "this script is meant to be run from within go generate"
    exit 1
    ;;
esac

cleanup
echo "go generate completed.  LLM runners: $(cd ${BUILD_DIR}/..; echo *)"
