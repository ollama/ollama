#!/bin/bash
# This script is intended to run inside the go generate
# working directory must be ./llm

# TODO - add hardening to detect missing tools (cmake, etc.)

set -ex
set -o pipefail
echo "Starting darwin generate script"
source $(dirname $0)/build_common.sh
init_vars
git_module_setup
apply_patches

sign() {
    if [ -n "$APPLE_IDENTITY" ]; then
        codesign -f --timestamp --deep --options=runtime --sign "$APPLE_IDENTITY" --identifier ai.ollama.ollama $1
    fi
}

COMMON_DARWIN_DEFS="-DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 -DCMAKE_SYSTEM_NAME=Darwin -DLLAMA_SERVER_VERBOSE=on"

case "${GOARCH}" in
"amd64")
    COMMON_CPU_DEFS="${COMMON_DARWIN_DEFS} -DCMAKE_SYSTEM_PROCESSOR=${ARCH} -DCMAKE_OSX_ARCHITECTURES=${ARCH} -DLLAMA_METAL=off -DLLAMA_NATIVE=off"

    #
    # CPU first for the default library, set up as lowest common denominator for maximum compatibility (including Rosetta)
    #
    CMAKE_DEFS="${COMMON_CPU_DEFS} -DLLAMA_ACCELERATE=off -DLLAMA_AVX=off -DLLAMA_AVX2=off -DLLAMA_AVX512=off -DLLAMA_FMA=off -DLLAMA_F16C=off ${CMAKE_DEFS}"
    BUILD_DIR="build/darwin/${ARCH}/cpu"
    echo "Building LCD CPU"
    build
    sign build/darwin/${ARCH}/cpu/bin/server
    compress

    #
    # ~2011 CPU Dynamic library with more capabilities turned on to optimize performance
    # Approximately 400% faster than LCD on same CPU
    #
    init_vars
    CMAKE_DEFS="${COMMON_CPU_DEFS} -DLLAMA_ACCELERATE=off -DLLAMA_AVX=on -DLLAMA_AVX2=off -DLLAMA_AVX512=off -DLLAMA_FMA=off -DLLAMA_F16C=off ${CMAKE_DEFS}"
    BUILD_DIR="build/darwin/${ARCH}/cpu_avx"
    echo "Building AVX CPU"
    build
    sign build/darwin/${ARCH}/cpu_avx/bin/server
    compress

    #
    # ~2013 CPU Dynamic library
    # Approximately 10% faster than AVX on same CPU
    #
    init_vars
    CMAKE_DEFS="${COMMON_CPU_DEFS} -DLLAMA_ACCELERATE=on -DLLAMA_AVX=on -DLLAMA_AVX2=on -DLLAMA_AVX512=off -DLLAMA_FMA=on -DLLAMA_F16C=on ${CMAKE_DEFS}"
    BUILD_DIR="build/darwin/${ARCH}/cpu_avx2"
    echo "Building AVX2 CPU"
    EXTRA_LIBS="${EXTRA_LIBS} -framework Accelerate -framework Foundation"
    build
    sign build/darwin/${ARCH}/cpu_avx2/bin/server
    compress
    ;;
"arm64")
    CMAKE_DEFS="${COMMON_DARWIN_DEFS} -DLLAMA_ACCELERATE=on -DCMAKE_SYSTEM_PROCESSOR=${ARCH} -DCMAKE_OSX_ARCHITECTURES=${ARCH} -DLLAMA_METAL=on ${CMAKE_DEFS}"
    BUILD_DIR="build/darwin/${ARCH}/metal"
    EXTRA_LIBS="${EXTRA_LIBS} -framework Accelerate -framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders"
    build
    sign build/darwin/${ARCH}/metal/bin/server
    compress
    ;;
*)
    echo "GOARCH must be set"
    echo "this script is meant to be run from within go generate"
    exit 1
    ;;
esac

cleanup
