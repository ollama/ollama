#!/bin/bash
# This script is intended to run inside the go generate
# working directory must be ./llm/generate/

# TODO - add hardening to detect missing tools (cmake, etc.)

set -ex
set -o pipefail
compress_pids=""
echo "Starting darwin generate script"
source $(dirname $0)/gen_common.sh
init_vars
git_module_setup
apply_patches

sign() {
    if [ -n "$APPLE_IDENTITY" ]; then
        codesign -f --timestamp --deep --options=runtime --sign "$APPLE_IDENTITY" --identifier ai.ollama.ollama $1
    fi
}

COMMON_DARWIN_DEFS="-DBUILD_SHARED_LIBS=off -DCMAKE_OSX_DEPLOYMENT_TARGET=11.3 -DGGML_METAL_MACOSX_VERSION_MIN=11.3 -DCMAKE_SYSTEM_NAME=Darwin -DGGML_METAL_EMBED_LIBRARY=on -DGGML_OPENMP=off"

case "${GOARCH}" in
"amd64")
    COMMON_CPU_DEFS="${COMMON_DARWIN_DEFS} -DCMAKE_SYSTEM_PROCESSOR=${ARCH} -DCMAKE_OSX_ARCHITECTURES=${ARCH} -DGGML_METAL=off -DGGML_NATIVE=off"

    if [ -z "$OLLAMA_SKIP_CPU_GENERATE" ]; then
        #
        # CPU first for the default library, set up as lowest common denominator for maximum compatibility (including Rosetta)
        #
        init_vars
        CMAKE_DEFS="${COMMON_CPU_DEFS} -DGGML_ACCELERATE=off -DGGML_BLAS=off -DGGML_AVX=off -DGGML_AVX2=off -DGGML_AVX512=off -DGGML_FMA=off -DGGML_F16C=off ${CMAKE_DEFS}"
        RUNNER=cpu
        BUILD_DIR="../build/darwin/${GOARCH}/${RUNNER}"
        echo "Building LCD CPU"
        build
        sign ${BUILD_DIR}/bin/ollama_llama_server
        compress

        #
        # ~2011 CPU Dynamic library with more capabilities turned on to optimize performance
        # Approximately 400% faster than LCD on same CPU
        #
        init_vars
        CMAKE_DEFS="${COMMON_CPU_DEFS} -DGGML_ACCELERATE=off -DGGML_BLAS=off -DGGML_AVX=on -DGGML_AVX2=off -DGGML_AVX512=off -DGGML_FMA=off -DGGML_F16C=off ${CMAKE_DEFS}"
        RUNNER=cpu_avx
        BUILD_DIR="../build/darwin/${GOARCH}/${RUNNER}"
        echo "Building AVX CPU"
        build
        sign ${BUILD_DIR}/bin/ollama_llama_server
        compress

        #
        # ~2013 CPU Dynamic library
        # Approximately 10% faster than AVX on same CPU
        #
        init_vars
        CMAKE_DEFS="${COMMON_CPU_DEFS} -DGGML_ACCELERATE=on -DGGML_BLAS=off -DGGML_AVX=on -DGGML_AVX2=on -DGGML_AVX512=off -DGGML_FMA=on -DGGML_F16C=on ${CMAKE_DEFS}"
        RUNNER=cpu_avx2
        BUILD_DIR="../build/darwin/${GOARCH}/${RUNNER}"
        echo "Building AVX2 CPU"
        EXTRA_LIBS="${EXTRA_LIBS} -framework Accelerate -framework Foundation"
        build
        sign ${BUILD_DIR}/bin/ollama_llama_server
        compress
    fi
    ;;
"arm64")

    if [ -z "$OLLAMA_SKIP_METAL_GENERATE" ]; then
        init_vars
        CMAKE_DEFS="${COMMON_DARWIN_DEFS} -DCMAKE_SYSTEM_PROCESSOR=${ARCH} -DCMAKE_OSX_ARCHITECTURES=${ARCH} ${CMAKE_DEFS}"
        RUNNER="metal"
        BUILD_DIR="../build/darwin/${GOARCH}/${RUNNER}"
        EXTRA_LIBS="${EXTRA_LIBS} -framework Accelerate -framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders"
        build
        sign ${BUILD_DIR}/bin/ollama_llama_server
        compress
    fi
    ;;
*)
    echo "GOARCH must be set"
    echo "this script is meant to be run from within go generate"
    exit 1
    ;;
esac

cleanup
wait_for_compress
echo "go generate completed.  LLM runners: $(cd ${BUILD_DIR}/..; echo *)"
