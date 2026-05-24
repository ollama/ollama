#!/bin/bash
# ============================================================================
# Complete Build Script: OLLaMA + All Extensions + TurboQuant
# Target: ROCm 7.x / gfx1201 (RX 9070 XT)
# Supports: Linux (so) and Windows (dll)
# ============================================================================
set -e

OLLAMA_ROOT="${OLLAMA_ROOT:-$(pwd)}"
BUILD_DIR="${OLLAMA_ROOT}/build_ext"
HIP_PATH="${HIP_PATH:-/opt/rocm}"
VULKAN_SDK="${VULKAN_SDK:-/usr}"

# Detect platform
case "$(uname -s)" in
    Linux*)   OS=Linux;  EXT=so;  PIC="-fPIC";;
    Darwin*)  OS=Mac;    EXT=dylib; PIC="-fPIC";;
    CYGWIN*|MINGW*|MSYS*) OS=Windows; EXT=dll; PIC="";;
    *)        OS=Linux;  EXT=so;  PIC="-fPIC";;
esac

mkdir -p "${BUILD_DIR}"

echo "=== Building HIP Extensions (libggml_hip_ext.${EXT}) ==="
hipcc -O3 -ffast-math ${PIC} -shared \
    -I"${HIP_PATH}/include" \
    -I"${HIP_PATH}/include/rocblas" \
    -D__HIP_PLATFORM_AMD__ \
    -DGGML_HIP_WAVE32=1 \
    -mllvm -amdgpu-inline-threshold=10000 \
    -o "${BUILD_DIR}/libggml_hip_ext.${EXT}" \
    ggml_hip_ext.cpp \
    -L"${HIP_PATH}/lib" -lamdhip64 -lrocblas

echo "=== Building Vulkan Extensions (libggml_vulkan_ext.${EXT}) ==="
if [ "$OS" = "Windows" ]; then
    g++ -O3 ${PIC} -shared \
        -I"${VULKAN_SDK}/Include" \
        -o "${BUILD_DIR}/libggml_vulkan_ext.${EXT}" \
        ggml_vulkan_ext.cpp \
        -L"${VULKAN_SDK}/Lib" -lvulkan-1
else
    g++ -O3 ${PIC} -shared \
        -I"${VULKAN_SDK}/include" \
        -DVK_USE_PLATFORM_XLIB_KHR \
        -o "${BUILD_DIR}/libggml_vulkan_ext.${EXT}" \
        ggml_vulkan_ext.cpp \
        -L"${VULKAN_SDK}/lib" -lvulkan
fi

echo "=== Building TurboQuant (libggml_turboquant.${EXT}) ==="
hipcc -O3 -ffast-math ${PIC} -shared \
    -I"${HIP_PATH}/include" \
    -D__HIP_PLATFORM_AMD__ \
    -o "${BUILD_DIR}/libggml_turboquant.${EXT}" \
    ggml_turboquant.cpp \
    -L"${HIP_PATH}/lib" -lamdhip64

echo "=== Installing ==="
if [ "$OS" = "Windows" ]; then
    OLLAMA_LIB_DIR="${OLLAMA_ROOT}/dist/windows-amd64/lib"
else
    OLLAMA_LIB_DIR="${OLLAMA_ROOT}/dist/linux-amd64/lib"
fi

if [ -d "${OLLAMA_LIB_DIR}" ]; then
    cp "${BUILD_DIR}/libggml_hip_ext.${EXT}" "${OLLAMA_LIB_DIR}"
    cp "${BUILD_DIR}/libggml_vulkan_ext.${EXT}" "${OLLAMA_LIB_DIR}"
    cp "${BUILD_DIR}/libggml_turboquant.${EXT}" "${OLLAMA_LIB_DIR}"
    cp ggml_ext.h "${OLLAMA_ROOT}/llm/llama.cpp/ggml/include/"
    cp ggml_turboquant.h "${OLLAMA_ROOT}/llm/llama.cpp/ggml/include/"
    echo "Installed to ${OLLAMA_LIB_DIR}"
else
    echo "OLLaMA lib dir not found. Copy manually:"
    echo "  cp ${BUILD_DIR}/*.${EXT} <ollama_install>/lib/"
    echo "  cp ggml_ext.h <ollama>/llm/llama.cpp/ggml/include/"
    echo "  cp ggml_turboquant.h <ollama>/llm/llama.cpp/ggml/include/"
fi

echo "=== Done ==="
echo ""
echo "To build OLLaMA with all optimizations:"
echo "  cd ${OLLAMA_ROOT}"
echo "  cmake -B build -DGGML_HIP=ON -DGGML_HIP_ROCWMMA_FATTN=ON -DGGML_HIP_GRAPHS=ON -DGGML_TURBOQUANT=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release"
echo "  cmake --build build --parallel"
echo ""
echo "Runtime:"
if [ "$OS" = "Windows" ]; then
    echo '  set "PATH=%PATH%;'"${BUILD_DIR}"'"'
    echo '  set "LLAMA_CACHE_TYPE_K=tbq3"'
    echo '  set "LLAMA_CACHE_TYPE_V=tbq3"'
    echo "  ollama serve"
else
    echo "  export LD_LIBRARY_PATH=${BUILD_DIR}:\$LD_LIBRARY_PATH"
    echo '  export LLAMA_CACHE_TYPE_K=tbq3'
    echo '  export LLAMA_CACHE_TYPE_V=tbq3'
    echo "  ./run_optimized.sh serve"
fi
