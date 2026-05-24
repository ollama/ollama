#!/bin/bash
# ============================================================================
# Build script for OLLaMA ROCm 7.x + Vulkan aggressive extensions
# Supports: Linux (.so), Windows (.dll)
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

echo "=== Installing ==="
if [ "$OS" = "Windows" ]; then
    OLLAMA_LIB_DIR="${OLLAMA_ROOT}/dist/windows-amd64/lib"
else
    OLLAMA_LIB_DIR="${OLLAMA_ROOT}/dist/linux-amd64/lib"
fi

if [ -d "${OLLAMA_LIB_DIR}" ]; then
    cp "${BUILD_DIR}/libggml_hip_ext.${EXT}" "${OLLAMA_LIB_DIR}"
    cp "${BUILD_DIR}/libggml_vulkan_ext.${EXT}" "${OLLAMA_LIB_DIR}"
    echo "Installed to ${OLLAMA_LIB_DIR}"
else
    echo "OLLaMA lib dir not found. Copy manually:"
    echo "  cp ${BUILD_DIR}/*.${EXT} <ollama_install>/lib/"
fi

echo "=== Done ==="
if [ "$OS" = "Windows" ]; then
    echo "Set PATH=${BUILD_DIR};%PATH% before running ollama"
else
    echo "Set LD_LIBRARY_PATH=${BUILD_DIR}:\$LD_LIBRARY_PATH before running ollama"
fi
