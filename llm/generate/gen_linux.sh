#!/bin/bash
# This script is intended to run inside the go generate
# working directory must be llm/generate/

# First we build one or more CPU based LLM libraries
#
# Then if we detect CUDA, we build a CUDA dynamic library, and carry the required
# library dependencies
#
# Then if we detect ROCm, we build a dynamically loaded ROCm lib.  The ROCM
# libraries are quite large, and also dynamically load data files at runtime
# which in turn are large, so we don't attempt to cary them as payload

set -ex
set -o pipefail

# See https://llvm.org/docs/AMDGPUUsage.html#processors for reference
amdGPUs() {
    if [ -n "${AMDGPU_TARGETS}" ]; then
        echo "${AMDGPU_TARGETS}"
        return
    fi
    GPU_LIST=(
        "gfx900"
        "gfx906:xnack-"
        "gfx908:xnack-"
        "gfx90a:xnack+"
        "gfx90a:xnack-"
        "gfx1010"
        "gfx1012"
        "gfx1030"
        "gfx1100"
        "gfx1101"
        "gfx1102"
    )
    (
        IFS=$';'
        echo "'${GPU_LIST[*]}'"
    )
}

echo "Starting linux generate script"
if [ -z "${CUDACXX}" ]; then
    if [ -x /usr/local/cuda/bin/nvcc ]; then
        export CUDACXX=/usr/local/cuda/bin/nvcc
    else
        # Try the default location in case it exists
        export CUDACXX=$(command -v nvcc)
    fi
fi
COMMON_CMAKE_DEFS="-DCMAKE_POSITION_INDEPENDENT_CODE=on -DLLAMA_NATIVE=off -DLLAMA_AVX=on -DLLAMA_AVX2=off -DLLAMA_AVX512=off -DLLAMA_FMA=off -DLLAMA_F16C=off"
source $(dirname $0)/gen_common.sh
init_vars
git_module_setup
apply_patches

if [ -z "${OLLAMA_SKIP_CPU_GENERATE}" ]; then
    # Users building from source can tune the exact flags we pass to cmake for configuring
    # llama.cpp, and we'll build only 1 CPU variant in that case as the default.
    if [ -n "${OLLAMA_CUSTOM_CPU_DEFS}" ]; then
        echo "OLLAMA_CUSTOM_CPU_DEFS=\"${OLLAMA_CUSTOM_CPU_DEFS}\""
        CMAKE_DEFS="${OLLAMA_CUSTOM_CPU_DEFS} -DCMAKE_POSITION_INDEPENDENT_CODE=on ${CMAKE_DEFS}"
        BUILD_DIR="${LLAMACPP_DIR}/build/linux/${ARCH}/cpu"
        echo "Building custom CPU"
        build
        compress_libs
    else
        # Darwin Rosetta x86 emulation does NOT support AVX, AVX2, AVX512
        # -DLLAMA_AVX -- 2011 Intel Sandy Bridge & AMD Bulldozer
        # -DLLAMA_F16C -- 2012 Intel Ivy Bridge & AMD 2011 Bulldozer (No significant improvement over just AVX)
        # -DLLAMA_AVX2 -- 2013 Intel Haswell & 2015 AMD Excavator / 2017 AMD Zen
        # -DLLAMA_FMA (FMA3) -- 2013 Intel Haswell & 2012 AMD Piledriver
        # Note: the following seem to yield slower results than AVX2 - ymmv
        # -DLLAMA_AVX512 -- 2017 Intel Skylake and High End DeskTop (HEDT)
        # -DLLAMA_AVX512_VBMI -- 2018 Intel Cannon Lake
        # -DLLAMA_AVX512_VNNI -- 2021 Intel Alder Lake

        COMMON_CPU_DEFS="-DCMAKE_POSITION_INDEPENDENT_CODE=on -DLLAMA_NATIVE=off"
        if [ -z "${OLLAMA_CPU_TARGET}" -o "${OLLAMA_CPU_TARGET}" = "cpu" ]; then
            #
            # CPU first for the default library, set up as lowest common denominator for maximum compatibility (including Rosetta)
            #
            CMAKE_DEFS="${COMMON_CPU_DEFS} -DLLAMA_AVX=off -DLLAMA_AVX2=off -DLLAMA_AVX512=off -DLLAMA_FMA=off -DLLAMA_F16C=off ${CMAKE_DEFS}"
            BUILD_DIR="${LLAMACPP_DIR}/build/linux/${ARCH}/cpu"
            echo "Building LCD CPU"
            build
            compress_libs
        fi

        if [ -z "${OLLAMA_CPU_TARGET}" -o "${OLLAMA_CPU_TARGET}" = "cpu_avx" ]; then
            #
            # ~2011 CPU Dynamic library with more capabilities turned on to optimize performance
            # Approximately 400% faster than LCD on same CPU
            #
            init_vars
            CMAKE_DEFS="${COMMON_CPU_DEFS} -DLLAMA_AVX=on -DLLAMA_AVX2=off -DLLAMA_AVX512=off -DLLAMA_FMA=off -DLLAMA_F16C=off ${CMAKE_DEFS}"
            BUILD_DIR="${LLAMACPP_DIR}/build/linux/${ARCH}/cpu_avx"
            echo "Building AVX CPU"
            build
            compress_libs
        fi

        if [ -z "${OLLAMA_CPU_TARGET}" -o "${OLLAMA_CPU_TARGET}" = "cpu_avx2" ]; then
            #
            # ~2013 CPU Dynamic library
            # Approximately 10% faster than AVX on same CPU
            #
            init_vars
            CMAKE_DEFS="${COMMON_CPU_DEFS} -DLLAMA_AVX=on -DLLAMA_AVX2=on -DLLAMA_AVX512=off -DLLAMA_FMA=on -DLLAMA_F16C=on ${CMAKE_DEFS}"
            BUILD_DIR="${LLAMACPP_DIR}/build/linux/${ARCH}/cpu_avx2"
            echo "Building AVX2 CPU"
            build
            compress_libs
        fi
    fi
else
    echo "Skipping CPU generation step as requested"
fi

# If needed, look for the default CUDA toolkit location
if [ -z "${CUDA_LIB_DIR}" ] && [ -d /usr/local/cuda/lib64 ]; then
    CUDA_LIB_DIR=/usr/local/cuda/lib64
fi

# If needed, look for CUDA on Arch Linux
if [ -z "${CUDA_LIB_DIR}" ] && [ -d /opt/cuda/targets/x86_64-linux/lib ]; then
    CUDA_LIB_DIR=/opt/cuda/targets/x86_64-linux/lib
fi

# Allow override in case libcudart is in the wrong place
if [ -z "${CUDART_LIB_DIR}" ]; then
    CUDART_LIB_DIR="${CUDA_LIB_DIR}"
fi

if [ -d "${CUDA_LIB_DIR}" ]; then
    echo "CUDA libraries detected - building dynamic CUDA library"
    init_vars
    CUDA_MAJOR=$(ls "${CUDA_LIB_DIR}"/libcudart.so.* | head -1 | cut -f3 -d. || true)
    if [ -n "${CUDA_MAJOR}" ]; then
        CUDA_VARIANT=_v${CUDA_MAJOR}
    fi
    CMAKE_DEFS="-DLLAMA_CUBLAS=on -DLLAMA_CUDA_FORCE_MMQ=on -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} ${COMMON_CMAKE_DEFS} ${CMAKE_DEFS}"
    BUILD_DIR="${LLAMACPP_DIR}/build/linux/${ARCH}/cuda${CUDA_VARIANT}"
    EXTRA_LIBS="-L${CUDA_LIB_DIR} -lcudart -lcublas -lcublasLt -lcuda"
    build

    # Cary the CUDA libs as payloads to help reduce dependency burden on users
    #
    # TODO - in the future we may shift to packaging these separately and conditionally
    #        downloading them in the install script.
    DEPS="$(ldd ${BUILD_DIR}/lib/libext_server.so )"
    for lib in libcudart.so libcublas.so libcublasLt.so ; do
        DEP=$(echo "${DEPS}" | grep ${lib} | cut -f1 -d' ' | xargs || true)
        if [ -n "${DEP}" -a -e "${CUDA_LIB_DIR}/${DEP}" ]; then
            cp "${CUDA_LIB_DIR}/${DEP}" "${BUILD_DIR}/lib/"
        elif [ -e "${CUDA_LIB_DIR}/${lib}.${CUDA_MAJOR}" ]; then
            cp "${CUDA_LIB_DIR}/${lib}.${CUDA_MAJOR}" "${BUILD_DIR}/lib/"
        elif [ -e "${CUDART_LIB_DIR}/${lib}" ]; then
            cp -d ${CUDART_LIB_DIR}/${lib}* "${BUILD_DIR}/lib/"
        else
            cp -d "${CUDA_LIB_DIR}/${lib}*" "${BUILD_DIR}/lib/"
        fi
    done
    compress_libs

fi

if [ -z "${ROCM_PATH}" ]; then
    # Try the default location in case it exists
    ROCM_PATH=/opt/rocm
fi

if [ -z "${CLBlast_DIR}" ]; then
    # Try the default location in case it exists
    if [ -d /usr/lib/cmake/CLBlast ]; then
        export CLBlast_DIR=/usr/lib/cmake/CLBlast
    fi
fi

if [ -d "${ROCM_PATH}" ]; then
    echo "ROCm libraries detected - building dynamic ROCm library"
    if [ -f ${ROCM_PATH}/lib/librocblas.so.*.*.????? ]; then
        ROCM_VARIANT=_v$(ls ${ROCM_PATH}/lib/librocblas.so.*.*.????? | cut -f5 -d. || true)
    fi
    init_vars
    CMAKE_DEFS="${COMMON_CMAKE_DEFS} ${CMAKE_DEFS} -DLLAMA_HIPBLAS=on -DCMAKE_C_COMPILER=$ROCM_PATH/llvm/bin/clang -DCMAKE_CXX_COMPILER=$ROCM_PATH/llvm/bin/clang++ -DAMDGPU_TARGETS=$(amdGPUs) -DGPU_TARGETS=$(amdGPUs)"
    BUILD_DIR="${LLAMACPP_DIR}/build/linux/${ARCH}/rocm${ROCM_VARIANT}"
    EXTRA_LIBS="-L${ROCM_PATH}/lib -L/opt/amdgpu/lib/x86_64-linux-gnu/ -Wl,-rpath,\$ORIGIN/../rocm/ -lhipblas -lrocblas -lamdhip64 -lrocsolver -lamd_comgr -lhsa-runtime64 -lrocsparse -ldrm -ldrm_amdgpu"
    build

    # Record the ROCM dependencies
    rm -f "${BUILD_DIR}/lib/deps.txt"
    touch "${BUILD_DIR}/lib/deps.txt"
    for dep in $(ldd "${BUILD_DIR}/lib/libext_server.so" | grep "=>" | cut -f2 -d= | cut -f2 -d' ' | grep -e rocm -e amdgpu -e libtinfo ); do
        echo "${dep}" >> "${BUILD_DIR}/lib/deps.txt"
    done
    compress_libs
fi

cleanup
