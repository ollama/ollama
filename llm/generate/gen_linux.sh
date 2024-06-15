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
        "gfx940"
        "gfx941"
        "gfx942"
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

init_vars
if [ -z "${OLLAMA_SKIP_STATIC_GENERATE}" -o "${OLLAMA_CPU_TARGET}" = "static" ]; then
    # Builds by default, allows skipping, forces build if OLLAMA_CPU_TARGET="static"
    # Enables optimized Dockerfile builds using a blanket skip and targeted overrides
    # Static build for linking into the Go binary
    init_vars
    CMAKE_TARGETS="--target llama --target ggml"
    CMAKE_DEFS="-DBUILD_SHARED_LIBS=off -DLLAMA_NATIVE=off -DLLAMA_AVX=off -DLLAMA_AVX2=off -DLLAMA_AVX512=off -DLLAMA_FMA=off -DLLAMA_F16C=off ${CMAKE_DEFS}"
    BUILD_DIR="../build/linux/${ARCH}_static"
    echo "Building static library"
    build
fi

init_vars
if [ -z "${OLLAMA_SKIP_CPU_GENERATE}" ]; then
    # Users building from source can tune the exact flags we pass to cmake for configuring
    # llama.cpp, and we'll build only 1 CPU variant in that case as the default.
    if [ -n "${OLLAMA_CUSTOM_CPU_DEFS}" ]; then
        init_vars
        echo "OLLAMA_CUSTOM_CPU_DEFS=\"${OLLAMA_CUSTOM_CPU_DEFS}\""
        CMAKE_DEFS="${OLLAMA_CUSTOM_CPU_DEFS} -DCMAKE_POSITION_INDEPENDENT_CODE=on ${CMAKE_DEFS}"
        BUILD_DIR="../build/linux/${ARCH}/cpu"
        echo "Building custom CPU"
        build
        compress
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
            init_vars
            CMAKE_DEFS="${COMMON_CPU_DEFS} -DLLAMA_AVX=off -DLLAMA_AVX2=off -DLLAMA_AVX512=off -DLLAMA_FMA=off -DLLAMA_F16C=off ${CMAKE_DEFS}"
            BUILD_DIR="../build/linux/${ARCH}/cpu"
            echo "Building LCD CPU"
            build
            compress
        fi

        if [ "${ARCH}" == "x86_64" ]; then
            #
            # ARM chips in M1/M2/M3-based MACs and NVidia Tegra devices do not currently support avx extensions.
            #
            if [ -z "${OLLAMA_CPU_TARGET}" -o "${OLLAMA_CPU_TARGET}" = "cpu_avx" ]; then
                #
                # ~2011 CPU Dynamic library with more capabilities turned on to optimize performance
                # Approximately 400% faster than LCD on same CPU
                #
                init_vars
                CMAKE_DEFS="${COMMON_CPU_DEFS} -DLLAMA_AVX=on -DLLAMA_AVX2=off -DLLAMA_AVX512=off -DLLAMA_FMA=off -DLLAMA_F16C=off ${CMAKE_DEFS}"
                BUILD_DIR="../build/linux/${ARCH}/cpu_avx"
                echo "Building AVX CPU"
                build
                compress
            fi

            if [ -z "${OLLAMA_CPU_TARGET}" -o "${OLLAMA_CPU_TARGET}" = "cpu_avx2" ]; then
                #
                # ~2013 CPU Dynamic library
                # Approximately 10% faster than AVX on same CPU
                #
                init_vars
                CMAKE_DEFS="${COMMON_CPU_DEFS} -DLLAMA_AVX=on -DLLAMA_AVX2=on -DLLAMA_AVX512=off -DLLAMA_FMA=on -DLLAMA_F16C=on ${CMAKE_DEFS}"
                BUILD_DIR="../build/linux/${ARCH}/cpu_avx2"
                echo "Building AVX2 CPU"
                build
                compress
            fi
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

if [ -z "${OLLAMA_SKIP_CUDA_GENERATE}" -a -d "${CUDA_LIB_DIR}" ]; then
    echo "CUDA libraries detected - building dynamic CUDA library"
    init_vars
    CUDA_MAJOR=$(ls "${CUDA_LIB_DIR}"/libcudart.so.* | head -1 | cut -f3 -d. || true)
    if [ -n "${CUDA_MAJOR}" ]; then
        CUDA_VARIANT=_v${CUDA_MAJOR}
    fi
    if [ "${ARCH}" == "arm64" ]; then
        echo "ARM CPU detected - disabling unsupported AVX instructions"

        # ARM-based CPUs such as M1 and Tegra do not support AVX extensions.
        #
        # CUDA compute < 6.0 lacks proper FP16 support on ARM.
        # Disabling has minimal performance effect while maintaining compatibility.
        ARM64_DEFS="-DLLAMA_AVX=off -DLLAMA_AVX2=off -DLLAMA_AVX512=off -DLLAMA_CUDA_F16=off"
    fi
    # Users building from source can tune the exact flags we pass to cmake for configuring llama.cpp
    if [ -n "${OLLAMA_CUSTOM_CUDA_DEFS}" ]; then
        echo "OLLAMA_CUSTOM_CUDA_DEFS=\"${OLLAMA_CUSTOM_CUDA_DEFS}\""
        CMAKE_CUDA_DEFS="-DLLAMA_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} ${OLLAMA_CUSTOM_CUDA_DEFS}"
        echo "Building custom CUDA GPU"
    else
        CMAKE_CUDA_DEFS="-DLLAMA_CUDA=on -DLLAMA_CUDA_FORCE_MMQ=on -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}"
    fi
    CMAKE_DEFS="${COMMON_CMAKE_DEFS} ${CMAKE_DEFS} ${ARM64_DEFS} ${CMAKE_CUDA_DEFS}"
    BUILD_DIR="../build/linux/${ARCH}/cuda${CUDA_VARIANT}"
    EXTRA_LIBS="-L${CUDA_LIB_DIR} -lcudart -lcublas -lcublasLt -lcuda"
    build

    # Carry the CUDA libs as payloads to help reduce dependency burden on users
    #
    # TODO - in the future we may shift to packaging these separately and conditionally
    #        downloading them in the install script.
    DEPS="$(ldd ${BUILD_DIR}/bin/ollama_llama_server )"
    for lib in libcudart.so libcublas.so libcublasLt.so ; do
        DEP=$(echo "${DEPS}" | grep ${lib} | cut -f1 -d' ' | xargs || true)
        if [ -n "${DEP}" -a -e "${CUDA_LIB_DIR}/${DEP}" ]; then
            cp "${CUDA_LIB_DIR}/${DEP}" "${BUILD_DIR}/bin/"
        elif [ -e "${CUDA_LIB_DIR}/${lib}.${CUDA_MAJOR}" ]; then
            cp "${CUDA_LIB_DIR}/${lib}.${CUDA_MAJOR}" "${BUILD_DIR}/bin/"
        elif [ -e "${CUDART_LIB_DIR}/${lib}" ]; then
            cp -d ${CUDART_LIB_DIR}/${lib}* "${BUILD_DIR}/bin/"
        else
            cp -d "${CUDA_LIB_DIR}/${lib}*" "${BUILD_DIR}/bin/"
        fi
    done
    compress

fi

if [ -z "${VULKAN_ROOT}" ]; then
    # Try the default location in case it exists
    VULKAN_ROOT=/usr/lib/
fi

if [ -z "${CAP_ROOT}" ]; then
    # Try the default location in case it exists
    CAP_ROOT=/usr/lib/
fi

if [ -z "${OLLAMA_SKIP_VULKAN_GENERATE}" -a -d "${VULKAN_ROOT}" ] && [ -z "${OLLAMA_SKIP_VULKAN_GENERATE}" -a -d "${CAP_ROOT}" ]; then
    echo "Vulkan and capabilities libraries detected - building dynamic Vulkan library"
    init_vars

    CMAKE_DEFS="${COMMON_CMAKE_DEFS} ${CMAKE_DEFS} -DLLAMA_VULKAN=1"
    BUILD_DIR="../build/linux/${ARCH}/vulkan"
    EXTRA_LIBS="-L${VULKAN_ROOT} -L${CAP_ROOT} -lvulkan -lcap"
    build

    # copy oneAPI dependencies
    for dep in $(ldd "${BUILD_DIR}/bin/ollama_llama_server" | grep "=>" | cut -f2 -d= | cut -f2 -d' ' | grep -e vulkan -e cap); do
        cp "${dep}" "${BUILD_DIR}/bin/"
    done
    cp "${VULKAN_ROOT}/libvulkan.so" "${BUILD_DIR}/bin/"
    cp "${CAP_ROOT}/libcap.so" "${BUILD_DIR}/bin/"
    compress
fi

if [ -z "${ONEAPI_ROOT}" ]; then
    # Try the default location in case it exists
    ONEAPI_ROOT=/opt/intel/oneapi
fi

if [ -z "${OLLAMA_SKIP_ONEAPI_GENERATE}" -a -d "${ONEAPI_ROOT}" ]; then
    echo "OneAPI libraries detected - building dynamic OneAPI library"
    init_vars
    source ${ONEAPI_ROOT}/setvars.sh --force # set up environment variables for oneAPI
    CC=icx
    CMAKE_DEFS="${COMMON_CMAKE_DEFS} ${CMAKE_DEFS} -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DLLAMA_SYCL=ON -DLLAMA_SYCL_F16=OFF"
    BUILD_DIR="../build/linux/${ARCH}/oneapi"
    EXTRA_LIBS="-fsycl -Wl,-rpath,${ONEAPI_ROOT}/compiler/latest/lib,-rpath,${ONEAPI_ROOT}/mkl/latest/lib,-rpath,${ONEAPI_ROOT}/tbb/latest/lib,-rpath,${ONEAPI_ROOT}/compiler/latest/opt/oclfpga/linux64/lib -lOpenCL -lmkl_core -lmkl_sycl_blas -lmkl_intel_ilp64 -lmkl_tbb_thread -ltbb"
    DEBUG_FLAGS="" # icx compiles with -O0 if we pass -g, so we must remove it
    build

    # copy oneAPI dependencies
    for dep in $(ldd "${BUILD_DIR}/bin/ollama_llama_server" | grep "=>" | cut -f2 -d= | cut -f2 -d' ' | grep -e sycl -e mkl -e tbb); do
        cp "${dep}" "${BUILD_DIR}/bin/"
    done
    cp "${ONEAPI_ROOT}/compiler/latest/lib/libOpenCL.so" "${BUILD_DIR}/bin/"
    cp "${ONEAPI_ROOT}/compiler/latest/lib/libimf.so" "${BUILD_DIR}/bin/"
    cp "${ONEAPI_ROOT}/compiler/latest/lib/libintlc.so.5" "${BUILD_DIR}/bin/"
    cp "${ONEAPI_ROOT}/compiler/latest/lib/libirng.so" "${BUILD_DIR}/bin/"
    cp "${ONEAPI_ROOT}/compiler/latest/lib/libpi_level_zero.so" "${BUILD_DIR}/bin/"
    cp "${ONEAPI_ROOT}/compiler/latest/lib/libsvml.so" "${BUILD_DIR}/bin/"
    cp "${ONEAPI_ROOT}/compiler/latest/lib/libur_loader.so.0" "${BUILD_DIR}/bin/"
    compress
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

if [ -z "${OLLAMA_SKIP_ROCM_GENERATE}" -a -d "${ROCM_PATH}" ]; then
    echo "ROCm libraries detected - building dynamic ROCm library"
    if [ -f ${ROCM_PATH}/lib/librocblas.so.*.*.????? ]; then
        ROCM_VARIANT=_v$(ls ${ROCM_PATH}/lib/librocblas.so.*.*.????? | cut -f5 -d. || true)
    fi
    init_vars
    CMAKE_DEFS="${COMMON_CMAKE_DEFS} ${CMAKE_DEFS} -DLLAMA_HIPBLAS=on -DCMAKE_C_COMPILER=$ROCM_PATH/llvm/bin/clang -DCMAKE_CXX_COMPILER=$ROCM_PATH/llvm/bin/clang++ -DAMDGPU_TARGETS=$(amdGPUs) -DGPU_TARGETS=$(amdGPUs)"
    # Users building from source can tune the exact flags we pass to cmake for configuring llama.cpp
    if [ -n "${OLLAMA_CUSTOM_ROCM_DEFS}" ]; then
        echo "OLLAMA_CUSTOM_ROCM_DEFS=\"${OLLAMA_CUSTOM_ROCM_DEFS}\""
        CMAKE_DEFS="${CMAKE_DEFS} ${OLLAMA_CUSTOM_ROCM_DEFS}"
        echo "Building custom ROCM GPU"
    fi
    BUILD_DIR="../build/linux/${ARCH}/rocm${ROCM_VARIANT}"
    EXTRA_LIBS="-L${ROCM_PATH}/lib -L/opt/amdgpu/lib/x86_64-linux-gnu/ -Wl,-rpath,\$ORIGIN/../../rocm/ -lhipblas -lrocblas -lamdhip64 -lrocsolver -lamd_comgr -lhsa-runtime64 -lrocsparse -ldrm -ldrm_amdgpu"
    build

    # Record the ROCM dependencies
    rm -f "${BUILD_DIR}/bin/deps.txt"
    touch "${BUILD_DIR}/bin/deps.txt"
    for dep in $(ldd "${BUILD_DIR}/bin/ollama_llama_server" | grep "=>" | cut -f2 -d= | cut -f2 -d' ' | grep -e rocm -e amdgpu -e libtinfo ); do
        echo "${dep}" >> "${BUILD_DIR}/bin/deps.txt"
    done
    # bomb out if for some reason we didn't get a few deps
    if [ $(cat "${BUILD_DIR}/bin/deps.txt" | wc -l ) -lt 8 ] ; then
        cat "${BUILD_DIR}/bin/deps.txt"
        echo "ERROR: deps file short"
        exit 1
    fi
    compress
fi

cleanup
echo "go generate completed.  LLM runners: $(cd ${BUILD_DIR}/..; echo *)"
