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
compress_pids=""

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
COMMON_CMAKE_DEFS="-DCMAKE_SKIP_RPATH=on -DBUILD_SHARED_LIBS=on -DCMAKE_POSITION_INDEPENDENT_CODE=on -DGGML_NATIVE=off -DGGML_AVX=on -DGGML_AVX2=off -DGGML_AVX512=off -DGGML_FMA=off -DGGML_F16C=off -DGGML_OPENMP=off"
source $(dirname $0)/gen_common.sh
init_vars
git_module_setup
apply_patches

init_vars
if [ -z "${OLLAMA_SKIP_CPU_GENERATE}" ]; then
    # Users building from source can tune the exact flags we pass to cmake for configuring
    # llama.cpp, and we'll build only 1 CPU variant in that case as the default.
    if [ -n "${OLLAMA_CUSTOM_CPU_DEFS}" ]; then
        init_vars
        echo "OLLAMA_CUSTOM_CPU_DEFS=\"${OLLAMA_CUSTOM_CPU_DEFS}\""
        CMAKE_DEFS="${OLLAMA_CUSTOM_CPU_DEFS} -DBUILD_SHARED_LIBS=on -DCMAKE_POSITION_INDEPENDENT_CODE=on ${CMAKE_DEFS}"
        RUNNER="cpu"
        BUILD_DIR="../build/linux/${GOARCH}/${RUNNER}"
        echo "Building custom CPU"
        build
        install
        dist
        compress
    else
        # Darwin Rosetta x86 emulation does NOT support AVX, AVX2, AVX512
        # -DGGML_AVX -- 2011 Intel Sandy Bridge & AMD Bulldozer
        # -DGGML_F16C -- 2012 Intel Ivy Bridge & AMD 2011 Bulldozer (No significant improvement over just AVX)
        # -DGGML_AVX2 -- 2013 Intel Haswell & 2015 AMD Excavator / 2017 AMD Zen
        # -DGGML_FMA (FMA3) -- 2013 Intel Haswell & 2012 AMD Piledriver
        # Note: the following seem to yield slower results than AVX2 - ymmv
        # -DGGML_AVX512 -- 2017 Intel Skylake and High End DeskTop (HEDT)
        # -DGGML_AVX512_VBMI -- 2018 Intel Cannon Lake
        # -DGGML_AVX512_VNNI -- 2021 Intel Alder Lake

        COMMON_CPU_DEFS="-DBUILD_SHARED_LIBS=on -DCMAKE_POSITION_INDEPENDENT_CODE=on -DGGML_NATIVE=off -DGGML_OPENMP=off"
        if [ -z "${OLLAMA_CPU_TARGET}" -o "${OLLAMA_CPU_TARGET}" = "cpu" ]; then
            #
            # CPU first for the default library, set up as lowest common denominator for maximum compatibility (including Rosetta)
            #
            init_vars
            CMAKE_DEFS="${COMMON_CPU_DEFS} -DGGML_AVX=off -DGGML_AVX2=off -DGGML_AVX512=off -DGGML_FMA=off -DGGML_F16C=off ${CMAKE_DEFS}"
            RUNNER=cpu
            BUILD_DIR="../build/linux/${GOARCH}/${RUNNER}"
            echo "Building LCD CPU"
            build
            install
            dist
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
                CMAKE_DEFS="${COMMON_CPU_DEFS} -DGGML_AVX=on -DGGML_AVX2=off -DGGML_AVX512=off -DGGML_FMA=off -DGGML_F16C=off ${CMAKE_DEFS}"
                RUNNER=cpu_avx
                BUILD_DIR="../build/linux/${GOARCH}/${RUNNER}"
                echo "Building AVX CPU"
                build
                install
                dist
                compress
            fi

            if [ -z "${OLLAMA_CPU_TARGET}" -o "${OLLAMA_CPU_TARGET}" = "cpu_avx2" ]; then
                #
                # ~2013 CPU Dynamic library
                # Approximately 10% faster than AVX on same CPU
                #
                init_vars
                CMAKE_DEFS="${COMMON_CPU_DEFS} -DGGML_AVX=on -DGGML_AVX2=on -DGGML_AVX512=off -DGGML_FMA=on -DGGML_F16C=on ${CMAKE_DEFS}"
                RUNNER=cpu_avx2
                BUILD_DIR="../build/linux/${GOARCH}/${RUNNER}"
                echo "Building AVX2 CPU"
                build
                install
                dist
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
    if [ -n "${CUDA_MAJOR}" -a -z "${CUDA_VARIANT}" ]; then
        CUDA_VARIANT=_v${CUDA_MAJOR}
    fi
    if [ "${ARCH}" == "arm64" ]; then
        echo "ARM CPU detected - disabling unsupported AVX instructions"

        # ARM-based CPUs such as M1 and Tegra do not support AVX extensions.
        #
        # CUDA compute < 6.0 lacks proper FP16 support on ARM.
        # Disabling has minimal performance effect while maintaining compatibility.
        ARM64_DEFS="-DGGML_AVX=off -DGGML_AVX2=off -DGGML_AVX512=off -DGGML_CUDA_F16=off"
    fi
    # Users building from source can tune the exact flags we pass to cmake for configuring llama.cpp
    if [ -n "${OLLAMA_CUSTOM_CUDA_DEFS}" ]; then
        echo "OLLAMA_CUSTOM_CUDA_DEFS=\"${OLLAMA_CUSTOM_CUDA_DEFS}\""
        CMAKE_CUDA_DEFS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} ${OLLAMA_CUSTOM_CUDA_DEFS}"
        echo "Building custom CUDA GPU"
    else
        CMAKE_CUDA_DEFS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}"
    fi
    export CUDAFLAGS="-t8"
    CMAKE_DEFS="${COMMON_CMAKE_DEFS} ${CMAKE_DEFS} ${ARM64_DEFS} ${CMAKE_CUDA_DEFS} -DGGML_STATIC=off"
    RUNNER=cuda${CUDA_VARIANT}
    BUILD_DIR="../build/linux/${GOARCH}/${RUNNER}"
    export LLAMA_SERVER_LDFLAGS="-L${CUDA_LIB_DIR} -lcudart -lcublas -lcublasLt -lcuda"
    CUDA_DIST_DIR="${CUDA_DIST_DIR:-${DIST_BASE}/lib/ollama}"
    build
    install
    dist
    echo "Installing CUDA dependencies in ${CUDA_DIST_DIR}"
    mkdir -p "${CUDA_DIST_DIR}"
    for lib in ${CUDA_LIB_DIR}/libcudart.so* ${CUDA_LIB_DIR}/libcublas.so* ${CUDA_LIB_DIR}/libcublasLt.so* ; do
        cp -a "${lib}" "${CUDA_DIST_DIR}"
    done
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
    CMAKE_DEFS="${COMMON_CMAKE_DEFS} ${CMAKE_DEFS} -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL=ON -DGGML_SYCL_F16=OFF"
    RUNNER=oneapi
    BUILD_DIR="../build/linux/${GOARCH}/${RUNNER}"
    ONEAPI_DIST_DIR="${DIST_BASE}/lib/ollama"
    export LLAMA_SERVER_LDFLAGS="-fsycl -lOpenCL -lmkl_core -lmkl_sycl_blas -lmkl_intel_ilp64 -lmkl_tbb_thread -ltbb"
    DEBUG_FLAGS="" # icx compiles with -O0 if we pass -g, so we must remove it
    build

    # copy oneAPI dependencies
    mkdir -p "${ONEAPI_DIST_DIR}"
    for dep in $(ldd "${BUILD_DIR}/bin/ollama_llama_server" | grep "=>" | cut -f2 -d= | cut -f2 -d' ' | grep -e sycl -e mkl -e tbb); do
        cp -a "${dep}" "${ONEAPI_DIST_DIR}"
    done
    cp "${ONEAPI_ROOT}/compiler/latest/lib/libOpenCL.so" "${ONEAPI_DIST_DIR}"
    cp "${ONEAPI_ROOT}/compiler/latest/lib/libimf.so" "${ONEAPI_DIST_DIR}"
    cp "${ONEAPI_ROOT}/compiler/latest/lib/libintlc.so.5" "${ONEAPI_DIST_DIR}"
    cp "${ONEAPI_ROOT}/compiler/latest/lib/libirng.so" "${ONEAPI_DIST_DIR}"
    cp "${ONEAPI_ROOT}/compiler/latest/lib/libpi_level_zero.so" "${ONEAPI_DIST_DIR}"
    cp "${ONEAPI_ROOT}/compiler/latest/lib/libsvml.so" "${ONEAPI_DIST_DIR}"
    cp "${ONEAPI_ROOT}/compiler/latest/lib/libur_loader.so.0" "${ONEAPI_DIST_DIR}"
    install
    dist
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
    CMAKE_DEFS="${COMMON_CMAKE_DEFS} ${CMAKE_DEFS} -DGGML_HIPBLAS=on -DCMAKE_C_COMPILER=$ROCM_PATH/llvm/bin/clang -DCMAKE_CXX_COMPILER=$ROCM_PATH/llvm/bin/clang++ -DAMDGPU_TARGETS=$(amdGPUs) -DGPU_TARGETS=$(amdGPUs)"
    # Users building from source can tune the exact flags we pass to cmake for configuring llama.cpp
    if [ -n "${OLLAMA_CUSTOM_ROCM_DEFS}" ]; then
        echo "OLLAMA_CUSTOM_ROCM_DEFS=\"${OLLAMA_CUSTOM_ROCM_DEFS}\""
        CMAKE_DEFS="${CMAKE_DEFS} ${OLLAMA_CUSTOM_ROCM_DEFS}"
        echo "Building custom ROCM GPU"
    fi
    RUNNER=rocm${ROCM_VARIANT}
    BUILD_DIR="../build/linux/${GOARCH}/${RUNNER}"
    # ROCm dependencies are too large to fit into a unified bundle
    ROCM_DIST_DIR="${DIST_BASE}/../linux-${GOARCH}-rocm/lib/ollama"
    # TODO figure out how to disable runpath (rpath)
    # export CMAKE_HIP_FLAGS="-fno-rtlib-add-rpath" # doesn't work
    export LLAMA_SERVER_LDFLAGS="-L${ROCM_PATH}/lib -L/opt/amdgpu/lib/x86_64-linux-gnu/ -lhipblas -lrocblas -lamdhip64 -lrocsolver -lamd_comgr -lhsa-runtime64 -lrocsparse -ldrm -ldrm_amdgpu"
    build

    # copy the ROCM dependencies
    mkdir -p "${ROCM_DIST_DIR}"
    for dep in $(ldd "${BUILD_DIR}/bin/ollama_llama_server" | grep "=>" | cut -f2 -d= | cut -f2 -d' ' | grep -v "${GOARCH}/rocm${ROCM_VARIANT}" | grep -e rocm -e amdgpu -e libtinfo -e libnuma -e libelf ); do
        cp -a "${dep}"* "${ROCM_DIST_DIR}"
        if [ $(readlink -f "${dep}") != "${dep}" ] ; then
            cp $(readlink -f "${dep}") "${ROCM_DIST_DIR}"
        fi
    done
    install
    dist
    compress
fi

cleanup
wait_for_compress
echo "go generate completed.  LLM runners: $(cd ${PAYLOAD_BASE}; echo *)"
