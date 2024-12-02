cmake_minimum_required(VERSION 3.14) # for add_link_options and implicit target directories.
project("ggml" C CXX)
include(CheckIncludeFileCXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(GGML_STANDALONE ON)

    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

    # configure project version
    # TODO
else()
    set(GGML_STANDALONE OFF)
endif()

if (EMSCRIPTEN)
    set(BUILD_SHARED_LIBS_DEFAULT OFF)

    option(GGML_WASM_SINGLE_FILE "ggml: embed WASM inside the generated ggml.js" ON)
else()
    if (MINGW)
        set(BUILD_SHARED_LIBS_DEFAULT OFF)
    else()
        set(BUILD_SHARED_LIBS_DEFAULT ON)
    endif()
endif()

# remove the lib prefix on win32 mingw
if (WIN32)
    set(CMAKE_STATIC_LIBRARY_PREFIX "")
    set(CMAKE_SHARED_LIBRARY_PREFIX "")
    set(CMAKE_SHARED_MODULE_PREFIX  "")
endif()

option(BUILD_SHARED_LIBS "ggml: build shared libraries" ${BUILD_SHARED_LIBS_DEFAULT})
option(GGML_BACKEND_DL   "ggml: build backends as dynamic libraries (requires BUILD_SHARED_LIBS)" OFF)

#
# option list
#

# TODO: mark all options as advanced when not GGML_STANDALONE

if (APPLE)
    set(GGML_METAL_DEFAULT ON)
    set(GGML_BLAS_DEFAULT ON)
    set(GGML_BLAS_VENDOR_DEFAULT "Apple")
else()
    set(GGML_METAL_DEFAULT OFF)
    set(GGML_BLAS_DEFAULT OFF)
    set(GGML_BLAS_VENDOR_DEFAULT "Generic")
endif()

if (CMAKE_CROSSCOMPILING)
    set(GGML_NATIVE_DEFAULT OFF)
else()
    set(GGML_NATIVE_DEFAULT ON)
endif()

# defaults
if (NOT GGML_LLAMAFILE_DEFAULT)
    set(GGML_LLAMAFILE_DEFAULT OFF)
endif()

if (NOT GGML_CUDA_GRAPHS_DEFAULT)
    set(GGML_CUDA_GRAPHS_DEFAULT OFF)
endif()

# general
option(GGML_STATIC "ggml: static link libraries"                     OFF)
option(GGML_NATIVE "ggml: optimize the build for the current system" ${GGML_NATIVE_DEFAULT})
option(GGML_LTO    "ggml: enable link time optimization"             OFF)
option(GGML_CCACHE "ggml: use ccache if available"                   ON)

# debug
option(GGML_ALL_WARNINGS           "ggml: enable all compiler warnings"                   ON)
option(GGML_ALL_WARNINGS_3RD_PARTY "ggml: enable all compiler warnings in 3rd party libs" OFF)
option(GGML_GPROF                  "ggml: enable gprof"                                   OFF)

# build
option(GGML_FATAL_WARNINGS    "ggml: enable -Werror flag"    OFF)

# sanitizers
option(GGML_SANITIZE_THREAD    "ggml: enable thread sanitizer"    OFF)
option(GGML_SANITIZE_ADDRESS   "ggml: enable address sanitizer"   OFF)
option(GGML_SANITIZE_UNDEFINED "ggml: enable undefined sanitizer" OFF)

# instruction set specific
if (GGML_NATIVE OR NOT GGML_NATIVE_DEFAULT)
    set(INS_ENB OFF)
else()
    set(INS_ENB ON)
endif()

option(GGML_CPU_HBM          "ggml: use memkind for CPU HBM" OFF)
option(GGML_CPU_AARCH64      "ggml: use runtime weight conversion of Q4_0 to Q4_X_X" ON)
option(GGML_AVX              "ggml: enable AVX"              ${INS_ENB})
option(GGML_AVX_VNNI         "ggml: enable AVX-VNNI"         OFF)
option(GGML_AVX2             "ggml: enable AVX2"             ${INS_ENB})
option(GGML_AVX512           "ggml: enable AVX512F"          OFF)
option(GGML_AVX512_VBMI      "ggml: enable AVX512-VBMI"      OFF)
option(GGML_AVX512_VNNI      "ggml: enable AVX512-VNNI"      OFF)
option(GGML_AVX512_BF16      "ggml: enable AVX512-BF16"      OFF)
if (NOT MSVC)
    # in MSVC F16C and FMA is implied with AVX2/AVX512
    option(GGML_FMA          "ggml: enable FMA"              ${INS_ENB})
    option(GGML_F16C         "ggml: enable F16C"             ${INS_ENB})
    # MSVC does not seem to support AMX
    option(GGML_AMX_TILE     "ggml: enable AMX-TILE"         OFF)
    option(GGML_AMX_INT8     "ggml: enable AMX-INT8"         OFF)
    option(GGML_AMX_BF16     "ggml: enable AMX-BF16"         OFF)
endif()
option(GGML_LASX             "ggml: enable lasx"             ON)
option(GGML_LSX              "ggml: enable lsx"              ON)
option(GGML_RVV              "ggml: enable rvv"              ON)

option(GGML_CPU_ALL_VARIANTS "ggml: build all variants of the CPU backend (requires GGML_BACKEND_DL)" OFF)
set(GGML_CPU_ARM_ARCH "" CACHE STRING "ggml: CPU architecture for ARM")


if (WIN32)
    set(GGML_WIN_VER "0x602" CACHE STRING   "ggml: Windows version")
endif()

# ggml core
set(GGML_SCHED_MAX_COPIES  "4" CACHE STRING "ggml: max input copies for pipeline parallelism")
option(GGML_CPU                             "ggml: enable CPU backend"                        ON)

# 3rd party libs / backends
option(GGML_ACCELERATE                      "ggml: enable Accelerate framework"               ON)
option(GGML_BLAS                            "ggml: use BLAS"                                  ${GGML_BLAS_DEFAULT})
set(GGML_BLAS_VENDOR ${GGML_BLAS_VENDOR_DEFAULT} CACHE STRING
                                            "ggml: BLAS library vendor")
option(GGML_LLAMAFILE                       "ggml: use LLAMAFILE"                             ${GGML_LLAMAFILE_DEFAULT})

option(GGML_CUDA                            "ggml: use CUDA"                                  OFF)
option(GGML_MUSA                            "ggml: use MUSA"                                  OFF)
option(GGML_CUDA_FORCE_MMQ                  "ggml: use mmq kernels instead of cuBLAS"         OFF)
option(GGML_CUDA_FORCE_CUBLAS               "ggml: always use cuBLAS instead of mmq kernels"  OFF)
option(GGML_CUDA_F16                        "ggml: use 16 bit floats for some calculations"   OFF)
set   (GGML_CUDA_PEER_MAX_BATCH_SIZE "128" CACHE STRING
                                            "ggml: max. batch size for using peer access")
option(GGML_CUDA_NO_PEER_COPY               "ggml: do not use peer to peer copies"            OFF)
option(GGML_CUDA_NO_VMM                     "ggml: do not try to use CUDA VMM"                OFF)
option(GGML_CUDA_FA_ALL_QUANTS              "ggml: compile all quants for FlashAttention"     OFF)
option(GGML_CUDA_GRAPHS                     "ggml: use CUDA graphs (llama.cpp only)"          ${GGML_CUDA_GRAPHS_DEFAULT})

option(GGML_HIP                             "ggml: use HIP"                                   OFF)
option(GGML_HIP_UMA                         "ggml: use HIP unified memory architecture"       OFF)
option(GGML_VULKAN                          "ggml: use Vulkan"                                OFF)
option(GGML_VULKAN_CHECK_RESULTS            "ggml: run Vulkan op checks"                      OFF)
option(GGML_VULKAN_DEBUG                    "ggml: enable Vulkan debug output"                OFF)
option(GGML_VULKAN_MEMORY_DEBUG             "ggml: enable Vulkan memory debug output"         OFF)
option(GGML_VULKAN_SHADER_DEBUG_INFO        "ggml: enable Vulkan shader debug info"           OFF)
option(GGML_VULKAN_PERF                     "ggml: enable Vulkan perf output"                 OFF)
option(GGML_VULKAN_VALIDATE                 "ggml: enable Vulkan validation"                  OFF)
option(GGML_VULKAN_RUN_TESTS                "ggml: run Vulkan tests"                          OFF)
option(GGML_KOMPUTE                         "ggml: use Kompute"                               OFF)
option(GGML_METAL                           "ggml: use Metal"                                 ${GGML_METAL_DEFAULT})
option(GGML_METAL_USE_BF16                  "ggml: use bfloat if available"                   OFF)
option(GGML_METAL_NDEBUG                    "ggml: disable Metal debugging"                   OFF)
option(GGML_METAL_SHADER_DEBUG              "ggml: compile Metal with -fno-fast-math"         OFF)
option(GGML_METAL_EMBED_LIBRARY             "ggml: embed Metal library"                       ${GGML_METAL})
set   (GGML_METAL_MACOSX_VERSION_MIN "" CACHE STRING
                                            "ggml: metal minimum macOS version")
set   (GGML_METAL_STD "" CACHE STRING       "ggml: metal standard version (-std flag)")
option(GGML_OPENMP                          "ggml: use OpenMP"                                ON)
option(GGML_RPC                             "ggml: use RPC"                                   OFF)
option(GGML_SYCL                            "ggml: use SYCL"                                  OFF)
option(GGML_SYCL_F16                        "ggml: use 16 bit floats for sycl calculations"   OFF)
set   (GGML_SYCL_TARGET "INTEL" CACHE STRING
                                            "ggml: sycl target device")
set   (GGML_SYCL_DEVICE_ARCH "" CACHE STRING
                                            "ggml: sycl device architecture")

option(GGML_OPENCL                          "ggml: use OpenCL"                                OFF)
option(GGML_OPENCL_PROFILING                "ggml: use OpenCL profiling (increases overhead)" OFF)
option(GGML_OPENCL_EMBED_KERNELS            "ggml: embed kernels"                             ON)
option(GGML_OPENCL_USE_ADRENO_KERNELS       "ggml: use optimized kernels for Adreno"          ON)

# extra artifacts
option(GGML_BUILD_TESTS    "ggml: build tests"    ${GGML_STANDALONE})
option(GGML_BUILD_EXAMPLES "ggml: build examples" ${GGML_STANDALONE})

#
# dependencies
#

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Threads REQUIRED)

#
# build the library
#

add_subdirectory(src)

#
# tests and examples
#

if (GGML_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()

if (GGML_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()

#
# install
#

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# all public headers
set(GGML_PUBLIC_HEADERS
    include/ggml.h
    include/ggml-cpu.h
    include/ggml-alloc.h
    include/ggml-backend.h
    include/ggml-blas.h
    include/ggml-cann.h
    include/ggml-cuda.h
    include/ggml-kompute.h
    include/ggml-opt.h
    include/ggml-metal.h
    include/ggml-rpc.h
    include/ggml-sycl.h
    include/ggml-vulkan.h)

set_target_properties(ggml PROPERTIES PUBLIC_HEADER "${GGML_PUBLIC_HEADERS}")
#if (GGML_METAL)
#    set_target_properties(ggml PROPERTIES RESOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/ggml-metal.metal")
#endif()
install(TARGETS ggml LIBRARY PUBLIC_HEADER)
install(TARGETS ggml-base LIBRARY)

if (GGML_STANDALONE)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ggml.pc.in
        ${CMAKE_CURRENT_BINARY_DIR}/ggml.pc
        @ONLY)

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ggml.pc
        DESTINATION share/pkgconfig)
endif()
