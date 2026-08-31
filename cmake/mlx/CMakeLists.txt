cmake_minimum_required(VERSION 3.24)

project(OllamaMLX C CXX)

include(CheckLanguage)
include(GNUInstallDirs)

find_package(Threads REQUIRED)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT DEFINED BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

if(APPLE)
    set(CMAKE_BUILD_RPATH "@loader_path")
    set(CMAKE_INSTALL_RPATH "@loader_path")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
elseif(UNIX)
    set(CMAKE_BUILD_RPATH "$ORIGIN")
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
endif()

if(NOT DEFINED OLLAMA_SOURCE_DIR OR "${OLLAMA_SOURCE_DIR}" STREQUAL "")
    get_filename_component(OLLAMA_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
endif()
get_filename_component(OLLAMA_SOURCE_DIR "${OLLAMA_SOURCE_DIR}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(OLLAMA_SOURCE_DIR "${OLLAMA_SOURCE_DIR}" CACHE PATH "Ollama repository root")

set(OLLAMA_LIB_DIR "lib/ollama" CACHE STRING "Install destination for Ollama runtime payloads")
set(OLLAMA_RUNNER_DIR "" CACHE STRING "Ollama runtime payload subdirectory")
set(OLLAMA_BUILD_DIR ${CMAKE_BINARY_DIR}/lib/ollama)
set(OLLAMA_INSTALL_DIR ${OLLAMA_LIB_DIR}/${OLLAMA_RUNNER_DIR})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY         ${OLLAMA_BUILD_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${OLLAMA_BUILD_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${OLLAMA_BUILD_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY         ${OLLAMA_BUILD_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG   ${OLLAMA_BUILD_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${OLLAMA_BUILD_DIR})

if(MLX_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES)
    check_language(CUDA)
endif()

option(OLLAMA_MLX_GENERATE_WRAPPERS "Regenerate MLX Go wrappers" OFF)

message(STATUS "Setting up MLX (this takes a while...)")
foreach(_cudnn_var CUDNN_INCLUDE_PATH CUDNN_LIBRARY_PATH)
    if((NOT DEFINED ${_cudnn_var} OR "${${_cudnn_var}}" STREQUAL "") AND DEFINED ENV{${_cudnn_var}})
        set(${_cudnn_var} "$ENV{${_cudnn_var}}" CACHE PATH "")
    endif()
endforeach()
add_subdirectory(${OLLAMA_SOURCE_DIR}/x/mlxrunner/mlx ${CMAKE_BINARY_DIR}/x/mlxrunner/mlx)

include(FetchContent)
set(XGRAMMAR_VERSION v0.2.5)
FetchContent_Declare(xgrammar
    GIT_REPOSITORY "https://github.com/mlc-ai/xgrammar.git"
    GIT_TAG ${XGRAMMAR_VERSION}
    GIT_SHALLOW TRUE
    GIT_SUBMODULES 3rdparty/dlpack
    # Do not add XGrammar's Python-oriented CMake project.
    SOURCE_SUBDIR cmake/ollama)
FetchContent_MakeAvailable(xgrammar)

file(GLOB_RECURSE XGRAMMAR_SOURCES CONFIGURE_DEPENDS "${xgrammar_SOURCE_DIR}/cpp/*.cc")
list(FILTER XGRAMMAR_SOURCES EXCLUDE REGEX "/cpp/tvm_ffi/.*\\.cc$")
add_library(xgrammar STATIC ${XGRAMMAR_SOURCES})
set_target_properties(xgrammar PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
target_include_directories(xgrammar PUBLIC "${xgrammar_SOURCE_DIR}/include")
target_include_directories(xgrammar SYSTEM PUBLIC
    "${xgrammar_SOURCE_DIR}/3rdparty/picojson"
    "${xgrammar_SOURCE_DIR}/3rdparty/dlpack/include")
target_compile_definitions(xgrammar PUBLIC
    XGRAMMAR_ENABLE_CPPTRACE=0
    XGRAMMAR_ENABLE_INTERNAL_CHECK=0)

add_library(ollama_xgrammar SHARED
    "${OLLAMA_SOURCE_DIR}/x/mlxrunner/xgrammar/native/xgrammar.cpp")
target_include_directories(ollama_xgrammar PRIVATE
    "${OLLAMA_SOURCE_DIR}/x/mlxrunner/xgrammar/native")
target_compile_definitions(ollama_xgrammar PRIVATE
    OLLAMA_XGRAMMAR_BUILD=1
    OLLAMA_XGRAMMAR_VERSION="${XGRAMMAR_VERSION}")
# Export only the ollama_xgrammar_* API; xgrammar's own C++ symbols stay hidden.
set_target_properties(ollama_xgrammar PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
target_link_libraries(ollama_xgrammar PRIVATE xgrammar Threads::Threads)

# Find CUDA toolkit if MLX is built with CUDA support.
find_package(CUDAToolkit)

# Build list of directories for runtime dependency resolution.
set(MLX_RUNTIME_DIRS ${CUDAToolkit_BIN_DIR} ${CUDAToolkit_BIN_DIR}/x64 ${CUDAToolkit_LIBRARY_DIR})
# Add cuDNN bin paths for DLLs (Windows MLX CUDA builds).
# CUDNN_ROOT_DIR is the standard CMake variable for cuDNN location.
if(CUDNN_ROOT_DIR)
    set(_cudnn_root "${CUDNN_ROOT_DIR}")
elseif(DEFINED ENV{CUDNN_ROOT_DIR})
    set(_cudnn_root "$ENV{CUDNN_ROOT_DIR}")
endif()
if(_cudnn_root)
    file(TO_CMAKE_PATH "${_cudnn_root}" _cudnn_root)
    # cuDNN 9.x has versioned subdirectories under bin/ (e.g., bin/13.0/).
    file(GLOB CUDNN_BIN_SUBDIRS "${_cudnn_root}/bin/*")
    list(APPEND MLX_RUNTIME_DIRS ${CUDNN_BIN_SUBDIRS})
endif()
# Add build output directory and MLX dependency build directories.
list(APPEND MLX_RUNTIME_DIRS ${OLLAMA_BUILD_DIR})
# OpenBLAS DLL location (pre-built zip extracts into openblas-src/bin/).
list(APPEND MLX_RUNTIME_DIRS ${CMAKE_BINARY_DIR}/_deps/openblas-src/bin)
# NCCL: on Linux, if real NCCL is found, cmake bundles libnccl.so via the
# regex below. If NCCL is not found, MLX links a static stub (OBJECT lib)
# so there is no runtime dependency. This path covers the stub build dir
# for windows so we include the DLL in our dependencies.
list(APPEND MLX_RUNTIME_DIRS ${CMAKE_BINARY_DIR}/_deps/mlx-build/mlx/distributed/nccl/nccl_stub-prefix/src/nccl_stub-build/Release)

# Base regexes for runtime dependencies (cross-platform).
set(MLX_INCLUDE_REGEXES cublas cublasLt cudart cufft nvrtc nvrtc-builtins cudnn nccl openblas gfortran)
# On Windows, also include dl.dll (dlfcn-win32 POSIX emulation layer).
if(WIN32)
    list(APPEND MLX_INCLUDE_REGEXES "^dl\\.dll$")
endif()

# Keep mlx/mlxc targets separate from runtime dependencies so --strip only
# applies to the binaries we build, not vendor DLLs/libs.
install(TARGETS mlx mlxc ollama_xgrammar
    RUNTIME_DEPENDENCY_SET mlx_runtime_deps
    RUNTIME DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT MLX
    LIBRARY DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT MLX
    FRAMEWORK DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT MLX
)
install(FILES
    "${xgrammar_SOURCE_DIR}/LICENSE"
    DESTINATION ${OLLAMA_INSTALL_DIR}
    RENAME XGRAMMAR_LICENSE
    COMPONENT MLX)
install(FILES
    "${xgrammar_SOURCE_DIR}/NOTICE"
    DESTINATION ${OLLAMA_INSTALL_DIR}
    RENAME XGRAMMAR_NOTICE
    COMPONENT MLX)
install(FILES
    "${xgrammar_SOURCE_DIR}/3rdparty/dlpack/LICENSE"
    DESTINATION ${OLLAMA_INSTALL_DIR}
    RENAME DLPACK_LICENSE
    COMPONENT MLX)
file(READ "${xgrammar_SOURCE_DIR}/3rdparty/picojson/picojson.h" _picojson_header LIMIT 4096)
string(FIND "${_picojson_header}" "*/" _picojson_license_end)
if(_picojson_license_end EQUAL -1)
    message(FATAL_ERROR "picojson license header not found")
endif()
math(EXPR _picojson_license_end "${_picojson_license_end} + 2")
string(SUBSTRING "${_picojson_header}" 0 ${_picojson_license_end} _picojson_license)
file(WRITE "${CMAKE_BINARY_DIR}/PICOJSON_LICENSE" "${_picojson_license}\n")
install(FILES
    "${CMAKE_BINARY_DIR}/PICOJSON_LICENSE"
    DESTINATION ${OLLAMA_INSTALL_DIR}
    COMPONENT MLX)
install(RUNTIME_DEPENDENCY_SET mlx_runtime_deps
    DIRECTORIES ${MLX_RUNTIME_DIRS}
    PRE_INCLUDE_REGEXES ${MLX_INCLUDE_REGEXES}
    PRE_EXCLUDE_REGEXES ".*"
    RUNTIME DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT MLX_VENDOR
    LIBRARY DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT MLX_VENDOR
)

get_target_property(_MLX_LINK_LIBRARIES mlx LINK_LIBRARIES)
if(TARGET jaccl AND "jaccl" IN_LIST _MLX_LINK_LIBRARIES)
    install(TARGETS jaccl
        RUNTIME DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT MLX
        LIBRARY DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT MLX
        FRAMEWORK DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT MLX
    )
endif()

# Install the Metal library for macOS arm64 (must be colocated with the binary).
# Metal backend is only built for arm64, not x86_64.
if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    install(FILES ${CMAKE_BINARY_DIR}/_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib
        DESTINATION ${OLLAMA_INSTALL_DIR}
        COMPONENT MLX)
endif()

# Install headers for NVRTC JIT compilation at runtime.
# MLX's own install rules use the default component so they get skipped by
# --component MLX. Headers are installed alongside libmlx in OLLAMA_INSTALL_DIR.
#
# Layout:
#   ${OLLAMA_INSTALL_DIR}/include/cccl/            - CCCL headers
#   ${OLLAMA_INSTALL_DIR}/include/{cute,cutlass}/  - CUTLASS/CUTE headers
#   ${OLLAMA_INSTALL_DIR}/include/                 - CUDA runtime/core headers
#
# MLX's jit_module.cpp resolves JIT support headers from the backend-local
# include directory. On Linux it also probes current_binary_dir().parent_path()
# / "include", so we create a symlink from lib/ollama/include to the backend
# include directory for archive packaging.
# This will need refinement if we add multiple CUDA versions for MLX in the future.
# CUDA runtime headers are found via CUDA_PATH env var (set by mlxrunner).
set(_mlx_jit_cccl_include_dir "")
if(CUDAToolkit_FOUND)
    foreach(_dir ${CUDAToolkit_INCLUDE_DIRS})
        if(EXISTS "${_dir}/cccl/cuda/std")
            set(_mlx_jit_cccl_include_dir "${_dir}/cccl")
            break()
        endif()
    endforeach()
endif()
if(NOT _mlx_jit_cccl_include_dir AND EXISTS ${CMAKE_BINARY_DIR}/_deps/cccl-src/include/cuda)
    set(_mlx_jit_cccl_include_dir "${CMAKE_BINARY_DIR}/_deps/cccl-src/include")
endif()
if(_mlx_jit_cccl_include_dir)
    foreach(_cccl_dir cuda nv cub thrust)
        if(EXISTS "${_mlx_jit_cccl_include_dir}/${_cccl_dir}")
            install(DIRECTORY "${_mlx_jit_cccl_include_dir}/${_cccl_dir}"
                DESTINATION ${OLLAMA_INSTALL_DIR}/include/cccl
                COMPONENT MLX)
        endif()
    endforeach()
endif()
if(EXISTS ${CMAKE_BINARY_DIR}/_deps/cutlass-src/include/cute)
    install(DIRECTORY ${CMAKE_BINARY_DIR}/_deps/cutlass-src/include/cute
        DESTINATION ${OLLAMA_INSTALL_DIR}/include
        COMPONENT MLX)
    install(DIRECTORY ${CMAKE_BINARY_DIR}/_deps/cutlass-src/include/cutlass
        DESTINATION ${OLLAMA_INSTALL_DIR}/include
        COMPONENT MLX)
endif()

# Install CUDA runtime/core headers needed by MLX JIT kernels.
# NVIDIA's NVRTC bundled-header model is CUDA Runtime + CCCL, not the entire
# toolkit include tree. Keep CCCL coherent above, include CUTLASS/CUTE above,
# and avoid shipping unrelated SDK headers such as NPP, CUPTI, cuRAND, NVML,
# cuBLAS, cuSPARSE, and cuSOLVER.
# The Go mlxrunner sets CUDA_PATH to OLLAMA_INSTALL_DIR so MLX finds them at
# $CUDA_PATH/include via NVRTC --include-path.
if(CUDAToolkit_FOUND)
    # CUDAToolkit_INCLUDE_DIRS may be a semicolon-separated list
    # (e.g. ".../include;.../include/cccl"). Find the entry that
    # contains the CUDA runtime headers we need.
    set(_cuda_inc "")
    foreach(_dir ${CUDAToolkit_INCLUDE_DIRS})
        if(EXISTS "${_dir}/cuda_runtime_api.h")
            set(_cuda_inc "${_dir}")
            break()
        endif()
    endforeach()
    if(NOT _cuda_inc)
        message(WARNING "Could not find cuda_runtime_api.h in CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
    else()
        set(_dst "${OLLAMA_INSTALL_DIR}/include")

        set(_mlx_jit_cuda_headers
            builtin_types.h
            channel_descriptor.h
            common_functions.h
            cooperative_groups.h
            cuComplex.h
            cuda.h
            cudaTypedefs.h
            cuda_awbarrier.h
            cuda_awbarrier_helpers.h
            cuda_awbarrier_primitives.h
            cuda_bf16.h
            cuda_bf16.hpp
            cuda_device_runtime_api.h
            cuda_fp4.h
            cuda_fp4.hpp
            cuda_fp6.h
            cuda_fp6.hpp
            cuda_fp8.h
            cuda_fp8.hpp
            cuda_fp16.h
            cuda_fp16.hpp
            cuda_occupancy.h
            cuda_pipeline.h
            cuda_pipeline_helpers.h
            cuda_pipeline_primitives.h
            cuda_runtime.h
            cuda_runtime_api.h
            cuda_stdint.h
            cudart_platform.h
            device_atomic_functions.h
            device_atomic_functions.hpp
            device_double_functions.h
            device_functions.h
            device_launch_parameters.h
            device_types.h
            driver_functions.h
            driver_types.h
            fatbinary_section.h
            host_config.h
            host_defines.h
            library_types.h
            math_constants.h
            math_functions.h
            mma.h
            nvrtc_device_runtime.h
            sm_20_atomic_functions.h
            sm_20_atomic_functions.hpp
            sm_20_intrinsics.h
            sm_20_intrinsics.hpp
            sm_30_intrinsics.h
            sm_30_intrinsics.hpp
            sm_32_atomic_functions.h
            sm_32_atomic_functions.hpp
            sm_32_intrinsics.h
            sm_32_intrinsics.hpp
            sm_35_atomic_functions.h
            sm_35_intrinsics.h
            sm_60_atomic_functions.h
            sm_60_atomic_functions.hpp
            sm_61_intrinsics.h
            sm_61_intrinsics.hpp
            surface_indirect_functions.h
            surface_types.h
            target
            texture_indirect_functions.h
            texture_types.h
            vector_functions.h
            vector_functions.hpp
            vector_types.h)
        set(_mlx_jit_cuda_header_paths "")
        foreach(_header IN LISTS _mlx_jit_cuda_headers)
            if(EXISTS "${_cuda_inc}/${_header}")
                list(APPEND _mlx_jit_cuda_header_paths "${_cuda_inc}/${_header}")
            endif()
        endforeach()
        if(_mlx_jit_cuda_header_paths)
            install(FILES ${_mlx_jit_cuda_header_paths}
                DESTINATION ${_dst}
                COMPONENT MLX)
        endif()

        foreach(_runtime_dir cooperative_groups crt)
            if(EXISTS "${_cuda_inc}/${_runtime_dir}")
                install(DIRECTORY "${_cuda_inc}/${_runtime_dir}"
                    DESTINATION ${_dst}
                    COMPONENT MLX)
            endif()
        endforeach()

        if(NOT WIN32 AND NOT APPLE)
            install(CODE "
                set(_link \"${CMAKE_INSTALL_PREFIX}/${OLLAMA_LIB_DIR}/include\")
                set(_target \"${OLLAMA_RUNNER_DIR}/include\")
                if(NOT EXISTS \${_link})
                    execute_process(COMMAND \${CMAKE_COMMAND} -E create_symlink \${_target} \${_link})
                endif()
            " COMPONENT MLX)
        endif()
    endif()
endif()

# On Windows, explicitly install dl.dll (dlfcn-win32 POSIX dlopen emulation).
# RUNTIME_DEPENDENCIES auto-excludes it via POST_EXCLUDE_FILES_STRICT because
# dlfcn-win32 is a known CMake target with its own install rules (which install
# to the wrong destination). We must install it explicitly here.
if(WIN32 AND TARGET dl)
    install(TARGETS dl
        RUNTIME DESTINATION ${OLLAMA_INSTALL_DIR}
        LIBRARY DESTINATION ${OLLAMA_INSTALL_DIR}
        COMPONENT MLX)
endif()

# Manually install CUDA runtime libraries that MLX loads via dlopen
# (not detected by RUNTIME_DEPENDENCIES since they aren't link-time deps).
if(CUDAToolkit_FOUND)
    file(GLOB MLX_CUDA_LIBS
        "${CUDAToolkit_LIBRARY_DIR}/libcudart.so*"
        "${CUDAToolkit_LIBRARY_DIR}/libcublas.so*"
        "${CUDAToolkit_LIBRARY_DIR}/libcublasLt.so*"
        "${CUDAToolkit_LIBRARY_DIR}/libnvrtc.so*"
        "${CUDAToolkit_LIBRARY_DIR}/libnvrtc-builtins.so*"
        "${CUDAToolkit_LIBRARY_DIR}/libcufft.so*"
        "${CUDAToolkit_LIBRARY_DIR}/libcudnn*.so*")
    if(WIN32)
        file(GLOB MLX_CUDA_DLLS
            "${CUDAToolkit_BIN_DIR}/nvrtc-builtins64_*.dll"
            "${CUDAToolkit_BIN_DIR}/x64/nvrtc-builtins64_*.dll")
        list(APPEND MLX_CUDA_LIBS ${MLX_CUDA_DLLS})
    endif()
    find_library(MLX_CUDNN_LIBRARY NAMES cudnn HINTS "$ENV{CUDNN_LIBRARY_PATH}")
    if(MLX_CUDNN_LIBRARY)
        get_filename_component(MLX_CUDNN_LIBRARY_DIR "${MLX_CUDNN_LIBRARY}" DIRECTORY)
        file(GLOB MLX_CUDNN_LIBS "${MLX_CUDNN_LIBRARY_DIR}/libcudnn*.so*")
        list(APPEND MLX_CUDA_LIBS ${MLX_CUDNN_LIBS})
    endif()
    if(WIN32 AND _cudnn_root)
        file(GLOB MLX_CUDNN_DLLS
            "${_cudnn_root}/bin/${CUDAToolkit_VERSION_MAJOR}.0/cudnn*.dll"
            "${_cudnn_root}/bin/x64/cudnn*.dll")
        list(APPEND MLX_CUDA_LIBS ${MLX_CUDNN_DLLS})
    endif()
    if(MLX_CUDA_LIBS)
        install(FILES ${MLX_CUDA_LIBS}
            DESTINATION ${OLLAMA_INSTALL_DIR}
            COMPONENT MLX_VENDOR)
    endif()
endif()
