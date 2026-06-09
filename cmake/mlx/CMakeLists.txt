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
add_subdirectory(${OLLAMA_SOURCE_DIR}/x/imagegen/mlx ${CMAKE_BINARY_DIR}/x/imagegen/mlx)

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
install(TARGETS mlx mlxc
    RUNTIME_DEPENDENCY_SET mlx_runtime_deps
    RUNTIME DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT MLX
    LIBRARY DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT MLX
    FRAMEWORK DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT MLX
)
install(RUNTIME_DEPENDENCY_SET mlx_runtime_deps
    DIRECTORIES ${MLX_RUNTIME_DIRS}
    PRE_INCLUDE_REGEXES ${MLX_INCLUDE_REGEXES}
    PRE_EXCLUDE_REGEXES ".*"
    RUNTIME DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT MLX_VENDOR
    LIBRARY DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT MLX_VENDOR
)

if(TARGET jaccl)
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
#   ${OLLAMA_INSTALL_DIR}/include/cccl/{cuda,nv}/  - CCCL headers
#   ${OLLAMA_INSTALL_DIR}/include/*.h              - CUDA toolkit headers
#
# MLX's jit_module.cpp resolves CCCL via
#   current_binary_dir()[.parent_path()] / "include" / "cccl"
# On Linux, MLX's jit_module.cpp resolves CCCL via
#   current_binary_dir().parent_path() / "include" / "cccl", so we create a
#   symlink from lib/ollama/include -> ${OLLAMA_RUNNER_DIR}/include.
# This will need refinement if we add multiple CUDA versions for MLX in the future.
# CUDA runtime headers are found via CUDA_PATH env var (set by mlxrunner).
if(EXISTS ${CMAKE_BINARY_DIR}/_deps/cccl-src/include/cuda)
    install(DIRECTORY ${CMAKE_BINARY_DIR}/_deps/cccl-src/include/cuda
        DESTINATION ${OLLAMA_INSTALL_DIR}/include/cccl
        COMPONENT MLX)
    install(DIRECTORY ${CMAKE_BINARY_DIR}/_deps/cccl-src/include/nv
        DESTINATION ${OLLAMA_INSTALL_DIR}/include/cccl
        COMPONENT MLX)
endif()

# Install minimal CUDA toolkit headers needed by MLX JIT kernels.
# These are the transitive closure of includes from mlx/backend/cuda/device/*.cuh.
# The Go mlxrunner sets CUDA_PATH to OLLAMA_INSTALL_DIR so MLX finds them at
# $CUDA_PATH/include/*.h via NVRTC --include-path.
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
        set(_MLX_JIT_CUDA_HEADERS
            builtin_types.h
            cooperative_groups.h
            cuda_bf16.h
            cuda_bf16.hpp
            cuda_device_runtime_api.h
            cuda_fp16.h
            cuda_fp16.hpp
            cuda_fp8.h
            cuda_fp8.hpp
            cuda_runtime_api.h
            device_types.h
            driver_types.h
            math_constants.h
            surface_types.h
            texture_types.h
            vector_functions.h
            vector_functions.hpp
            vector_types.h
        )
        foreach(_hdr ${_MLX_JIT_CUDA_HEADERS})
            install(FILES "${_cuda_inc}/${_hdr}"
                DESTINATION ${_dst}
                COMPONENT MLX)
        endforeach()
        # Subdirectory headers.
        install(DIRECTORY "${_cuda_inc}/cooperative_groups"
            DESTINATION ${_dst}
            COMPONENT MLX
            FILES_MATCHING PATTERN "*.h")
        install(FILES "${_cuda_inc}/crt/host_defines.h"
            DESTINATION "${_dst}/crt"
            COMPONENT MLX)
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
if(WIN32)
    install(FILES ${OLLAMA_BUILD_DIR}/dl.dll
        DESTINATION ${OLLAMA_INSTALL_DIR}
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
        "${CUDAToolkit_LIBRARY_DIR}/libcudnn.so*")
    if(MLX_CUDA_LIBS)
        install(FILES ${MLX_CUDA_LIBS}
            DESTINATION ${OLLAMA_INSTALL_DIR}
            COMPONENT MLX_VENDOR)
    endif()
endif()
