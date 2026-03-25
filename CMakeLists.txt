cmake_minimum_required(VERSION 3.21)

project(Ollama C CXX)

# Handle cross-compilation on macOS: when CMAKE_OSX_ARCHITECTURES is set to a
# single architecture different from the host, override CMAKE_SYSTEM_PROCESSOR
# to match. This is necessary because CMAKE_SYSTEM_PROCESSOR defaults to the
# host architecture, but downstream projects (like MLX) use it to detect the
# target architecture.
if(CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_OSX_ARCHITECTURES MATCHES ";")
    # Single architecture specified
    if(CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64" AND NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        message(STATUS "Cross-compiling for x86_64: overriding CMAKE_SYSTEM_PROCESSOR from ${CMAKE_SYSTEM_PROCESSOR} to x86_64")
        set(CMAKE_SYSTEM_PROCESSOR "x86_64")
    elseif(CMAKE_OSX_ARCHITECTURES STREQUAL "arm64" AND NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        message(STATUS "Cross-compiling for arm64: overriding CMAKE_SYSTEM_PROCESSOR from ${CMAKE_SYSTEM_PROCESSOR} to arm64")
        set(CMAKE_SYSTEM_PROCESSOR "arm64")
    endif()
endif()

include(CheckLanguage)
include(GNUInstallDirs)

find_package(Threads REQUIRED)

set(CMAKE_BUILD_TYPE Release)

# These defaults can be overridden by presets (e.g., for static macOS llama-server builds)
if(NOT DEFINED BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON) # Recent versions of MLX Requires gnu++17 extensions to compile properly

# GGML backend for inference is provided by llama-server (built separately via
# llama/server/CMakeLists.txt using FetchContent from upstream llama.cpp).
# This root CMakeLists.txt only builds MLX (when enabled).

if(APPLE)
    set(CMAKE_BUILD_RPATH "@loader_path")
    set(CMAKE_INSTALL_RPATH "@loader_path")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
endif()

set(OLLAMA_BUILD_DIR ${CMAKE_BINARY_DIR}/lib/ollama)
set(OLLAMA_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/lib/ollama/${OLLAMA_RUNNER_DIR})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY         ${OLLAMA_BUILD_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${OLLAMA_BUILD_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${OLLAMA_BUILD_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY         ${OLLAMA_BUILD_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG   ${OLLAMA_BUILD_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${OLLAMA_BUILD_DIR})

option(MLX_ENGINE "Enable MLX backend" OFF)
if(MLX_ENGINE)
    message(STATUS "Setting up MLX (this takes a while...)")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/x/imagegen/mlx)

    # Find CUDA toolkit if MLX is built with CUDA support
    find_package(CUDAToolkit)

    # Build list of directories for runtime dependency resolution
    set(MLX_RUNTIME_DIRS ${CUDAToolkit_BIN_DIR} ${CUDAToolkit_BIN_DIR}/x64 ${CUDAToolkit_LIBRARY_DIR})
    # Add cuDNN bin paths for DLLs (Windows MLX CUDA builds)
    # CUDNN_ROOT_DIR is the standard CMake variable for cuDNN location
    if(DEFINED ENV{CUDNN_ROOT_DIR})
        # cuDNN 9.x has versioned subdirectories under bin/ (e.g., bin/13.0/)
        file(GLOB CUDNN_BIN_SUBDIRS "$ENV{CUDNN_ROOT_DIR}/bin/*")
        list(APPEND MLX_RUNTIME_DIRS ${CUDNN_BIN_SUBDIRS})
    endif()
    # Add build output directory and MLX dependency build directories
    list(APPEND MLX_RUNTIME_DIRS ${OLLAMA_BUILD_DIR})
    # OpenBLAS DLL location (pre-built zip extracts into openblas-src/bin/)
    list(APPEND MLX_RUNTIME_DIRS ${CMAKE_BINARY_DIR}/_deps/openblas-src/bin)
    # NCCL: on Linux, if real NCCL is found, cmake bundles libnccl.so via the
    # regex below. If NCCL is not found, MLX links a static stub (OBJECT lib)
    # so there is no runtime dependency. This path covers the stub build dir
    # for windows so we include the DLL in our dependencies.
    list(APPEND MLX_RUNTIME_DIRS ${CMAKE_BINARY_DIR}/_deps/mlx-build/mlx/distributed/nccl/nccl_stub-prefix/src/nccl_stub-build/Release)

    # Base regexes for runtime dependencies (cross-platform)
    set(MLX_INCLUDE_REGEXES cublas cublasLt cudart cufft nvrtc nvrtc-builtins cudnn nccl openblas gfortran)
    # On Windows, also include dl.dll (dlfcn-win32 POSIX emulation layer)
    if(WIN32)
        list(APPEND MLX_INCLUDE_REGEXES "^dl\\.dll$")
    endif()

    install(TARGETS mlx mlxc
        RUNTIME_DEPENDENCIES
            DIRECTORIES ${MLX_RUNTIME_DIRS}
            PRE_INCLUDE_REGEXES ${MLX_INCLUDE_REGEXES}
            PRE_EXCLUDE_REGEXES ".*"
        RUNTIME DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT MLX
        LIBRARY DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT MLX
        FRAMEWORK DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT MLX
    )

    # Install the Metal library for macOS arm64 (must be colocated with the binary)
    # Metal backend is only built for arm64, not x86_64
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
    #   ${OLLAMA_INSTALL_DIR}/include/cccl/{cuda,nv}/  — CCCL headers
    #   ${OLLAMA_INSTALL_DIR}/include/*.h               — CUDA toolkit headers
    #
    # MLX's jit_module.cpp resolves CCCL via
    #   current_binary_dir()[.parent_path()] / "include" / "cccl"
    # On Linux, MLX's jit_module.cpp resolves CCCL via
    #   current_binary_dir().parent_path() / "include" / "cccl", so we create a
    #   symlink from lib/ollama/include -> ${OLLAMA_RUNNER_DIR}/include
    #   This will need refinement if we add multiple CUDA versions for MLX in the future.
    # CUDA runtime headers are found via CUDA_PATH env var (set by mlxrunner).
    if(EXISTS ${CMAKE_BINARY_DIR}/_deps/cccl-src/include/cuda)
        install(DIRECTORY ${CMAKE_BINARY_DIR}/_deps/cccl-src/include/cuda
            DESTINATION ${OLLAMA_INSTALL_DIR}/include/cccl
            COMPONENT MLX)
        install(DIRECTORY ${CMAKE_BINARY_DIR}/_deps/cccl-src/include/nv
            DESTINATION ${OLLAMA_INSTALL_DIR}/include/cccl
            COMPONENT MLX)
        if(NOT WIN32 AND NOT APPLE)
            install(CODE "
                set(_link \"${CMAKE_INSTALL_PREFIX}/lib/ollama/include\")
                set(_target \"${OLLAMA_RUNNER_DIR}/include\")
                if(NOT EXISTS \${_link})
                    execute_process(COMMAND \${CMAKE_COMMAND} -E create_symlink \${_target} \${_link})
                endif()
            " COMPONENT MLX)
        endif()
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
            # Subdirectory headers
            install(DIRECTORY "${_cuda_inc}/cooperative_groups"
                DESTINATION ${_dst}
                COMPONENT MLX
                FILES_MATCHING PATTERN "*.h")
            install(FILES "${_cuda_inc}/crt/host_defines.h"
                DESTINATION "${_dst}/crt"
                COMPONENT MLX)
        endif()
    endif()

    # On Windows, explicitly install dl.dll (dlfcn-win32 POSIX dlopen emulation)
    # RUNTIME_DEPENDENCIES auto-excludes it via POST_EXCLUDE_FILES_STRICT because
    # dlfcn-win32 is a known CMake target with its own install rules (which install
    # to the wrong destination). We must install it explicitly here.
    if(WIN32)
        install(FILES ${OLLAMA_BUILD_DIR}/dl.dll
            DESTINATION ${OLLAMA_INSTALL_DIR}
            COMPONENT MLX)
    endif()

    # Manually install CUDA runtime libraries that MLX loads via dlopen
    # (not detected by RUNTIME_DEPENDENCIES since they aren't link-time deps)
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
                COMPONENT MLX)
        endif()
    endif()
endif()


