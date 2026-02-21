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
set(BUILD_SHARED_LIBS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON) # Recent versions of MLX Requires gnu++17 extensions to compile properly

set(GGML_BUILD ON)
set(GGML_SHARED ON)
set(GGML_CCACHE ON)
set(GGML_BACKEND_DL ON)
set(GGML_BACKEND_SHARED ON)
set(GGML_SCHED_MAX_COPIES 4)

set(GGML_LLAMAFILE ON)
set(GGML_CUDA_PEER_MAX_BATCH_SIZE 128)
set(GGML_CUDA_GRAPHS ON)
set(GGML_CUDA_FA ON)
set(GGML_CUDA_COMPRESSION_MODE default)

if((CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
    OR (NOT CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64|ARMv[0-9]+"))
    set(GGML_CPU_ALL_VARIANTS ON)
endif()

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

# Store ggml include paths for use with target_include_directories later.
# We avoid global include_directories() to prevent polluting the include path
# for other projects like MLX (whose openblas dependency has its own common.h).
set(GGML_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src
    ${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-cpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-cpu/amx
)

add_compile_definitions(NDEBUG GGML_VERSION=0x0 GGML_COMMIT=0x0)

# Define GGML version variables for shared library SOVERSION
# These are required by ggml/src/CMakeLists.txt for proper library versioning
set(GGML_VERSION_MAJOR 0)
set(GGML_VERSION_MINOR 0)
set(GGML_VERSION_PATCH 0)
set(GGML_VERSION "${GGML_VERSION_MAJOR}.${GGML_VERSION_MINOR}.${GGML_VERSION_PATCH}")

set(GGML_CPU ON)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src)
set_property(TARGET ggml PROPERTY EXCLUDE_FROM_ALL TRUE)

get_target_property(CPU_VARIANTS ggml-cpu MANUALLY_ADDED_DEPENDENCIES)
if(NOT CPU_VARIANTS)
    set(CPU_VARIANTS "ggml-cpu")
endif()

# Apply ggml include directories to ggml targets only (not globally)
target_include_directories(ggml-base PRIVATE ${GGML_INCLUDE_DIRS})
foreach(variant ${CPU_VARIANTS})
    if(TARGET ${variant})
        target_include_directories(${variant} PRIVATE ${GGML_INCLUDE_DIRS})
    endif()
endforeach()

install(TARGETS ggml-base ${CPU_VARIANTS}
    RUNTIME_DEPENDENCIES
        PRE_EXCLUDE_REGEXES ".*"
    RUNTIME DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT CPU
    LIBRARY DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT CPU
    FRAMEWORK DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT CPU
)

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24" AND NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "native")
    endif()

    find_package(CUDAToolkit)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-cuda)
    target_include_directories(ggml-cuda PRIVATE ${GGML_INCLUDE_DIRS})
    install(TARGETS ggml-cuda
        RUNTIME_DEPENDENCIES
            DIRECTORIES ${CUDAToolkit_BIN_DIR} ${CUDAToolkit_BIN_DIR}/x64 ${CUDAToolkit_LIBRARY_DIR}
            PRE_INCLUDE_REGEXES cublas cublasLt cudart
            PRE_EXCLUDE_REGEXES ".*"
        RUNTIME DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT CUDA
        LIBRARY DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT CUDA
    )
endif()

set(WINDOWS_AMDGPU_TARGETS_EXCLUDE_REGEX "^gfx(908|90a|1200|1201):xnack[+-]$"
    CACHE STRING
    "Regular expression describing AMDGPU_TARGETS not supported on Windows. Override to force building these targets. Default \"^gfx(908|90a|1200|1201):xnack[+-]$\"."
)

check_language(HIP)
if(CMAKE_HIP_COMPILER)
    set(HIP_PLATFORM "amd")

    if(NOT AMDGPU_TARGETS)
        find_package(hip REQUIRED)
        list(FILTER AMDGPU_TARGETS INCLUDE REGEX "^gfx(94[012]|101[02]|1030|110[012]|120[01])$")
    endif()

    if(WIN32 AND WINDOWS_AMDGPU_TARGETS_EXCLUDE_REGEX)
        list(FILTER AMDGPU_TARGETS EXCLUDE REGEX ${WINDOWS_AMDGPU_TARGETS_EXCLUDE_REGEX})
    endif()

    if(AMDGPU_TARGETS)
        find_package(hip REQUIRED)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-hip)
        target_include_directories(ggml-hip PRIVATE ${GGML_INCLUDE_DIRS})

        if (WIN32)
            target_compile_definitions(ggml-hip PRIVATE GGML_CUDA_NO_PEER_COPY)
        endif()

        target_compile_definitions(ggml-hip PRIVATE GGML_HIP_NO_VMM)

        install(TARGETS ggml-hip
            RUNTIME_DEPENDENCY_SET rocm
            RUNTIME DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT HIP
            LIBRARY DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT HIP
        )
        install(RUNTIME_DEPENDENCY_SET rocm
                DIRECTORIES ${HIP_BIN_INSTALL_DIR} ${HIP_LIB_INSTALL_DIR}
                PRE_INCLUDE_REGEXES hipblas rocblas amdhip64 rocsolver amd_comgr hsa-runtime64 rocsparse tinfo rocprofiler-register drm drm_amdgpu numa elf
                PRE_EXCLUDE_REGEXES ".*"
                POST_EXCLUDE_REGEXES "system32"
            RUNTIME DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT HIP
            LIBRARY DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT HIP
        )

        foreach(HIP_LIB_BIN_INSTALL_DIR IN ITEMS ${HIP_BIN_INSTALL_DIR} ${HIP_LIB_INSTALL_DIR})
            if(EXISTS ${HIP_LIB_BIN_INSTALL_DIR}/rocblas)
                install(DIRECTORY ${HIP_LIB_BIN_INSTALL_DIR}/rocblas DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT HIP)
                break()
            endif()
        endforeach()
    endif()
endif()

if(NOT APPLE)
    find_package(Vulkan)
    if(Vulkan_FOUND)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-vulkan)
        target_include_directories(ggml-vulkan PRIVATE ${GGML_INCLUDE_DIRS})
        install(TARGETS ggml-vulkan
            RUNTIME_DEPENDENCIES
                PRE_INCLUDE_REGEXES vulkan
                PRE_EXCLUDE_REGEXES ".*"
            RUNTIME DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT Vulkan
            LIBRARY DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT Vulkan
        )
    endif()
endif()

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
    # NCCL stub DLL location
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

    # Install CCCL headers for NVRTC JIT compilation at runtime.
    # MLX's own install rules use the default component so they get skipped by
    # --component MLX. The JIT compiler (jit_module.cpp) looks for these at
    # current_binary_dir() / "include" / "cccl" on Windows.
    if(EXISTS ${CMAKE_BINARY_DIR}/_deps/cccl-src/include/cuda)
        install(DIRECTORY ${CMAKE_BINARY_DIR}/_deps/cccl-src/include/cuda
            DESTINATION ${OLLAMA_INSTALL_DIR}/include/cccl
            COMPONENT MLX)
        install(DIRECTORY ${CMAKE_BINARY_DIR}/_deps/cccl-src/include/nv
            DESTINATION ${OLLAMA_INSTALL_DIR}/include/cccl
            COMPONENT MLX)
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

    # Manually install cudart and cublas since they might not be picked up as direct dependencies
    if(CUDAToolkit_FOUND)
        file(GLOB CUDART_LIBS
            "${CUDAToolkit_LIBRARY_DIR}/libcudart.so*"
            "${CUDAToolkit_LIBRARY_DIR}/libcublas.so*")
        if(CUDART_LIBS)
            install(FILES ${CUDART_LIBS}
                DESTINATION ${OLLAMA_INSTALL_DIR}
                COMPONENT MLX)
        endif()
    endif()
endif()
