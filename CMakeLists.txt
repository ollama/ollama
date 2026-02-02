cmake_minimum_required(VERSION 3.21)

option(ENABLE_SYCL "Enable Intel SYCL support" OFF)
if(ENABLE_SYCL)
    if(WIN32)
        set(CMAKE_C_COMPILER icx)
        set(CMAKE_CXX_COMPILER icx)
        set(CMAKE_CXX_COMPILER_ID "IntelLLVM")
        message(STATUS "Setting C and C++ compiler to icx for Windows SYCL support (before project)")
    else()
        set(CMAKE_C_COMPILER icx)
        set(CMAKE_CXX_COMPILER icpx)
        message(STATUS "Setting C compiler to icx and C++ compiler to icpx for Linux SYCL support (before project)")
    endif()
endif()

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

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-cpu)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-cpu/amx)

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
        install(TARGETS ggml-vulkan
            RUNTIME_DEPENDENCIES
                PRE_INCLUDE_REGEXES vulkan
                PRE_EXCLUDE_REGEXES ".*"
            RUNTIME DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT Vulkan
            LIBRARY DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT Vulkan
        )
    endif()
endif()

if(ENABLE_SYCL)
    find_package(IntelSYCL)
    if(IntelSYCL_FOUND)
        set(GGML_SYCL_DNN ON)

        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-sycl)

        set(OLLAMA_SYCL_INSTALL_DIR ${OLLAMA_INSTALL_DIR}/sycl)
        install(TARGETS ggml-sycl
            RUNTIME_DEPENDENCIES
            DIRECTORIES "$ENV{CMPLR_ROOT}/bin" "$ENV{DNNLROOT}/bin" "$ENV{MKLROOT}/bin" "$ENV{TBBROOT}/bin" "$ENV{CMPLR_ROOT}/lib" "$ENV{DNNLROOT}/lib" "$ENV{MKLROOT}/lib" "$ENV{TBBROOT}/lib"
            PRE_INCLUDE_REGEXES sycl OpenCL svml irng intlc adapter ur_loader dnnl tbb mkl_sycl_blas mkl_intel_ilp64 mkl_tbb_thread mkl_core
            PRE_EXCLUDE_REGEXES ".*"
            POST_INCLUDE_REGEXES "oneapi|intel"
            POST_EXCLUDE_REGEXES "system32" "^/lib" "^/usr/lib"
            RUNTIME DESTINATION ${OLLAMA_SYCL_INSTALL_DIR} COMPONENT SYCL
            LIBRARY DESTINATION ${OLLAMA_SYCL_INSTALL_DIR} COMPONENT SYCL
        )

        if(WIN32)
            file(TO_CMAKE_PATH "$ENV{CMPLR_ROOT}/bin" CMPLR_BIN_DIR)
            file(TO_CMAKE_PATH "$ENV{DNNLROOT}/bin" DNNL_BIN_DIR)
            file(TO_CMAKE_PATH "$ENV{TCM_ROOT}/bin" TCM_BIN_DIR)
            file(TO_CMAKE_PATH "$ENV{UMF_ROOT}/bin" UMF_BIN_DIR)

            foreach(SYCL_BIN_DIR IN ITEMS "${CMPLR_BIN_DIR}" "${DNNL_BIN_DIR}" "${TCM_BIN_DIR}" "${UMF_BIN_DIR}")
                if(EXISTS ${SYCL_BIN_DIR})
                    file(GLOB DLL_FILES "${SYCL_BIN_DIR}/*.dll")
                    if(DLL_FILES)
                        install(FILES ${DLL_FILES} DESTINATION ${OLLAMA_SYCL_INSTALL_DIR} COMPONENT SYCL)
                    endif()

                    file(GLOB SYCL_SPV_FILES "${SYCL_BIN_DIR}/*.spv")
                    if(SYCL_SPV_FILES)
                        install(FILES ${SYCL_SPV_FILES} DESTINATION ${OLLAMA_SYCL_INSTALL_DIR} COMPONENT SYCL)
                    endif()
                endif()
            endforeach()
        else()
            foreach(SYCL_LIB_DIR IN ITEMS "$ENV{CMPLR_ROOT}/lib" "$ENV{DNNLROOT}/lib" "$ENV{TCM_ROOT}/lib" "$ENV{UMF_ROOT}/lib")
                if(EXISTS ${SYCL_LIB_DIR})
                    file(GLOB SYCL_SO_CANDIDATES "${SYCL_LIB_DIR}/*.so*")
                    set(SYCL_SO_FILES "")

                    foreach(SO_FILE ${SYCL_SO_CANDIDATES})
                        if(NOT SO_FILE MATCHES "\\.py$")
                            list(APPEND SYCL_SO_FILES "${SO_FILE}")
                        endif()
                    endforeach()

                    if(SYCL_SO_FILES)
                        install(FILES ${SYCL_SO_FILES} DESTINATION ${OLLAMA_SYCL_INSTALL_DIR} COMPONENT SYCL)
                    endif()

                    file(GLOB SYCL_SPV_FILES "${SYCL_LIB_DIR}/*.spv")
                    if(SYCL_SPV_FILES)
                        install(FILES ${SYCL_SPV_FILES} DESTINATION ${OLLAMA_SYCL_INSTALL_DIR} COMPONENT SYCL)
                    endif()
                endif()
            endforeach()
        endif()

        # Modify RUNPATH/RPATH of installed SYCL libraries (Linux/Unix only)
        if(NOT WIN32)
            find_program(PATCHELF_EXECUTABLE patchelf)
            if(PATCHELF_EXECUTABLE)
                install(CODE "
                  file(GLOB SYCL_LIBS \"${OLLAMA_SYCL_INSTALL_DIR}/*.so*\")
                  foreach(LIB_FILE \${SYCL_LIBS})
                      if(EXISTS \${LIB_FILE} AND NOT IS_SYMLINK \${LIB_FILE})
                          execute_process(
                              COMMAND ${PATCHELF_EXECUTABLE} --print-rpath \${LIB_FILE}
                              OUTPUT_VARIABLE CURRENT_RPATH
                              OUTPUT_STRIP_TRAILING_WHITESPACE
                              ERROR_QUIET
                          )
                          string(LENGTH \"\${CURRENT_RPATH}\" RPATH_LENGTH)
                          if(CURRENT_RPATH AND NOT CURRENT_RPATH STREQUAL \"\")
                              set(NEW_RPATH \"\$ORIGIN:\$ORIGIN/..:\${CURRENT_RPATH}\")
                          else()
                              set(NEW_RPATH \"\$ORIGIN:\$ORIGIN/..\")
                          endif()
                          execute_process(
                              COMMAND ${PATCHELF_EXECUTABLE} --set-rpath \"\${NEW_RPATH}\" \${LIB_FILE}
                              RESULT_VARIABLE PATCHELF_RESULT
                              ERROR_VARIABLE PATCHELF_ERROR
                          )
                          if(NOT PATCHELF_RESULT EQUAL 0)
                              message(WARNING \"Failed to modify RUNPATH for \${LIB_FILE}: \${PATCHELF_ERROR}\")
                          endif()
                      endif()
                  endforeach()
              "
              COMPONENT SYCL)
            else()
                message(WARNING "patchelf not found - RUNPATH of SYCL libraries will not be modified. Install patchelf for proper library resolution.")
            endif()
        endif()
    else()
        message(STATUS "SYCL compiler not found, SYCL support will be disabled")
    endif()
endif()

option(MLX_ENGINE "Enable MLX backend" OFF)

if(MLX_ENGINE)
    message(STATUS "Setting up MLX (this takes a while...)")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/x/ml/backend/mlx)

    # Find CUDA toolkit if MLX is built with CUDA support
    find_package(CUDAToolkit)

    install(TARGETS mlx mlxc
        RUNTIME_DEPENDENCIES
            DIRECTORIES ${CUDAToolkit_BIN_DIR} ${CUDAToolkit_BIN_DIR}/x64 ${CUDAToolkit_LIBRARY_DIR}
            PRE_INCLUDE_REGEXES cublas cublasLt cudart nvrtc nvrtc-builtins cudnn nccl openblas gfortran
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