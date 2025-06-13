cmake_minimum_required(VERSION 3.21)

project(Ollama C CXX)

include(CheckLanguage)

find_package(Threads REQUIRED)

set(CMAKE_BUILD_TYPE Release)
set(BUILD_SHARED_LIBS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

if(CMAKE_OSX_ARCHITECTURES MATCHES "x86_64")
    set(CMAKE_BUILD_RPATH "@loader_path")
    set(CMAKE_INSTALL_RPATH "@loader_path")
endif()

set(OLLAMA_BUILD_DIR ${CMAKE_BINARY_DIR}/lib/ollama)
set(OLLAMA_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/lib/ollama)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OLLAMA_BUILD_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${OLLAMA_BUILD_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${OLLAMA_BUILD_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OLLAMA_BUILD_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${OLLAMA_BUILD_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${OLLAMA_BUILD_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-cpu)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-cpu/amx)

add_compile_definitions(NDEBUG)

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
    set(OLLAMA_CUDA_INSTALL_DIR ${OLLAMA_INSTALL_DIR}/cuda_v${CUDAToolkit_VERSION_MAJOR})
    install(TARGETS ggml-cuda
        RUNTIME_DEPENDENCIES
        DIRECTORIES ${CUDAToolkit_BIN_DIR} ${CUDAToolkit_LIBRARY_DIR}
        PRE_INCLUDE_REGEXES cublas cublasLt cudart
        PRE_EXCLUDE_REGEXES ".*"
        RUNTIME DESTINATION ${OLLAMA_CUDA_INSTALL_DIR} COMPONENT CUDA
        LIBRARY DESTINATION ${OLLAMA_CUDA_INSTALL_DIR} COMPONENT CUDA
    )
endif()

set(WINDOWS_AMDGPU_TARGETS_EXCLUDE_REGEX "^gfx(906|908|90a|1200|1201):xnack[+-]$"
    CACHE STRING
    "Regular expression describing AMDGPU_TARGETS not supported on Windows. Override to force building these targets. Default \"^gfx(906|908|90a|1200|1201):xnack[+-]$\"."
)

check_language(HIP)
if(CMAKE_HIP_COMPILER)
    set(HIP_PLATFORM "amd")

    find_package(hip REQUIRED)
    if(NOT AMDGPU_TARGETS)
        list(FILTER AMDGPU_TARGETS INCLUDE REGEX "^gfx(900|94[012]|101[02]|1030|110[012]|120[01])$")
    elseif(WIN32 AND WINDOWS_AMDGPU_TARGETS_EXCLUDE_REGEX)
        list(FILTER AMDGPU_TARGETS EXCLUDE REGEX ${WINDOWS_AMDGPU_TARGETS_EXCLUDE_REGEX})
    endif()

    if(AMDGPU_TARGETS)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-hip)

        if(WIN32)
            target_compile_definitions(ggml-hip PRIVATE GGML_CUDA_NO_PEER_COPY)
        endif()

        target_compile_definitions(ggml-hip PRIVATE GGML_HIP_NO_VMM)

        set(OLLAMA_HIP_INSTALL_DIR ${OLLAMA_INSTALL_DIR}/rocm)
        install(TARGETS ggml-hip
            RUNTIME_DEPENDENCIES
            DIRECTORIES ${HIP_BIN_INSTALL_DIR} ${HIP_LIB_INSTALL_DIR}
            PRE_INCLUDE_REGEXES hipblas rocblas amdhip64 rocsolver amd_comgr hsa-runtime64 rocsparse tinfo rocprofiler-register drm drm_amdgpu numa elf
            PRE_EXCLUDE_REGEXES ".*"
            POST_EXCLUDE_REGEXES "system32"
            RUNTIME DESTINATION ${OLLAMA_HIP_INSTALL_DIR} COMPONENT HIP
            LIBRARY DESTINATION ${OLLAMA_HIP_INSTALL_DIR} COMPONENT HIP
        )

        foreach(HIP_LIB_BIN_INSTALL_DIR IN ITEMS ${HIP_BIN_INSTALL_DIR} ${HIP_LIB_INSTALL_DIR})
            if(EXISTS ${HIP_LIB_BIN_INSTALL_DIR}/rocblas)
                install(DIRECTORY ${HIP_LIB_BIN_INSTALL_DIR}/rocblas DESTINATION ${OLLAMA_HIP_INSTALL_DIR} COMPONENT HIP)
                break()
            endif()
        endforeach()
    endif()
endif()

option(ENABLE_SYCL "Enable SYCL support" OFF)
if(ENABLE_SYCL)
    find_package(IntelSYCL)
    if(IntelSYCL_FOUND)
        set(GGML_SYCL_TARGET "INTEL")
        set(GGML_SYCL_DNN ON)

        if(WIN32)
            set(CMAKE_C_COMPILER icx)
            set(CMAKE_CXX_COMPILER icx)
            set(CMAKE_CXX_COMPILER_ID "IntelLLVM")
            message(STATUS "Setting C and C++ compiler to icx for Windows SYCL support")
        else()
            set(CMAKE_C_COMPILER icx)
            set(CMAKE_CXX_COMPILER icpx)
            message(STATUS "Setting C compiler to icx and C++ compiler to icpx for Linux SYCL support")
        endif()

        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-sycl)

        set(OLLAMA_SYCL_INSTALL_DIR ${OLLAMA_INSTALL_DIR}/sycl)
        install(TARGETS ggml-sycl
            RUNTIME_DEPENDENCIES
            DIRECTORIES "$ENV{CMPLR_ROOT}/bin" "$ENV{DNNLROOT}/bin" "$ENV{TBBROOT}/bin" "$ENV{MKLROOT}/bin" "$ENV{CMPLR_ROOT}/lib" "$ENV{DNNLROOT}/lib" "$ENV{TBBROOT}/lib" "$ENV{MKLROOT}/lib"
            PRE_INCLUDE_REGEXES sycl OpenCL svml irng intlc ur_loader dnnl tbb mkl_sycl_blas mkl_intel_ilp64 mkl_tbb_thread mkl_core
            PRE_EXCLUDE_REGEXES ".*"
            POST_EXCLUDE_REGEXES "system32"
            RUNTIME DESTINATION ${OLLAMA_SYCL_INSTALL_DIR} COMPONENT SYCL
            LIBRARY DESTINATION ${OLLAMA_SYCL_INSTALL_DIR} COMPONENT SYCL
        )

        foreach(SYCL_LIB_DIR IN ITEMS "$ENV{CMPLR_ROOT}/lib" "$ENV{DNNLROOT}/lib" "$ENV{TBBROOT}/lib" "$ENV{MKLROOT}/lib" "$ENV{TCM_ROOT}/lib" "$ENV{UMF_ROOT}/lib")
            if(EXISTS ${SYCL_LIB_DIR})
                file(GLOB SYCL_SO_FILES "${SYCL_LIB_DIR}/*.so*")
                if(SYCL_SO_FILES)
                    message(STATUS "Installing SYCL .so files from: ${SYCL_LIB_DIR}")
                    install(FILES ${SYCL_SO_FILES} DESTINATION ${OLLAMA_SYCL_INSTALL_DIR} COMPONENT SYCL)
                endif()
            endif()
        endforeach()

        # Modify RUNPATH/RPATH of installed SYCL libraries (Linux/Unix only)
        if(NOT WIN32)
            find_program(PATCHELF_EXECUTABLE patchelf)
            if(PATCHELF_EXECUTABLE)
                install(CODE "
                  file(GLOB SYCL_LIBS \"${OLLAMA_SYCL_INSTALL_DIR}/*.so*\")
                  foreach(LIB_FILE \${SYCL_LIBS})
                      if(EXISTS \${LIB_FILE} AND NOT IS_SYMLINK \${LIB_FILE})
                          message(STATUS \"Modifying RUNPATH for: \${LIB_FILE}\")
                          # Get current RUNPATH
                          execute_process(
                              COMMAND ${PATCHELF_EXECUTABLE} --print-rpath \${LIB_FILE}
                              OUTPUT_VARIABLE CURRENT_RPATH
                              OUTPUT_STRIP_TRAILING_WHITESPACE
                              ERROR_QUIET
                          )
                          # Prepend new paths to existing RUNPATH
                          message(STATUS \"Current RPATH: '\${CURRENT_RPATH}'\")
                          message(STATUS \"CURRENT_RPATH length: \${CURRENT_RPATH}\")
                          string(LENGTH \"\${CURRENT_RPATH}\" RPATH_LENGTH)
                          message(STATUS \"CURRENT_RPATH length: \${RPATH_LENGTH}\")
                          if(CURRENT_RPATH AND NOT CURRENT_RPATH STREQUAL \"\")
                              set(NEW_RPATH \"\$ORIGIN:\$ORIGIN/..:\${CURRENT_RPATH}\")
                              message(STATUS \"Setting NEW_RPATH with existing: '\${NEW_RPATH}'\")
                          else()
                              set(NEW_RPATH \"\$ORIGIN:\$ORIGIN/..\")
                              message(STATUS \"Setting NEW_RPATH without existing: '\${NEW_RPATH}'\")
                          endif()
                          execute_process(
                              COMMAND ${PATCHELF_EXECUTABLE} --set-rpath \"\${NEW_RPATH}\" \${LIB_FILE}
                              RESULT_VARIABLE PATCHELF_RESULT
                              OUTPUT_VARIABLE PATCHELF_OUTPUT
                              ERROR_VARIABLE PATCHELF_ERROR
                          )
                          if(NOT PATCHELF_RESULT EQUAL 0)
                              message(WARNING \"Failed to modify RUNPATH for \${LIB_FILE}: \${PATCHELF_ERROR}\")
                          else()
                              message(STATUS \"Successfully prepended RUNPATH with \$ORIGIN:\$ORIGIN/.. for \${LIB_FILE}\")
                          endif()
                      endif()
                  endforeach()
              "                    COMPONENT SYCL)
            else()
                message(FATAL_ERROR "patchelf not found. RUNPATH of SYCL libraries cannot be modified.")
            endif()
        else()
            message(STATUS "Windows detected: DLL search path modification not needed (uses current directory and PATH)")
        endif()
    else()
        message(STATUS "SYCL compiler not found, SYCL support will be disabled")
    endif()
else()
    message(STATUS "SYCL support disabled")
endif()