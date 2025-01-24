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
set(GGML_CPU_ALL_VARIANTS ON)
set(GGML_CUDA_PEER_MAX_BATCH_SIZE 128)
set(GGML_CUDA_GRAPHS ON)

set(OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/ollama)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY         ${OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY         ${OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG   ${OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIRECTORY})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-cpu)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-cpu/amx)

set(GGML_CPU ON)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src)
set_property(TARGET ggml PROPERTY EXCLUDE_FROM_ALL TRUE)

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24" AND NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "native")
    endif()

    find_package(CUDAToolkit)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-cuda)
    set(CUDA_OUTPUT_DIRECTORY ${OUTPUT_DIRECTORY}/cuda_v${CUDAToolkit_VERSION_MAJOR})
    install(TARGETS ggml-cuda
        RUNTIME_DEPENDENCIES
            DIRECTORIES ${CUDAToolkit_BIN_DIR} ${CUDAToolkit_LIBRARY_DIR}
            PRE_INCLUDE_REGEXES cublas cublasLt cudart
            PRE_EXCLUDE_REGEXES ".*"
        RUNTIME DESTINATION ${CUDA_OUTPUT_DIRECTORY}
        LIBRARY DESTINATION ${CUDA_OUTPUT_DIRECTORY}
    )
endif()

check_language(HIP)
if(CMAKE_HIP_COMPILER)
    set(HIP_PLATFORM "amd")

    find_package(hip REQUIRED)
    if(NOT AMDGPU_TARGETS)
        list(FILTER AMDGPU_TARGETS INCLUDE REGEX "^(gfx900|gfx94[012]|gfx101[02]|gfx1030|gfx110[012])$")
    endif()

    if(AMDGPU_TARGETS)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-hip)
        set(HIP_OUTPUT_DIRECTORY ${OUTPUT_DIRECTORY}/rocm)
        install(TARGETS ggml-hip
            RUNTIME_DEPENDENCIES
                DIRECTORIES ${HIP_BIN_INSTALL_DIR} ${HIP_LIBRARY_INSTALL_DIR}
                PRE_INCLUDE_REGEXES amdhip64 hipblas rocblas amd_comgr hsa_runtime64 rocprofiler-register drm_amdgpu drm numa
                PRE_EXCLUDE_REGEXES ".*"
                POST_EXCLUDE_REGEXES "system32"
            RUNTIME DESTINATION ${HIP_OUTPUT_DIRECTORY}
            LIBRARY DESTINATION ${HIP_OUTPUT_DIRECTORY}
        )

        if(LINUX)
            install(DIRECTORY ${HIP_LIB_INSTALL_DIR}/rocblas/ DESTINATION ${HIP_OUTPUT_DIRECTORY})
        elseif(WIN32)
            install(DIRECTORY ${HIP_BIN_INSTALL_DIR}/rocblas/ DESTINATION ${HIP_OUTPUT_DIRECTORY})
        endif()
    endif()
endif()
