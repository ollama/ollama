cmake_minimum_required(VERSION 3.21)

project(Ollama C CXX)

include(CheckLanguage)

find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

set(CMAKE_BUILD_TYPE Release)
set(BUILD_SHARED_LIBS ON)

set(GGML_CCACHE ON)
set(GGML_SCHED_MAX_COPIES 4)
set(GGML_CPU_ALL_VARIANTS ON)
set(GGML_CUDA_PEER_MAX_BATCH_SIZE 128)

add_compile_definitions(GGML_BUILD)
add_compile_definitions(GGML_SHARED)
add_compile_definitions(GGML_BACKEND_DL)
add_compile_definitions(GGML_BACKEND_SHARED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-cpu)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-cpu/amx)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src EXCLUDE_FROM_ALL)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-cpu)

find_package(BLAS)
if(NOT BLAS_VENDOR)
    set(GGML_BLAS_VENDOR "Generic")
else()
    set(GGML_BLAS_VENDOR ${BLAS_VENDOR})
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-blas)
target_compile_features(ggml-blas PRIVATE cxx_std_11)

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-cuda)
endif()

check_language(HIP)
if(CMAKE_HIP_COMPILER)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-hip)
endif()
