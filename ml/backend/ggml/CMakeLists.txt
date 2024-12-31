include(CheckLanguage)

find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(GGML_CCACHE ON)
set(GGML_SCHED_MAX_COPIES 4)
set(GGML_CPU_ALL_VARIANTS ON)
set(GGML_CUDA_PEER_MAX_BATCH_SIZE 128)
set(GGML_LLAMAFILE ON)

add_compile_definitions(GGML_BUILD)
add_compile_definitions(GGML_SHARED)
add_compile_definitions(GGML_BACKEND_DL)
add_compile_definitions(GGML_BACKEND_SHARED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ggml/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/ggml-cpu)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/ggml-cpu/amx)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ggml/src EXCLUDE_FROM_ALL)
set_target_output_directory(ggml-base)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/ggml-cpu)
set_target_output_directory(ggml-cpu)

find_package(BLAS)
if(NOT BLAS_VENDOR)
    set(GGML_BLAS_VENDOR "Generic")
else()
    set(GGML_BLAS_VENDOR ${BLAS_VENDOR})
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/ggml-blas)
set_target_output_directory(ggml-blas)

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/ggml-cuda)
    set_target_output_directory(ggml-cuda)
endif()

check_language(HIP)
if(CMAKE_HIP_COMPILER)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/ggml-hip)
    set_target_output_directory(ggml-hip)
endif()
