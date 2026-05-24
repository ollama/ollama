# ============================================================================
# cmake/ggml_fattn_pool.cmake
# Ninja pool configuration for rocWMMA FATTN serial compilation
# ============================================================================

if(CMAKE_GENERATOR MATCHES "Ninja" AND GGML_HIP_ROCWMMA_FATTN)
    # Create a "fattn_serial" pool with depth=1 (serial execution)
    set_property(GLOBAL PROPERTY JOB_POOLS fattn_serial=1)

    # Apply pool to FATTN source files
    set_source_files_properties(
        ${CMAKE_SOURCE_DIR}/ggml/src/ggml-cuda/fattn-mma-f16-instance-ncols1_4-ncols2_8.cu
        ${CMAKE_SOURCE_DIR}/ggml/src/ggml-cuda/fattn-mma-f16-instance-ncols1_8-ncols2_8.cu
        ${CMAKE_SOURCE_DIR}/ggml/src/ggml-cuda/fattn-mma-f16-instance-ncols1_1-ncols2_8.cu
        ${CMAKE_SOURCE_DIR}/ggml/src/ggml-cuda/fattn-mma-f16-instance-ncols1_2-ncols2_8.cu
        ${CMAKE_SOURCE_DIR}/ggml/src/ggml-cuda/fattn-mma-f16-instance-ncols1_4-ncols2_1.cu
        ${CMAKE_SOURCE_DIR}/ggml/src/ggml-cuda/fattn-mma-f16-instance-ncols1_8-ncols2_1.cu
        ${CMAKE_SOURCE_DIR}/ggml/src/ggml-cuda/fattn-mma-f32-instance-ncols1_4-ncols2_8.cu
        ${CMAKE_SOURCE_DIR}/ggml/src/ggml-cuda/fattn-mma-f32-instance-ncols1_8-ncols2_8.cu
        PROPERTIES
            JOB_POOL_COMPILE fattn_serial
            COMPILE_FLAGS "-O2 -mllvm -amdgpu-early-inline-all=false"
    )

    message(STATUS "Ninja: FATTN files will compile serially in 'fattn_serial' pool")
endif()
