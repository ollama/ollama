# ============================================================================
# cmake/ggml_fattn_serial.cmake
# Fix for rocWMMA Flash Attention template kernels consuming too much RAM
# These files need ~8GB each and crash when compiled in parallel
# ============================================================================

# List of rocWMMA FATTN files that must be compiled serially
set(GGML_FATTN_SERIAL_FILES
    "ggml/src/ggml-cuda/fattn-mma-f16-instance-ncols1_4-ncols2_8.cu"
    "ggml/src/ggml-cuda/fattn-mma-f16-instance-ncols1_8-ncols2_8.cu"
    "ggml/src/ggml-cuda/fattn-mma-f16-instance-ncols1_1-ncols2_8.cu"
    "ggml/src/ggml-cuda/fattn-mma-f16-instance-ncols1_2-ncols2_8.cu"
    "ggml/src/ggml-cuda/fattn-mma-f16-instance-ncols1_4-ncols2_1.cu"
    "ggml/src/ggml-cuda/fattn-mma-f16-instance-ncols1_8-ncols2_1.cu"
    "ggml/src/ggml-cuda/fattn-mma-f32-instance-ncols1_4-ncols2_8.cu"
    "ggml/src/ggml-cuda/fattn-mma-f32-instance-ncols1_8-ncols2_8.cu"
)

# Function to compile a single FATTN file serially
function(compile_fattn_serial SOURCE_FILE OUTPUT_OBJ)
    get_filename_component(FATTN_NAME ${SOURCE_FILE} NAME_WE)
    set(FATTN_OBJ "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ggml.dir/${FATTN_NAME}.obj")

    # Serial compilation: -j1, no parallel jobs
    # Add memory-saving flags for the compiler instance
    set(FATTN_FLAGS "-O2")  # Use -O2 instead of -O3 for these files
    set(FATTN_FLAGS "${FATTN_FLAGS} -mllvm -amdgpu-early-inline-all=false")
    set(FATTN_FLAGS "${FATTN_FLAGS} -mllvm -amdgpu-inline-threshold=100")

    add_custom_command(
        OUTPUT ${FATTN_OBJ}
        COMMAND ${CMAKE_HIP_COMPILER}
            -c ${SOURCE_FILE}
            -o ${FATTN_OBJ}
            ${CMAKE_HIP_FLAGS}
            ${FATTN_FLAGS}
            -I${CMAKE_SOURCE_DIR}/ggml/include
            -I${CMAKE_SOURCE_DIR}/ggml/src
            -I${HIP_PATH}/include
            -DGGML_CUDA_DMMV_X=32
            -DGGML_CUDA_MMV_Y=1
            -DGGML_CUDA_F16
            -DGGML_USE_HIP
            -DGGML_USE_CUDA
            --offload-arch=${AMDGPU_TARGETS}
        DEPENDS ${SOURCE_FILE}
        COMMENT "Serial compilation: ${FATTN_NAME} (high RAM usage)"
        VERBATIM
    )

    set(${OUTPUT_OBJ} ${FATTN_OBJ} PARENT_SCOPE)
endfunction()

# Main FATTN build logic
if(GGML_HIP_ROCWMMA_FATTN)
    message(STATUS "rocWMMA Flash Attention: Using serial compilation for heavy templates")
    message(STATUS "  Each FATTN file needs ~8GB RAM, compiling one at a time")

    set(GGML_FATTN_OBJECTS)
    foreach(FATTN_FILE ${GGML_FATTN_SERIAL_FILES})
        if(EXISTS "${CMAKE_SOURCE_DIR}/${FATTN_FILE}")
            compile_fattn_serial("${CMAKE_SOURCE_DIR}/${FATTN_FILE}" FATTN_OBJ)
            list(APPEND GGML_FATTN_OBJECTS ${FATTN_OBJ})
        endif()
    endforeach()

    # Add precompiled objects to ggml target
    target_sources(ggml PRIVATE ${GGML_FATTN_OBJECTS})
endif()
