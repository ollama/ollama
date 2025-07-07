if (NOT EXISTS $ENV{MUSA_PATH})
    if (NOT EXISTS /opt/musa)
        set(MUSA_PATH /usr/local/musa)
    else()
        set(MUSA_PATH /opt/musa)
    endif()
else()
    set(MUSA_PATH $ENV{MUSA_PATH})
endif()

set(CMAKE_C_COMPILER "${MUSA_PATH}/bin/clang")
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_COMPILER "${MUSA_PATH}/bin/clang++")
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${MUSA_PATH}/cmake")

find_package(MUSAToolkit)

if (MUSAToolkit_FOUND)
    message(STATUS "MUSA Toolkit found")

    if (NOT DEFINED MUSA_ARCHITECTURES)
        set(MUSA_ARCHITECTURES "21;22;31")
    endif()
    message(STATUS "Using MUSA architectures: ${MUSA_ARCHITECTURES}")

    file(GLOB   GGML_HEADERS_MUSA "../ggml-cuda/*.cuh")
    list(APPEND GGML_HEADERS_MUSA "../../include/ggml-cuda.h")
    list(APPEND GGML_HEADERS_MUSA "../ggml-musa/mudnn.cuh")

    file(GLOB   GGML_SOURCES_MUSA "../ggml-cuda/*.cu")
    file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-mma*.cu")
    list(APPEND GGML_SOURCES_MUSA ${SRCS})
    file(GLOB   SRCS "../ggml-cuda/template-instances/mmq*.cu")
    list(APPEND GGML_SOURCES_MUSA ${SRCS})
    file(GLOB   SRCS "../ggml-musa/*.cu")
    list(APPEND GGML_SOURCES_MUSA ${SRCS})

    if (GGML_CUDA_FA_ALL_QUANTS)
        file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-vec*.cu")
        list(APPEND GGML_SOURCES_MUSA ${SRCS})
        add_compile_definitions(GGML_CUDA_FA_ALL_QUANTS)
    else()
        file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-vec*q4_0-q4_0.cu")
        list(APPEND GGML_SOURCES_MUSA ${SRCS})
        file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-vec*q8_0-q8_0.cu")
        list(APPEND GGML_SOURCES_MUSA ${SRCS})
        file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-vec*f16-f16.cu")
        list(APPEND GGML_SOURCES_MUSA ${SRCS})
    endif()

    set_source_files_properties(${GGML_SOURCES_MUSA} PROPERTIES LANGUAGE CXX)
    foreach(SOURCE ${GGML_SOURCES_MUSA})
        set(COMPILE_FLAGS "-fsigned-char -x musa -mtgpu")
        foreach(ARCH ${MUSA_ARCHITECTURES})
            set(COMPILE_FLAGS "${COMPILE_FLAGS} --cuda-gpu-arch=mp_${ARCH}")
        endforeach()
        set_property(SOURCE ${SOURCE} PROPERTY COMPILE_FLAGS ${COMPILE_FLAGS})
    endforeach()

    ggml_add_backend_library(ggml-musa
                             ${GGML_HEADERS_MUSA}
                             ${GGML_SOURCES_MUSA}
                            )

    # TODO: do not use CUDA definitions for MUSA
    if (NOT GGML_BACKEND_DL)
        target_compile_definitions(ggml PUBLIC GGML_USE_CUDA)
    endif()

    add_compile_definitions(GGML_USE_MUSA)
    add_compile_definitions(GGML_CUDA_PEER_MAX_BATCH_SIZE=${GGML_CUDA_PEER_MAX_BATCH_SIZE})

    if (GGML_CUDA_FORCE_MMQ)
        add_compile_definitions(GGML_CUDA_FORCE_MMQ)
    endif()

    if (GGML_CUDA_FORCE_CUBLAS)
        add_compile_definitions(GGML_CUDA_FORCE_CUBLAS)
    endif()

    if (GGML_CUDA_NO_VMM)
        add_compile_definitions(GGML_CUDA_NO_VMM)
    endif()

    if (NOT GGML_CUDA_FA)
        add_compile_definitions(GGML_CUDA_NO_FA)
    endif()

    if (GGML_CUDA_F16 OR GGML_CUDA_DMMV_F16)
        add_compile_definitions(GGML_CUDA_F16)
    endif()

    if (GGML_CUDA_NO_PEER_COPY)
        add_compile_definitions(GGML_CUDA_NO_PEER_COPY)
    endif()

    if (GGML_STATIC)
        # TODO: mudnn has not provided static libraries yet
        target_link_libraries(ggml-musa PRIVATE MUSA::musart_static MUSA::mublas_static)
    else()
        target_link_libraries(ggml-musa PRIVATE MUSA::musart MUSA::mublas mudnn)
    endif()

    if (GGML_CUDA_NO_VMM)
        # No VMM requested, no need to link directly with the musa driver lib (libmusa.so)
    else()
        target_link_libraries(ggml-musa PRIVATE MUSA::musa_driver)
    endif()
else()
    message(FATAL_ERROR "MUSA Toolkit not found")
endif()
