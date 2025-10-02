if (GGML_STATIC)
    set(BLA_STATIC ON)
endif()
#if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.22)
#    set(BLA_SIZEOF_INTEGER 8)
#endif()

set(BLA_VENDOR ${GGML_BLAS_VENDOR})
find_package(BLAS)

if (BLAS_FOUND)
    message(STATUS "BLAS found, Libraries: ${BLAS_LIBRARIES}")

    ggml_add_backend_library(ggml-blas
                             ggml-blas.cpp
                            )

    if (${GGML_BLAS_VENDOR} MATCHES "Apple")
        add_compile_definitions(ACCELERATE_NEW_LAPACK)
        add_compile_definitions(ACCELERATE_LAPACK_ILP64)
        add_compile_definitions(GGML_BLAS_USE_ACCELERATE)
    elseif ("${BLAS_INCLUDE_DIRS}" STREQUAL "")
        # BLAS_INCLUDE_DIRS is missing in FindBLAS.cmake.
        # see https://gitlab.kitware.com/cmake/cmake/-/issues/20268
        find_package(PkgConfig REQUIRED)
        if (${GGML_BLAS_VENDOR} MATCHES "Generic")
            pkg_check_modules(DepBLAS blas)
        elseif (${GGML_BLAS_VENDOR} MATCHES "OpenBLAS")
            # As of openblas v0.3.22, the 64-bit is named openblas64.pc
            pkg_check_modules(DepBLAS openblas64)
            if (NOT DepBLAS_FOUND)
                pkg_check_modules(DepBLAS openblas)
            endif()
        elseif (${GGML_BLAS_VENDOR} MATCHES "FLAME")
            add_compile_definitions(GGML_BLAS_USE_BLIS)
            pkg_check_modules(DepBLAS blis)
        elseif (${GGML_BLAS_VENDOR} MATCHES "ATLAS")
            pkg_check_modules(DepBLAS blas-atlas)
        elseif (${GGML_BLAS_VENDOR} MATCHES "FlexiBLAS")
            pkg_check_modules(DepBLAS flexiblas_api)
        elseif (${GGML_BLAS_VENDOR} MATCHES "Intel")
            add_compile_definitions(GGML_BLAS_USE_MKL)
            # all Intel* libraries share the same include path
            pkg_check_modules(DepBLAS mkl-sdl)
        elseif (${GGML_BLAS_VENDOR} MATCHES "NVHPC")
            # this doesn't provide pkg-config
            # suggest to assign BLAS_INCLUDE_DIRS on your own
            if ("${NVHPC_VERSION}" STREQUAL "")
                message(WARNING "Better to set NVHPC_VERSION")
            else()
                set(DepBLAS_FOUND ON)
                set(DepBLAS_INCLUDE_DIRS "/opt/nvidia/hpc_sdk/${CMAKE_SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR}/${NVHPC_VERSION}/math_libs/include")
            endif()
        endif()
        if (DepBLAS_FOUND)
            set(BLAS_INCLUDE_DIRS ${DepBLAS_INCLUDE_DIRS})
        else()
            message(WARNING "BLAS_INCLUDE_DIRS neither been provided nor been automatically"
            " detected by pkgconfig, trying to find cblas.h from possible paths...")
            find_path(BLAS_INCLUDE_DIRS
                NAMES cblas.h
                HINTS
                    /usr/include
                    /usr/local/include
                    /usr/include/openblas
                    /opt/homebrew/opt/openblas/include
                    /usr/local/opt/openblas/include
                    /usr/include/x86_64-linux-gnu/openblas/include
            )
        endif()
    endif()

    message(STATUS "BLAS found, Includes: ${BLAS_INCLUDE_DIRS}")

    target_compile_options(ggml-blas PRIVATE ${BLAS_LINKER_FLAGS})

    if (${BLAS_INCLUDE_DIRS} MATCHES "mkl" AND (${GGML_BLAS_VENDOR} MATCHES "Generic" OR ${GGML_BLAS_VENDOR} MATCHES "Intel"))
        add_compile_definitions(GGML_BLAS_USE_MKL)
    endif()

    target_link_libraries     (ggml-blas PRIVATE ${BLAS_LIBRARIES})
    target_include_directories(ggml-blas PRIVATE ${BLAS_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "BLAS not found, please refer to "
                        "https://cmake.org/cmake/help/latest/module/FindBLAS.html#blas-lapack-vendors"
                        " to set correct GGML_BLAS_VENDOR")
endif()
