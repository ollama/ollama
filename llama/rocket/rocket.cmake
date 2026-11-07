# Rockchip NPU (`rocket` + `rknpu`) GGML backend integration for Ollama.
#
# Builds `libggml-rocket.so` against the in-tree `ggml` / `ggml-base` targets from
# `llama.cpp`, linking a static `rocketnpu` library that includes both:
#   - the mainline Linux DRM-accel `rocket` submit provider (`rocket-userspace`)
#   - the Rockchip BSP DRM `rknpu` (>= 0.9.6) submit provider (`rknpu-submit`)
# selected automatically at runtime by `llama/rocket/rocket_dual_provider.c`.

set(_ggml_rocket_default OFF)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT OLLAMA_RUNNER_DIR)
    set(_ggml_rocket_default ON)
endif()
option(GGML_ROCKET "ggml: use Rockchip NPU (rocket / rknpu)" ${_ggml_rocket_default})

if(GGML_ROCKET)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(WARNING "GGML_ROCKET is only supported on Linux; skipping ggml-rocket target")
    else()
        include(FetchContent)
        find_package(Threads REQUIRED)

        set(GGML_ROCKET_GIT_TAG "f995b86c3e8083db20bedd93624783f336485412" CACHE STRING "Pinned ggml-rocket commit")
        set(ROCKET_USERSPACE_GIT_TAG "88cae6d37367bcf65e467524a0a15b89dec01d0d" CACHE STRING "Pinned rocket-userspace commit")
        set(RKNPU_SUBMIT_GIT_TAG "0d1fa72d28b0112c918c7865b18904e6fa140752" CACHE STRING "Pinned rknpu-submit commit")

        if(DEFINED ENV{OLLAMA_GGML_ROCKET_SOURCE})
            get_filename_component(_ggml_rocket_src "$ENV{OLLAMA_GGML_ROCKET_SOURCE}" ABSOLUTE BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
            set(FETCHCONTENT_SOURCE_DIR_GGML_ROCKET "${_ggml_rocket_src}" CACHE PATH "" FORCE)
            message(STATUS "Using local ggml-rocket source: ${_ggml_rocket_src}")
        endif()
        if(DEFINED ENV{OLLAMA_ROCKET_USERSPACE_SOURCE})
            get_filename_component(_rocket_userspace_src "$ENV{OLLAMA_ROCKET_USERSPACE_SOURCE}" ABSOLUTE BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
            set(FETCHCONTENT_SOURCE_DIR_ROCKET_USERSPACE "${_rocket_userspace_src}" CACHE PATH "" FORCE)
            message(STATUS "Using local rocket-userspace source: ${_rocket_userspace_src}")
        endif()
        if(DEFINED ENV{OLLAMA_RKNPU_SUBMIT_SOURCE})
            get_filename_component(_rknpu_submit_src "$ENV{OLLAMA_RKNPU_SUBMIT_SOURCE}" ABSOLUTE BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
            set(FETCHCONTENT_SOURCE_DIR_RKNPU_SUBMIT "${_rknpu_submit_src}" CACHE PATH "" FORCE)
            message(STATUS "Using local rknpu-submit source: ${_rknpu_submit_src}")
        endif()

        FetchContent_Declare(
            rocket_userspace
            GIT_REPOSITORY "https://github.com/gregordinary/rocket-userspace.git"
            GIT_TAG ${ROCKET_USERSPACE_GIT_TAG}
            SOURCE_SUBDIR __no_cmake_subdir__
        )
        FetchContent_Declare(
            rknpu_submit
            GIT_REPOSITORY "https://github.com/gregordinary/rknpu-submit.git"
            GIT_TAG ${RKNPU_SUBMIT_GIT_TAG}
            SOURCE_SUBDIR __no_cmake_subdir__
        )
        FetchContent_Declare(
            ggml_rocket
            GIT_REPOSITORY "https://github.com/gregordinary/ggml-rocket.git"
            GIT_TAG ${GGML_ROCKET_GIT_TAG}
            SOURCE_SUBDIR __no_cmake_subdir__
        )
        FetchContent_MakeAvailable(rocket_userspace rknpu_submit ggml_rocket)

        set(_rocket_dir "${CMAKE_CURRENT_LIST_DIR}")
        set(_ru_dir "${rocket_userspace_SOURCE_DIR}")
        set(_rs_dir "${rknpu_submit_SOURCE_DIR}")
        set(_gr_dir "${ggml_rocket_SOURCE_DIR}")

        set(_rocketnpu_sources
            "${_ru_dir}/src/rocket_log.c"
            "${_ru_dir}/src/rocket_sysfs.c"
            "${_ru_dir}/src/rocket_busy_poll.c"
            "${_ru_dir}/src/rocket_fence_watch.c"
            "${_ru_dir}/src/rocket_hw_profile.c"
            "${_ru_dir}/src/npu_regcmd.c"
            "${_ru_dir}/src/npu_regcmd_rk3576.c"
            "${_ru_dir}/src/rocket_chain.c"
            "${_ru_dir}/src/rocket_fanout.c"
            "${_ru_dir}/src/rocket_op.c"
            "${_ru_dir}/src/rocket_matmul.c"
            "${_ru_dir}/src/rocket_matmul_rk3576.c"
            "${_ru_dir}/src/rocket_rk3576_cube_pack.c"
            "${_ru_dir}/src/rocket_conv2d_rk3576.c"
            "${_ru_dir}/src/rocket_conv.c"
            "${_ru_dir}/src/rocket_conv_transpose.c"
            "${_ru_dir}/src/rocket_conv_resident.c"
            "${_ru_dir}/src/rocket_resize.c"
            "${_ru_dir}/src/rocket_pool.c"
            "${_ru_dir}/src/rocket_pool_rk3576.c"
            "${_ru_dir}/src/rocket_lut_rk3576.c"
            "${_ru_dir}/src/rocket_reduce.c"
            "${_ru_dir}/src/rocket_norm.c"
            "${_ru_dir}/src/rocket_normvision.c"
            "${_ru_dir}/src/rocket_ffn.c"
            "${_ru_dir}/src/rocket_softmax.c"
            "${_ru_dir}/src/rocket_attn.c"
            "${_ru_dir}/src/rocket_encoder.c"
            "${_ru_dir}/src/rocket_siglip_encoder.c"
            "${_ru_dir}/src/rocket_sam_encoder.c"
            "${_ru_dir}/src/rocket_modernbert_encoder.c"
            "${_ru_dir}/src/rocket_activation.c"
            "${_ru_dir}/src/rocket_matmul_mt.c"
            "${_ru_dir}/src/rocket_bf16_stream.c"
            "${_ru_dir}/src/rocket_prepacked.c"
            "${_ru_dir}/src/rocket_prepacked_int8.c"
            "${_ru_dir}/src/rocket_prepacked_int4.c"
            "${_ru_dir}/src/rocket_affinity.c"
            "${_rocket_dir}/rocket_dual_provider.c"
        )

        add_library(rocketnpu STATIC ${_rocketnpu_sources})
        add_library(rocketnpu::rocketnpu ALIAS rocketnpu)
        set_target_properties(rocketnpu PROPERTIES
            C_STANDARD 11
            C_STANDARD_REQUIRED ON
            POSITION_INDEPENDENT_CODE ON
        )
        target_include_directories(rocketnpu
            PUBLIC
                "${_ru_dir}/include"
                "${_rs_dir}/include"
            PRIVATE
                "${_ru_dir}/src"
                "${_rs_dir}/src"
                "${_rocket_dir}/include"
        )
        target_compile_options(rocketnpu PRIVATE -O2 -Wall -Wextra)
        target_link_libraries(rocketnpu PUBLIC Threads::Threads m)

        add_library(ggml-rocket SHARED "${_rocket_dir}/ggml-rocket-ollama.cpp")
        set_target_properties(ggml-rocket PROPERTIES
            CXX_STANDARD 20
            CXX_STANDARD_REQUIRED ON
            POSITION_INDEPENDENT_CODE ON
            PREFIX "lib"
            OUTPUT_NAME "ggml-rocket"
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
            BUILD_RPATH "$ORIGIN;$ORIGIN/.."
            INSTALL_RPATH "$ORIGIN;$ORIGIN/.."
        )
        target_compile_definitions(ggml-rocket PRIVATE GGML_BACKEND_DL)
        target_include_directories(ggml-rocket PRIVATE
            "${_gr_dir}"
            "${llama_cpp_SOURCE_DIR}/ggml/include"
            "${llama_cpp_SOURCE_DIR}/ggml/src"
        )
        target_compile_options(ggml-rocket PRIVATE -Wall -Wextra)

        option(GGML_ROCKET_NATIVE_FP16
            "Build ggml-rocket convert kernels for armv8.2-a+fp16 (RK3588) baseline on arm64"
            ON)
        if(GGML_ROCKET_NATIVE_FP16 AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
            include(CheckCXXCompilerFlag)
            check_cxx_compiler_flag("-march=armv8.2-a+fp16" GGML_ROCKET_HAS_ARMV82_FP16)
            if(GGML_ROCKET_HAS_ARMV82_FP16)
                target_compile_options(ggml-rocket PRIVATE -march=armv8.2-a+fp16)
            endif()
        endif()

        target_link_libraries(ggml-rocket PRIVATE ggml ggml-base rocketnpu::rocketnpu)
        if(TARGET ggml-cpu)
            get_target_property(_ggml_cpu_type ggml-cpu TYPE)
            if(NOT _ggml_cpu_type STREQUAL "MODULE_LIBRARY")
                target_link_libraries(ggml-rocket PRIVATE ggml-cpu)
            endif()
        endif()
    endif()
endif()
