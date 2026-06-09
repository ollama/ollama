# Local Ollama superbuild targets.
#
# This file keeps the repository-root CMake project focused on orchestration:
# it builds a runnable local Ollama payload by delegating llama.cpp work to the
# llama/server CMake project and building the Go binary into a matching layout.

include(ExternalProject)

set(OLLAMA_LLAMA_BACKENDS "" CACHE STRING
    "Semicolon-separated llama-server GPU backends to build: cuda_v12;cuda_v13;rocm_v7_1;rocm_v7_2;vulkan;cuda_jetpack5;cuda_jetpack6")
set(_ollama_mlx_backends_doc "Semicolon-separated MLX backends to build: cuda_v13;metal_v3;metal_v4")
set(OLLAMA_VERSION "0.0.0" CACHE STRING "Ollama version embedded in the local Go binary")
set(OLLAMA_PAYLOAD_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH
    "Build-time staging prefix for nested Ollama native payloads")

string(REGEX REPLACE "^v" "" OLLAMA_VERSION "${OLLAMA_VERSION}")

set(OLLAMA_NATIVE_CONFIG_ARG)
if(CMAKE_CONFIGURATION_TYPES)
    set(OLLAMA_NATIVE_CONFIG_ARG --config Release)
endif()

set(OLLAMA_NATIVE_EXTERNAL_OPTIONS)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.28)
    list(APPEND OLLAMA_NATIVE_EXTERNAL_OPTIONS BUILD_JOB_SERVER_AWARE TRUE)
endif()

function(ollama_check_metal_toolchain output_version)
    find_program(_ollama_xcrun xcrun)
    if(NOT _ollama_xcrun)
        message(FATAL_ERROR
            "MLX Metal requires Xcode command line tools. Install Xcode, run "
            "`sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`, "
            "then install the Metal toolchain with "
            "`xcodebuild -downloadComponent MetalToolchain`.")
    endif()

    execute_process(
        COMMAND zsh "-c"
            "echo \"__METAL_VERSION__\" | \"${_ollama_xcrun}\" -sdk macosx metal -E -x metal -P - 2>/dev/null | tail -1 | tr -d '\n'"
        OUTPUT_VARIABLE _metal_version
        RESULT_VARIABLE _metal_result)
    if(NOT _metal_result EQUAL 0 OR NOT _metal_version MATCHES "^[0-9]+$")
        message(FATAL_ERROR
            "MLX Metal requires Xcode's Metal toolchain. Install Xcode, run "
            "`sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`, "
            "then install the Metal toolchain with "
            "`xcodebuild -downloadComponent MetalToolchain`.")
    endif()

    set(${output_version} "${_metal_version}" PARENT_SCOPE)
endfunction()

function(ollama_macos_major_version output)
    execute_process(
        COMMAND sw_vers -productVersion
        OUTPUT_VARIABLE _macos_version
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _macos_result
        ERROR_QUIET)
    if(_macos_result EQUAL 0)
        string(REGEX MATCH "^[0-9]+" _macos_major "${_macos_version}")
    endif()
    set(${output} "${_macos_major}" PARENT_SCOPE)
endfunction()

function(ollama_macos_sdk_major_version output)
    execute_process(
        COMMAND xcrun --sdk macosx --show-sdk-version
        OUTPUT_VARIABLE _sdk_version
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _sdk_result
        ERROR_QUIET)
    if(_sdk_result EQUAL 0)
        string(REGEX MATCH "^[0-9]+" _sdk_major "${_sdk_version}")
    endif()
    set(${output} "${_sdk_major}" PARENT_SCOPE)
endfunction()

function(ollama_default_mlx_backends output)
    set(_backends "")
    if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        ollama_check_metal_toolchain(_metal_version)
        ollama_macos_major_version(_macos_major)
        ollama_macos_sdk_major_version(_sdk_major)
        if(_macos_major AND _sdk_major AND _macos_major GREATER_EQUAL 26 AND _sdk_major GREATER_EQUAL 26)
            set(_backends "metal_v4")
        else()
            set(_backends "metal_v3")
        endif()
        message(STATUS "Defaulting OLLAMA_MLX_BACKENDS=${_backends} for macOS arm64")
    endif()
    set(${output} "${_backends}" PARENT_SCOPE)
endfunction()

if(NOT DEFINED OLLAMA_MLX_BACKENDS)
    ollama_default_mlx_backends(_ollama_default_mlx_backends)
    set(OLLAMA_MLX_BACKENDS "${_ollama_default_mlx_backends}" CACHE STRING "${_ollama_mlx_backends_doc}")
else()
    set(OLLAMA_MLX_BACKENDS "${OLLAMA_MLX_BACKENDS}" CACHE STRING "${_ollama_mlx_backends_doc}")
endif()

if(NOT OLLAMA_HAVE_LLAMA_SERVER)
    if(OLLAMA_LLAMA_BACKENDS)
        message(FATAL_ERROR "llama/server is required when OLLAMA_LLAMA_BACKENDS is set")
    endif()
    if(NOT OLLAMA_MLX_BACKENDS)
        message(FATAL_ERROR "llama/server is required for local Ollama builds")
    endif()
else()
    file(READ "${CMAKE_SOURCE_DIR}/LLAMA_CPP_VERSION" OLLAMA_LLAMA_CPP_GIT_TAG)
    string(STRIP "${OLLAMA_LLAMA_CPP_GIT_TAG}" OLLAMA_LLAMA_CPP_GIT_TAG)
    include(${CMAKE_SOURCE_DIR}/llama/compat/compat.cmake)
    if(DEFINED FETCHCONTENT_SOURCE_DIR_LLAMA_CPP AND NOT "${FETCHCONTENT_SOURCE_DIR_LLAMA_CPP}" STREQUAL "")
        get_filename_component(OLLAMA_LLAMA_CPP_SOURCE_DIR
            "${FETCHCONTENT_SOURCE_DIR_LLAMA_CPP}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
        message(STATUS "Using llama.cpp source override: ${OLLAMA_LLAMA_CPP_SOURCE_DIR}")
        add_custom_target(ollama-llama-cpp-source)
    elseif(DEFINED ENV{OLLAMA_LLAMA_CPP_SOURCE})
        get_filename_component(OLLAMA_LLAMA_CPP_SOURCE_DIR
            "$ENV{OLLAMA_LLAMA_CPP_SOURCE}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
        message(STATUS "Using local llama.cpp source: ${OLLAMA_LLAMA_CPP_SOURCE_DIR}")
        add_custom_target(ollama-llama-cpp-source)
    else()
        set(OLLAMA_LLAMA_CPP_SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/llama_cpp-src")
        ExternalProject_Add(ollama-llama-cpp-source
            GIT_REPOSITORY "https://github.com/ggml-org/llama.cpp.git"
            GIT_TAG ${OLLAMA_LLAMA_CPP_GIT_TAG}
            GIT_SHALLOW TRUE
            SOURCE_DIR ${OLLAMA_LLAMA_CPP_SOURCE_DIR}
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
            PATCH_COMMAND ${OLLAMA_LLAMA_CPP_COMPAT_PATCH_COMMAND}
            USES_TERMINAL_DOWNLOAD TRUE
            USES_TERMINAL_PATCH TRUE)
    endif()
endif()

set(_mlx_source_targets)
if(OLLAMA_MLX_BACKENDS)
    file(READ "${CMAKE_SOURCE_DIR}/MLX_VERSION" OLLAMA_MLX_GIT_TAG)
    string(STRIP "${OLLAMA_MLX_GIT_TAG}" OLLAMA_MLX_GIT_TAG)
    file(READ "${CMAKE_SOURCE_DIR}/MLX_C_VERSION" OLLAMA_MLX_C_GIT_TAG)
    string(STRIP "${OLLAMA_MLX_C_GIT_TAG}" OLLAMA_MLX_C_GIT_TAG)

    if(DEFINED FETCHCONTENT_SOURCE_DIR_MLX AND NOT "${FETCHCONTENT_SOURCE_DIR_MLX}" STREQUAL "")
        get_filename_component(OLLAMA_MLX_SOURCE_DIR
            "${FETCHCONTENT_SOURCE_DIR_MLX}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
        message(STATUS "Using MLX source override: ${OLLAMA_MLX_SOURCE_DIR}")
    elseif(DEFINED ENV{OLLAMA_MLX_SOURCE})
        get_filename_component(OLLAMA_MLX_SOURCE_DIR
            "$ENV{OLLAMA_MLX_SOURCE}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
        message(STATUS "Using local MLX source: ${OLLAMA_MLX_SOURCE_DIR}")
    else()
        set(OLLAMA_MLX_SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/mlx-src")
        ExternalProject_Add(ollama-mlx-source
            GIT_REPOSITORY "https://github.com/ml-explore/mlx.git"
            GIT_TAG ${OLLAMA_MLX_GIT_TAG}
            # MLX uses commit hashes while we track closely; switch to shallow when MLX pins move to tags.
            GIT_SHALLOW FALSE
            SOURCE_DIR ${OLLAMA_MLX_SOURCE_DIR}
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
            USES_TERMINAL_DOWNLOAD TRUE)
        list(APPEND _mlx_source_targets ollama-mlx-source)
    endif()

    if(DEFINED "FETCHCONTENT_SOURCE_DIR_MLX-C" AND NOT "${FETCHCONTENT_SOURCE_DIR_MLX-C}" STREQUAL "")
        get_filename_component(OLLAMA_MLX_C_SOURCE_DIR
            "${FETCHCONTENT_SOURCE_DIR_MLX-C}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
        message(STATUS "Using MLX-C source override: ${OLLAMA_MLX_C_SOURCE_DIR}")
    elseif(DEFINED ENV{OLLAMA_MLX_C_SOURCE})
        get_filename_component(OLLAMA_MLX_C_SOURCE_DIR
            "$ENV{OLLAMA_MLX_C_SOURCE}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
        message(STATUS "Using local MLX-C source: ${OLLAMA_MLX_C_SOURCE_DIR}")
    else()
        set(OLLAMA_MLX_C_SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/mlx-c-src")
        ExternalProject_Add(ollama-mlx-c-source
            GIT_REPOSITORY "https://github.com/ml-explore/mlx-c.git"
            GIT_TAG ${OLLAMA_MLX_C_GIT_TAG}
            # MLX-C uses commit hashes while we track closely; switch to shallow when MLX-C pins move to tags.
            GIT_SHALLOW FALSE
            SOURCE_DIR ${OLLAMA_MLX_C_SOURCE_DIR}
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
            USES_TERMINAL_DOWNLOAD TRUE)
        list(APPEND _mlx_source_targets ollama-mlx-c-source)
    endif()
    add_custom_target(ollama-mlx-sources DEPENDS ${_mlx_source_targets})
endif()

set(OLLAMA_NATIVE_BUILD_TOOL_COMMAND
    ${CMAKE_COMMAND} --build <BINARY_DIR>)
set(OLLAMA_NATIVE_BUILD_TARGET_ARG --target)
if(CMAKE_GENERATOR MATCHES "Makefiles")
    set(OLLAMA_NATIVE_BUILD_TOOL_COMMAND
        "$(MAKE)" -C <BINARY_DIR>)
    set(OLLAMA_NATIVE_BUILD_TARGET_ARG)
endif()

function(ollama_escape_cmake_list input output)
    string(REPLACE ";" "|" _escaped "${input}")
    set(${output} "${_escaped}" PARENT_SCOPE)
endfunction()

function(ollama_collect_cache_args_with_prefix prefix output)
    get_cmake_property(_cache_variables CACHE_VARIABLES)
    list(SORT _cache_variables)

    set(_args)
    foreach(_var IN LISTS _cache_variables)
        if(_var MATCHES "^${prefix}")
            ollama_escape_cmake_list("${${_var}}" _value)
            list(APPEND _args "-D${_var}=${_value}")
        endif()
    endforeach()

    set(${output} "${_args}" PARENT_SCOPE)
endfunction()

function(ollama_append_cache_arg_if_set output name)
    if(DEFINED ${name} AND NOT "${${name}}" STREQUAL "")
        ollama_escape_cmake_list("${${name}}" _value)
        set(${output} ${${output}} "-D${name}=${_value}" PARENT_SCOPE)
    endif()
endfunction()

function(ollama_cache_arg_is_set name output)
    if(DEFINED ${name} AND NOT "${${name}}" STREQUAL "")
        set(${output} TRUE PARENT_SCOPE)
    else()
        set(${output} FALSE PARENT_SCOPE)
    endif()
endfunction()

function(ollama_llama_cuda_preset backend output)
    ollama_cache_arg_is_set(CMAKE_CUDA_ARCHITECTURES _has_cuda_arch)
    if(_has_cuda_arch)
        set(_preset "llama_${backend}_user_arch")
    elseif(WIN32)
        set(_preset "llama_${backend}_windows")
    else()
        set(_preset "llama_${backend}_linux")
    endif()
    set(${output} "${_preset}" PARENT_SCOPE)
endfunction()

function(ollama_mlx_cuda_preset output)
    ollama_cache_arg_is_set(MLX_CUDA_ARCHITECTURES _has_mlx_arch)
    ollama_cache_arg_is_set(CMAKE_CUDA_ARCHITECTURES _has_cuda_arch)
    if(_has_mlx_arch OR _has_cuda_arch)
        set(_preset "mlx_cuda_v13_user_arch")
    elseif(WIN32)
        set(_preset "mlx_cuda_v13_windows")
    else()
        set(_preset "mlx_cuda_v13_linux")
    endif()
    set(${output} "${_preset}" PARENT_SCOPE)
endfunction()

function(ollama_rocm_preset backend output)
    ollama_cache_arg_is_set(AMDGPU_TARGETS _has_amdgpu_targets)
    ollama_cache_arg_is_set(CMAKE_HIP_ARCHITECTURES _has_hip_arch)
    if(_has_amdgpu_targets OR _has_hip_arch)
        if(backend STREQUAL "rocm_v7_1" AND NOT WIN32)
            message(FATAL_ERROR "OLLAMA_LLAMA_BACKENDS=rocm_v7_1 is only supported for Windows ROCm builds")
        elseif(backend STREQUAL "rocm_v7_2" AND WIN32)
            message(FATAL_ERROR "OLLAMA_LLAMA_BACKENDS=rocm_v7_2 is only supported for Linux ROCm builds")
        endif()
    elseif(backend STREQUAL "rocm_v7_1")
        if(NOT WIN32)
            message(FATAL_ERROR "OLLAMA_LLAMA_BACKENDS=rocm_v7_1 is only supported for Windows ROCm builds")
        endif()
        set(_preset "${backend}_windows")
    elseif(backend STREQUAL "rocm_v7_2")
        if(WIN32)
            message(FATAL_ERROR "OLLAMA_LLAMA_BACKENDS=rocm_v7_2 is only supported for Linux ROCm builds")
        endif()
        set(_preset "${backend}_linux")
    else()
        message(FATAL_ERROR "Unknown ROCm backend '${backend}'")
    endif()
    if(_has_amdgpu_targets OR _has_hip_arch)
        set(_preset "${backend}_user_arch")
    endif()
    set(${output} "${_preset}" PARENT_SCOPE)
endfunction()

function(ollama_add_llama_server_build name)
    cmake_parse_arguments(ARG "" "PRESET;RUNNER_DIR" "TARGETS;CMAKE_ARGS" ${ARGN})
    if(NOT ARG_TARGETS)
        message(FATAL_ERROR "ollama_add_llama_server_build(${name}) requires TARGETS")
    endif()

    if(WIN32 AND name STREQUAL "vulkan")
        # The Vulkan shader generator nests deeply enough to hit Windows MAX_PATH.
        set(_build_dir ${CMAKE_BINARY_DIR}/ls-vk)
    else()
        set(_build_dir ${CMAKE_BINARY_DIR}/llama-server-${name})
    endif()
    ollama_collect_cache_args_with_prefix("GGML_" _ggml_cache_args)
    ollama_collect_cache_args_with_prefix("LLAMA_" _llama_cache_args)
    set(_cmake_args
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_INSTALL_PREFIX=${OLLAMA_PAYLOAD_INSTALL_PREFIX}
        -DOLLAMA_LIB_DIR:STRING=${OLLAMA_LIB_DIR}
        -DOLLAMA_RUNNER_DIR=${ARG_RUNNER_DIR}
        -DFETCHCONTENT_SOURCE_DIR_LLAMA_CPP=${OLLAMA_LLAMA_CPP_SOURCE_DIR}
        -DOLLAMA_LLAMA_CPP_SKIP_COMPAT_PATCH=ON
        -DGGML_NATIVE=OFF
        -DGGML_OPENMP=OFF
        ${ARG_CMAKE_ARGS}
        ${_ggml_cache_args}
        ${_llama_cache_args}
    )

    if(APPLE)
        if(CMAKE_OSX_ARCHITECTURES)
            list(APPEND _cmake_args
                -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES})
        endif()
        if(CMAKE_OSX_DEPLOYMENT_TARGET)
            list(APPEND _cmake_args
                -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET})
        endif()
    endif()
    set(_configure_command ${CMAKE_COMMAND}
        -S ${CMAKE_SOURCE_DIR}/llama/server
        -B <BINARY_DIR>
        ${_cmake_args})
    if(ARG_PRESET)
        set(_configure_command ${CMAKE_COMMAND}
            -S ${CMAKE_SOURCE_DIR}/llama/server
            --preset ${ARG_PRESET}
            -B <BINARY_DIR>
            ${_cmake_args})
    endif()
    ExternalProject_Add(ollama-llama-server-${name}
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/llama/server
        BINARY_DIR ${_build_dir}
        CONFIGURE_COMMAND ${_configure_command}
        BUILD_COMMAND ${OLLAMA_NATIVE_BUILD_TOOL_COMMAND}
            ${OLLAMA_NATIVE_CONFIG_ARG}
            ${OLLAMA_NATIVE_BUILD_TARGET_ARG} ${ARG_TARGETS}
        INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR>
            ${OLLAMA_NATIVE_CONFIG_ARG}
            --component llama-server
        DEPENDS ollama-llama-cpp-source
        LIST_SEPARATOR |
        # ExternalProject cannot reliably infer when nested FetchContent
        # sources, compat patches, or forwarded GGML/LLAMA cache settings need
        # a rebuild. Always entering the sub-build keeps direct `cmake --build`
        # iteration correct; the nested generator still performs incremental
        # compilation.
        BUILD_ALWAYS TRUE
        ${OLLAMA_NATIVE_EXTERNAL_OPTIONS}
        USES_TERMINAL_CONFIGURE TRUE
        USES_TERMINAL_BUILD TRUE
        USES_TERMINAL_INSTALL TRUE)
endfunction()

function(ollama_add_mlx_build name)
    cmake_parse_arguments(ARG "" "PRESET;RUNNER_DIR" "CMAKE_ARGS" ${ARGN})
    if(NOT ARG_RUNNER_DIR)
        message(FATAL_ERROR "ollama_add_mlx_build(${name}) requires RUNNER_DIR")
    endif()

    set(_build_dir ${CMAKE_BINARY_DIR}/${ARG_RUNNER_DIR})
    ollama_collect_cache_args_with_prefix("MLX_" _mlx_cache_args)
    set(_cmake_args
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_INSTALL_PREFIX=${OLLAMA_PAYLOAD_INSTALL_PREFIX}
        -DOLLAMA_LIB_DIR:STRING=${OLLAMA_LIB_DIR}
        -DOLLAMA_RUNNER_DIR=${ARG_RUNNER_DIR}
        -DOLLAMA_SOURCE_DIR=${CMAKE_SOURCE_DIR}
        -DFETCHCONTENT_SOURCE_DIR_MLX=${OLLAMA_MLX_SOURCE_DIR}
        -DFETCHCONTENT_SOURCE_DIR_MLX-C=${OLLAMA_MLX_C_SOURCE_DIR}
        -DOLLAMA_MLX_GENERATE_WRAPPERS=OFF
        ${ARG_CMAKE_ARGS}
        ${_mlx_cache_args}
    )
    foreach(_arg IN ITEMS
            BLAS_INCLUDE_DIRS
            LAPACK_INCLUDE_DIRS
            CUDAToolkit_ROOT
            CUDNN_ROOT_DIR
            CUDNN_INCLUDE_PATH
            CUDNN_LIBRARY_PATH
            CMAKE_CUDA_COMPILER
            CMAKE_CUDA_HOST_COMPILER
            CMAKE_INCLUDE_PATH
            CMAKE_LIBRARY_PATH
            CMAKE_PREFIX_PATH)
        ollama_append_cache_arg_if_set(_cmake_args ${_arg})
    endforeach()

    if(APPLE)
        if(CMAKE_OSX_ARCHITECTURES)
            list(APPEND _cmake_args
                -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES})
        endif()
    endif()
    set(_configure_command ${CMAKE_COMMAND}
        -S ${CMAKE_SOURCE_DIR}/cmake/mlx
        -B <BINARY_DIR>
        ${_cmake_args})
    if(ARG_PRESET)
        set(_configure_command ${CMAKE_COMMAND}
            -S ${CMAKE_SOURCE_DIR}/cmake/mlx
            --preset ${ARG_PRESET}
            -B <BINARY_DIR>
            ${_cmake_args})
    endif()

    ExternalProject_Add(ollama-mlx-${name}
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/cmake/mlx
        BINARY_DIR ${_build_dir}
        CONFIGURE_COMMAND ${_configure_command}
        BUILD_COMMAND ${OLLAMA_NATIVE_BUILD_TOOL_COMMAND}
            ${OLLAMA_NATIVE_CONFIG_ARG}
            ${OLLAMA_NATIVE_BUILD_TARGET_ARG} mlx
            ${OLLAMA_NATIVE_BUILD_TARGET_ARG} mlxc
        INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR>
            ${OLLAMA_NATIVE_CONFIG_ARG}
            --component MLX
            COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR>
            ${OLLAMA_NATIVE_CONFIG_ARG}
            --component MLX_VENDOR
        DEPENDS ollama-mlx-sources
        LIST_SEPARATOR |
        BUILD_ALWAYS TRUE
        ${OLLAMA_NATIVE_EXTERNAL_OPTIONS}
        USES_TERMINAL_CONFIGURE TRUE
        USES_TERMINAL_BUILD TRUE
        USES_TERMINAL_INSTALL TRUE)
endfunction()

find_program(GO_EXECUTABLE go)

if(OLLAMA_MLX_BACKENDS)
    set(_mlx_c_headers_dir "${OLLAMA_MLX_C_SOURCE_DIR}/mlx/c")
    set(_mlx_c_headers_dest "${CMAKE_SOURCE_DIR}/x/mlxrunner/mlx/include/mlx/c")

    if(GO_EXECUTABLE AND (NOT APPLE OR CMAKE_SYSTEM_PROCESSOR STREQUAL CMAKE_HOST_SYSTEM_PROCESSOR))
        add_custom_target(ollama-mlx-generate-wrappers
            COMMAND ${CMAKE_COMMAND}
                -DMLX_C_HEADERS_DIR=${_mlx_c_headers_dir}
                -DMLX_C_HEADERS_DEST=${_mlx_c_headers_dest}
                -P "${CMAKE_SOURCE_DIR}/cmake/vendor-mlx-c-headers.cmake"
            COMMAND ${CMAKE_COMMAND} -E env
                CC= CGO_CFLAGS= CGO_CXXFLAGS=
                ${GO_EXECUTABLE} generate ./x/...
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            DEPENDS ollama-mlx-sources
            COMMENT "Regenerating MLX Go wrappers"
            VERBATIM)
    else()
        add_custom_target(ollama-mlx-generate-wrappers
            COMMAND ${CMAKE_COMMAND} -E echo
                "Cannot regenerate MLX wrappers while Go is unavailable or while cross-compiling"
            COMMAND ${CMAKE_COMMAND} -E false
            DEPENDS ollama-mlx-sources
            VERBATIM)
    endif()
endif()

if(OLLAMA_HAVE_LLAMA_SERVER)
    if(NOT OLLAMA_GO_OUTPUT)
        if(WIN32)
            set(OLLAMA_GO_OUTPUT ${CMAKE_SOURCE_DIR}/ollama.exe)
        else()
            set(OLLAMA_GO_OUTPUT ${CMAKE_SOURCE_DIR}/ollama)
        endif()
    endif()
    if(NOT IS_ABSOLUTE "${OLLAMA_GO_OUTPUT}")
        set(OLLAMA_GO_OUTPUT "${CMAKE_SOURCE_DIR}/${OLLAMA_GO_OUTPUT}")
    endif()
    get_filename_component(OLLAMA_GO_OUTPUT "${OLLAMA_GO_OUTPUT}" ABSOLUTE)
    set(OLLAMA_GO_OUTPUT "${OLLAMA_GO_OUTPUT}" CACHE FILEPATH "Output path for the local Ollama Go binary")
    get_filename_component(OLLAMA_GO_OUTPUT_DIR "${OLLAMA_GO_OUTPUT}" DIRECTORY)

    set(OLLAMA_GO_LDFLAGS
        "-s -w -X=github.com/ollama/ollama/version.Version=${OLLAMA_VERSION} -X=github.com/ollama/ollama/server.mode=release")
    if(GO_EXECUTABLE)
        add_custom_target(ollama-go ALL
            COMMAND ${CMAKE_COMMAND} -E make_directory "${OLLAMA_GO_OUTPUT_DIR}"
            COMMAND ${CMAKE_COMMAND} -E env CGO_ENABLED=1
                ${GO_EXECUTABLE} build -trimpath -ldflags "${OLLAMA_GO_LDFLAGS}" -o "${OLLAMA_GO_OUTPUT}" .
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            BYPRODUCTS ${OLLAMA_GO_OUTPUT}
            COMMENT "Building Ollama Go binary"
            VERBATIM)
    else()
        add_custom_target(ollama-go ALL
            COMMAND ${CMAKE_COMMAND} -E echo
                "Go executable not found. Install Go or set GO_EXECUTABLE to build the local Ollama binary."
            COMMAND ${CMAKE_COMMAND} -E false
            COMMENT "Building Ollama Go binary"
            VERBATIM)
    endif()

    set(_cpu_args)
    if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        list(APPEND _cpu_args
            -DBUILD_SHARED_LIBS=OFF
            -DGGML_BACKEND_DL=OFF
            -DGGML_METAL=ON
            -DGGML_METAL_EMBED_LIBRARY=ON)
    else()
        list(APPEND _cpu_args
            -DBUILD_SHARED_LIBS=ON
            -DGGML_BACKEND_DL=ON
            -DGGML_CPU_ALL_VARIANTS=ON)
        if(WIN32)
            list(APPEND _cpu_args -DGGML_OPENMP=ON)
        endif()
        if(APPLE)
            list(APPEND _cpu_args -DGGML_METAL=OFF)
        endif()
    endif()

    ollama_add_llama_server_build(local
        RUNNER_DIR ""
        TARGETS llama-server llama-quantize
        CMAKE_ARGS ${_cpu_args})

    add_custom_target(ollama-local ALL
        DEPENDS ollama-go ollama-llama-server-local
        COMMENT "Building local Ollama payload")

    install(PROGRAMS "${OLLAMA_GO_OUTPUT}"
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT ollama-local)
endif()

set(_backend_targets)
if(OLLAMA_HAVE_LLAMA_SERVER)
    foreach(_backend IN LISTS OLLAMA_LLAMA_BACKENDS)
        if(_backend STREQUAL "cuda_v12")
            ollama_llama_cuda_preset(${_backend} _cuda_preset)
            set(_cuda_args)
            ollama_append_cache_arg_if_set(_cuda_args CMAKE_CUDA_ARCHITECTURES)
            ollama_append_cache_arg_if_set(_cuda_args CMAKE_CUDA_FLAGS)
            ollama_add_llama_server_build(${_backend}
                PRESET ${_cuda_preset}
                RUNNER_DIR ${_backend}
                TARGETS ggml-cuda
                CMAKE_ARGS ${_cuda_args})
            list(APPEND _backend_targets ollama-llama-server-${_backend})
        elseif(_backend STREQUAL "cuda_v13")
            ollama_llama_cuda_preset(${_backend} _cuda_preset)
            set(_cuda_args)
            ollama_append_cache_arg_if_set(_cuda_args CMAKE_CUDA_ARCHITECTURES)
            ollama_append_cache_arg_if_set(_cuda_args CMAKE_CUDA_FLAGS)
            ollama_add_llama_server_build(${_backend}
                PRESET ${_cuda_preset}
                RUNNER_DIR ${_backend}
                TARGETS ggml-cuda
                CMAKE_ARGS ${_cuda_args})
            list(APPEND _backend_targets ollama-llama-server-${_backend})
        elseif(_backend STREQUAL "rocm_v7_1" OR _backend STREQUAL "rocm_v7_2")
            # ROCm 7.1 and 7.2 currently share build settings. Keep the backend
            # names versioned so future packaging can install side-by-side ROCm
            # payloads without changing the superbuild interface.
            ollama_rocm_preset(${_backend} _rocm_preset)
            set(_rocm_args
                -DBUILD_SHARED_LIBS=ON
                -DGGML_BACKEND_DL=ON
                -DGGML_HIP=ON
                -DCMAKE_HIP_PLATFORM=amd
                -DOLLAMA_GPU_BACKEND=hip)
            ollama_append_cache_arg_if_set(_rocm_args AMDGPU_TARGETS)
            ollama_append_cache_arg_if_set(_rocm_args CMAKE_HIP_ARCHITECTURES)
            ollama_append_cache_arg_if_set(_rocm_args CMAKE_HIP_FLAGS)
            ollama_append_cache_arg_if_set(_rocm_args GGML_CUDA_NO_PEER_COPY)
            ollama_append_cache_arg_if_set(_rocm_args CMAKE_PREFIX_PATH)
            ollama_add_llama_server_build(${_backend}
                PRESET ${_rocm_preset}
                RUNNER_DIR ${_backend}
                TARGETS ggml-hip
                CMAKE_ARGS ${_rocm_args})
            list(APPEND _backend_targets ollama-llama-server-${_backend})
        elseif(_backend STREQUAL "vulkan")
            ollama_add_llama_server_build(vulkan
                RUNNER_DIR vulkan
                TARGETS ggml-vulkan
                CMAKE_ARGS
                    -DBUILD_SHARED_LIBS=ON
                    -DGGML_BACKEND_DL=ON
                    -DGGML_VULKAN=ON
                    -DOLLAMA_GPU_BACKEND=vulkan)
            list(APPEND _backend_targets ollama-llama-server-vulkan)
        elseif(_backend STREQUAL "cuda_jetpack5")
            if(CMAKE_CUDA_ARCHITECTURES)
                set(_cuda_preset llama_cuda_jetpack5_user_arch)
            else()
                set(_cuda_preset llama_cuda_jetpack5)
            endif()
            set(_cuda_args)
            ollama_append_cache_arg_if_set(_cuda_args CMAKE_CUDA_ARCHITECTURES)
            ollama_append_cache_arg_if_set(_cuda_args CMAKE_CUDA_FLAGS)
            ollama_add_llama_server_build(${_backend}
                PRESET ${_cuda_preset}
                RUNNER_DIR ${_backend}
                TARGETS ggml-cuda
                CMAKE_ARGS ${_cuda_args})
            list(APPEND _backend_targets ollama-llama-server-${_backend})
        elseif(_backend STREQUAL "cuda_jetpack6")
            if(CMAKE_CUDA_ARCHITECTURES)
                set(_cuda_preset llama_cuda_jetpack6_user_arch)
            else()
                set(_cuda_preset llama_cuda_jetpack6)
            endif()
            set(_cuda_args)
            ollama_append_cache_arg_if_set(_cuda_args CMAKE_CUDA_ARCHITECTURES)
            ollama_append_cache_arg_if_set(_cuda_args CMAKE_CUDA_FLAGS)
            ollama_add_llama_server_build(${_backend}
                PRESET ${_cuda_preset}
                RUNNER_DIR ${_backend}
                TARGETS ggml-cuda
                CMAKE_ARGS ${_cuda_args})
            list(APPEND _backend_targets ollama-llama-server-${_backend})
        else()
            message(FATAL_ERROR
                "Unknown OLLAMA_LLAMA_BACKENDS entry '${_backend}'")
        endif()
    endforeach()
endif()

if(_backend_targets)
    add_custom_target(ollama-llama-server-backends ALL
        DEPENDS ${_backend_targets}
        COMMENT "Building llama-server GPU backends")
endif()

set(_mlx_targets)
foreach(_backend IN LISTS OLLAMA_MLX_BACKENDS)
    if(_backend STREQUAL "cuda_v13")
        ollama_mlx_cuda_preset(_mlx_cuda_preset)
        set(_mlx_cuda_args)
        ollama_append_cache_arg_if_set(_mlx_cuda_args CMAKE_CUDA_ARCHITECTURES)
        ollama_append_cache_arg_if_set(_mlx_cuda_args MLX_CUDA_ARCHITECTURES)
        ollama_append_cache_arg_if_set(_mlx_cuda_args CMAKE_CUDA_FLAGS)
        ollama_add_mlx_build(cuda_v13
            PRESET ${_mlx_cuda_preset}
            RUNNER_DIR mlx_cuda_v13
            CMAKE_ARGS ${_mlx_cuda_args})
        list(APPEND _mlx_targets ollama-mlx-cuda_v13)
    elseif(_backend STREQUAL "metal_v3")
        if(NOT APPLE)
            message(FATAL_ERROR "OLLAMA_MLX_BACKENDS=metal_v3 is only supported on macOS")
        endif()
        ollama_check_metal_toolchain(_metal_version)
        ollama_add_mlx_build(metal_v3
            PRESET mlx_metal_v3
            RUNNER_DIR mlx_metal_v3)
        list(APPEND _mlx_targets ollama-mlx-metal_v3)
    elseif(_backend STREQUAL "metal_v4")
        if(NOT APPLE)
            message(FATAL_ERROR "OLLAMA_MLX_BACKENDS=metal_v4 is only supported on macOS")
        endif()
        ollama_check_metal_toolchain(_metal_version)
        ollama_macos_sdk_major_version(_ollama_mlx_sdk_major)
        if(_ollama_mlx_sdk_major AND _ollama_mlx_sdk_major GREATER_EQUAL 26)
            ollama_add_mlx_build(metal_v4
                PRESET mlx_metal_v4
                RUNNER_DIR mlx_metal_v4)
            list(APPEND _mlx_targets ollama-mlx-metal_v4)
        else()
            message(FATAL_ERROR
                "OLLAMA_MLX_BACKENDS=metal_v4 requires the macOS 26 SDK. "
                "Install a newer Xcode or use OLLAMA_MLX_BACKENDS=metal_v3.")
        endif()
    else()
        message(FATAL_ERROR
            "Unknown OLLAMA_MLX_BACKENDS entry '${_backend}'")
    endif()
endforeach()

if(_mlx_targets)
    add_custom_target(ollama-mlx-backends ALL
        DEPENDS ${_mlx_targets}
        COMMENT "Building MLX backends")
endif()

install(DIRECTORY "${OLLAMA_PAYLOAD_INSTALL_PREFIX}/${OLLAMA_LIB_DIR}/"
    DESTINATION "${OLLAMA_LIB_DIR}"
    COMPONENT ollama-local
    USE_SOURCE_PERMISSIONS)
