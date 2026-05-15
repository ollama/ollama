# Local Ollama superbuild targets.
#
# This file keeps the repository-root CMake project focused on orchestration:
# it builds a runnable local Ollama payload by delegating llama.cpp work to the
# llama/server CMake project and building the Go binary into a matching layout.

include(ExternalProject)

set(OLLAMA_LLAMA_SERVER_BACKENDS "" CACHE STRING
    "Semicolon-separated llama-server GPU backends to build: cuda-v12;cuda-v13;cuda-v13-windows;rocm;rocm-windows;vulkan;jetpack5;jetpack6")
set(OLLAMA_VERSION "0.0.0" CACHE STRING "Ollama version embedded in the local Go binary")

string(REGEX REPLACE "^v" "" OLLAMA_VERSION "${OLLAMA_VERSION}")

set(OLLAMA_LLAMA_SERVER_CONFIG_ARG)
if(CMAKE_CONFIGURATION_TYPES)
    set(OLLAMA_LLAMA_SERVER_CONFIG_ARG --config Release)
endif()

set(OLLAMA_LLAMA_SERVER_EXTERNAL_OPTIONS)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.28)
    list(APPEND OLLAMA_LLAMA_SERVER_EXTERNAL_OPTIONS BUILD_JOB_SERVER_AWARE TRUE)
endif()

function(ollama_escape_cmake_list input output)
    string(REPLACE ";" "|" _escaped "${input}")
    set(${output} "${_escaped}" PARENT_SCOPE)
endfunction()

function(ollama_add_llama_server_build name)
    cmake_parse_arguments(ARG "" "RUNNER_DIR" "TARGETS;CMAKE_ARGS" ${ARGN})
    if(NOT ARG_TARGETS)
        message(FATAL_ERROR "ollama_add_llama_server_build(${name}) requires TARGETS")
    endif()

    if(WIN32 AND name STREQUAL "vulkan")
        # The Vulkan shader generator nests deeply enough to hit Windows MAX_PATH.
        set(_build_dir ${CMAKE_BINARY_DIR}/ls-vk)
    else()
        set(_build_dir ${CMAKE_BINARY_DIR}/llama-server-${name})
    endif()
    set(_cmake_args
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}
        -DOLLAMA_RUNNER_DIR=${ARG_RUNNER_DIR}
        -DGGML_NATIVE=OFF
        -DGGML_OPENMP=OFF
        ${ARG_CMAKE_ARGS}
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
    if(DEFINED FETCHCONTENT_SOURCE_DIR_LLAMA_CPP)
        list(APPEND _cmake_args
            -DFETCHCONTENT_SOURCE_DIR_LLAMA_CPP=${FETCHCONTENT_SOURCE_DIR_LLAMA_CPP})
    endif()

    ExternalProject_Add(ollama-llama-server-${name}
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/llama/server
        BINARY_DIR ${_build_dir}
        CMAKE_ARGS ${_cmake_args}
        BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR>
            ${OLLAMA_LLAMA_SERVER_CONFIG_ARG}
            --target ${ARG_TARGETS}
        INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR>
            ${OLLAMA_LLAMA_SERVER_CONFIG_ARG}
            --component llama-server
        LIST_SEPARATOR |
        BUILD_ALWAYS TRUE
        ${OLLAMA_LLAMA_SERVER_EXTERNAL_OPTIONS}
        USES_TERMINAL_CONFIGURE TRUE
        USES_TERMINAL_BUILD TRUE
        USES_TERMINAL_INSTALL TRUE)
endfunction()

find_program(GO_EXECUTABLE go)

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

set(_backend_targets)
foreach(_backend IN LISTS OLLAMA_LLAMA_SERVER_BACKENDS)
    if(_backend STREQUAL "cuda-v12")
        set(_cuda_arch "${CMAKE_CUDA_ARCHITECTURES}")
        if(NOT _cuda_arch)
            set(_cuda_arch "50-virtual;52-virtual;60-virtual;61-virtual;70;75;80;86;89;90;90a;120")
        endif()
        ollama_escape_cmake_list("${_cuda_arch}" _cuda_arch_arg)
        set(_cuda_flags "${CMAKE_CUDA_FLAGS}")
        if(NOT _cuda_flags)
            set(_cuda_flags "-Wno-deprecated-gpu-targets -t 2")
        endif()
        ollama_add_llama_server_build(cuda-v12
            RUNNER_DIR cuda_v12
            TARGETS ggml-cuda
            CMAKE_ARGS
                -DBUILD_SHARED_LIBS=ON
                -DGGML_BACKEND_DL=ON
                -DGGML_CUDA=ON
                -DCMAKE_CUDA_ARCHITECTURES=${_cuda_arch_arg}
                -DCMAKE_CUDA_FLAGS=${_cuda_flags}
                -DOLLAMA_GPU_BACKEND=cuda)
        list(APPEND _backend_targets ollama-llama-server-cuda-v12)
    elseif(_backend STREQUAL "cuda-v13")
        set(_cuda_arch "${CMAKE_CUDA_ARCHITECTURES}")
        if(NOT _cuda_arch)
            set(_cuda_arch "75-virtual;80-virtual;86-virtual;89-virtual;90-virtual;90a-virtual;100-virtual;103-virtual;110-virtual;120-virtual;121-virtual")
        endif()
        ollama_escape_cmake_list("${_cuda_arch}" _cuda_arch_arg)
        set(_cuda_flags "${CMAKE_CUDA_FLAGS}")
        if(NOT _cuda_flags)
            set(_cuda_flags "-t 4")
        endif()
        ollama_add_llama_server_build(cuda-v13
            RUNNER_DIR cuda_v13
            TARGETS ggml-cuda
            CMAKE_ARGS
                -DBUILD_SHARED_LIBS=ON
                -DGGML_BACKEND_DL=ON
                -DGGML_CUDA=ON
                -DCMAKE_CUDA_ARCHITECTURES=${_cuda_arch_arg}
                -DCMAKE_CUDA_FLAGS=${_cuda_flags}
                -DOLLAMA_GPU_BACKEND=cuda)
        list(APPEND _backend_targets ollama-llama-server-cuda-v13)
    elseif(_backend STREQUAL "cuda-v13-windows")
        set(_cuda_arch "${CMAKE_CUDA_ARCHITECTURES}")
        if(NOT _cuda_arch)
            set(_cuda_arch "75-virtual;89-virtual;100-virtual;120-virtual")
        endif()
        ollama_escape_cmake_list("${_cuda_arch}" _cuda_arch_arg)
        set(_cuda_flags "${CMAKE_CUDA_FLAGS}")
        if(NOT _cuda_flags)
            set(_cuda_flags "-t 4")
        endif()
        ollama_add_llama_server_build(cuda-v13-windows
            RUNNER_DIR cuda_v13
            TARGETS ggml-cuda
            CMAKE_ARGS
                -DBUILD_SHARED_LIBS=ON
                -DGGML_BACKEND_DL=ON
                -DGGML_CUDA=ON
                -DCMAKE_CUDA_ARCHITECTURES=${_cuda_arch_arg}
                -DCMAKE_CUDA_FLAGS=${_cuda_flags}
                -DOLLAMA_GPU_BACKEND=cuda)
        list(APPEND _backend_targets ollama-llama-server-cuda-v13-windows)
    elseif(_backend STREQUAL "rocm" OR _backend STREQUAL "rocm-windows")
        set(_amd_targets "${AMDGPU_TARGETS}")
        if(NOT _amd_targets)
            set(_amd_targets "${CMAKE_HIP_ARCHITECTURES}")
        endif()
        if(NOT _amd_targets)
            if(_backend STREQUAL "rocm-windows")
                set(_amd_targets "gfx942;gfx950;gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1200;gfx1201")
            else()
                set(_amd_targets "gfx942;gfx950;gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1200;gfx1201;gfx908:xnack-;gfx90a:xnack+;gfx90a:xnack-")
            endif()
        endif()
        ollama_escape_cmake_list("${_amd_targets}" _amd_targets_arg)
        set(_hip_flags "${CMAKE_HIP_FLAGS}")
        if(NOT _hip_flags)
            set(_hip_flags "-parallel-jobs=4")
        endif()
        set(_rocm_args
            -DBUILD_SHARED_LIBS=ON
            -DGGML_BACKEND_DL=ON
            -DGGML_HIP=ON
            -DCMAKE_HIP_PLATFORM=amd
            -DAMDGPU_TARGETS=${_amd_targets_arg}
            -DCMAKE_HIP_FLAGS=${_hip_flags}
            -DOLLAMA_GPU_BACKEND=hip)
        if(CMAKE_PREFIX_PATH)
            ollama_escape_cmake_list("${CMAKE_PREFIX_PATH}" _prefix_path_arg)
            list(APPEND _rocm_args -DCMAKE_PREFIX_PATH=${_prefix_path_arg})
        endif()
        if(_backend STREQUAL "rocm-windows")
            list(APPEND _rocm_args
                "-DCMAKE_C_FLAGS=-parallel-jobs=4 -Wno-ignored-attributes -Wno-deprecated-pragma"
                "-DCMAKE_CXX_FLAGS=-parallel-jobs=4 -Wno-ignored-attributes -Wno-deprecated-pragma")
        endif()
        ollama_add_llama_server_build(${_backend}
            RUNNER_DIR rocm
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
    elseif(_backend STREQUAL "jetpack5")
        ollama_add_llama_server_build(jetpack5
            RUNNER_DIR cuda_jetpack5
            TARGETS ggml-cuda
            CMAKE_ARGS
                -DBUILD_SHARED_LIBS=ON
                -DGGML_BACKEND_DL=ON
                -DGGML_CUDA=ON
                -DCMAKE_CUDA_ARCHITECTURES=72\;87
                -DOLLAMA_GPU_BACKEND=cuda)
        list(APPEND _backend_targets ollama-llama-server-jetpack5)
    elseif(_backend STREQUAL "jetpack6")
        ollama_add_llama_server_build(jetpack6
            RUNNER_DIR cuda_jetpack6
            TARGETS ggml-cuda
            CMAKE_ARGS
                -DBUILD_SHARED_LIBS=ON
                -DGGML_BACKEND_DL=ON
                -DGGML_CUDA=ON
                -DCMAKE_CUDA_ARCHITECTURES=87
                -DOLLAMA_GPU_BACKEND=cuda)
        list(APPEND _backend_targets ollama-llama-server-jetpack6)
    else()
        message(FATAL_ERROR
            "Unknown OLLAMA_LLAMA_SERVER_BACKENDS entry '${_backend}'")
    endif()
endforeach()

if(_backend_targets)
    add_custom_target(ollama-llama-server-backends ALL
        DEPENDS ${_backend_targets}
        COMMENT "Building llama-server GPU backends")
endif()

install(DIRECTORY "${CMAKE_BINARY_DIR}/lib/ollama/"
    DESTINATION "lib/ollama"
    COMPONENT ollama-local
    USE_SOURCE_PERMISSIONS)
