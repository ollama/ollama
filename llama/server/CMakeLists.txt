cmake_minimum_required(VERSION 3.24)
project(ollama-llama-server C CXX)

# Handle cross-compilation on macOS: when CMAKE_OSX_ARCHITECTURES is set to a
# single architecture different from the host, make downstream architecture
# detection match the target slice. ggml uses CMAKE_SYSTEM_PROCESSOR to decide
# which CPU backend variants to build.
if(CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_OSX_ARCHITECTURES MATCHES ";")
    if(CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64" AND NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        message(STATUS "Cross-compiling for x86_64: overriding CMAKE_SYSTEM_PROCESSOR from ${CMAKE_SYSTEM_PROCESSOR} to x86_64")
        set(CMAKE_SYSTEM_PROCESSOR "x86_64")
    elseif(CMAKE_OSX_ARCHITECTURES STREQUAL "arm64" AND NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        message(STATUS "Cross-compiling for arm64: overriding CMAKE_SYSTEM_PROCESSOR from ${CMAKE_SYSTEM_PROCESSOR} to arm64")
        set(CMAKE_SYSTEM_PROCESSOR "arm64")
    endif()
endif()

# Ensure all shared libraries and executables can find their dependencies
# in the same directory at runtime (via $ORIGIN on Linux, @loader_path on macOS).
if(NOT APPLE)
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
    set(CMAKE_BUILD_RPATH "$ORIGIN")
endif()

include(FetchContent)

# Read pinned upstream commit from version file (shared with Dockerfile)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../../LLAMA_CPP_VERSION" LLAMA_CPP_GIT_TAG)
string(STRIP "${LLAMA_CPP_GIT_TAG}" LLAMA_CPP_GIT_TAG)

# Allow local source override via environment variable (like OLLAMA_MLX_SOURCE)
if(DEFINED ENV{OLLAMA_LLAMA_CPP_SOURCE})
    get_filename_component(_src "$ENV{OLLAMA_LLAMA_CPP_SOURCE}" ABSOLUTE BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    set(FETCHCONTENT_SOURCE_DIR_LLAMA_CPP "${_src}" CACHE PATH "" FORCE)
    message(STATUS "Using local llama.cpp source: ${_src}")
endif()

# Ollama-compat shim: overlays the fetched llama.cpp source with a tiny
# in-memory translation layer that lets upstream llama-server load GGUFs
# produced by older Ollama versions (e.g. existing ~/.ollama/models/blobs).
# See llama/compat/README.md for details.
#
# The patch only runs when fetching from GitHub — if a local source override
# is active, leave the developer's tree alone (they can apply by hand if
# they want to iterate on the compat layer).
set(_ollama_compat_patch_cmd "")
if(NOT DEFINED ENV{OLLAMA_LLAMA_CPP_SOURCE})
    include(${CMAKE_CURRENT_SOURCE_DIR}/../compat/compat.cmake)
    set(_ollama_compat_patch_cmd PATCH_COMMAND ${OLLAMA_LLAMA_CPP_COMPAT_PATCH_COMMAND})
endif()

# Configure upstream build options BEFORE FetchContent_MakeAvailable.
# When included via FetchContent, llama.cpp sets LLAMA_STANDALONE=OFF
# so all optional builds default to OFF. We explicitly enable what we need.
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_HTML OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_TOOLS_INSTALL OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY "https://github.com/ggml-org/llama.cpp.git"
    GIT_TAG ${LLAMA_CPP_GIT_TAG}
    GIT_SHALLOW TRUE
    ${_ollama_compat_patch_cmd}
)
FetchContent_MakeAvailable(llama_cpp)

# Find GPU toolkits for runtime dependency bundling.
# The upstream llama.cpp build finds these internally, but we need the
# variables (CUDAToolkit_LIBRARY_DIR, etc.) in our install scope.
if(GGML_CUDA)
    find_package(CUDAToolkit)
endif()
if(GGML_HIP)
    find_package(hip)
endif()

# Install layout under lib/ollama/:
#
# CPU build (OLLAMA_RUNNER_DIR=""):
#   lib/ollama/llama-server
#   lib/ollama/libggml-base.so
#   lib/ollama/libggml.so
#   lib/ollama/libllama.so
#   lib/ollama/libggml-cpu*.so  (all CPU variants)
#
# GPU build (OLLAMA_RUNNER_DIR="cuda_v12" etc.):
#   lib/ollama/cuda_v12/libggml-cuda.so  (GPU backend only)

set(_base_dest "lib/ollama")

if(OLLAMA_RUNNER_DIR)
    # GPU backend build — install the GPU backend .so/.dll module.
    # install(CODE) runs at install time to handle both single-config and
    # multi-config generator layouts.
    install(CODE "
        file(GLOB _gpu_backends
            \"${CMAKE_BINARY_DIR}/bin/libggml-${OLLAMA_GPU_BACKEND}*\"
            \"${CMAKE_BINARY_DIR}/bin/Release/ggml-${OLLAMA_GPU_BACKEND}*.dll\"
            \"${CMAKE_BINARY_DIR}/bin/ggml-${OLLAMA_GPU_BACKEND}*.dll\"
        )
        foreach(_f \${_gpu_backends})
            file(INSTALL \${_f} DESTINATION \"\${CMAKE_INSTALL_PREFIX}/${_base_dest}/${OLLAMA_RUNNER_DIR}\")
        endforeach()
    " COMPONENT llama-server)

    # Bundle GPU runtime libraries (cublas, cudart, rocblas, etc.)
    # These are needed at runtime by the GPU backend .so
    if(GGML_CUDA AND CUDAToolkit_FOUND)
        # Find the actual ggml-cuda target to get its runtime dependencies
        if(TARGET ggml-cuda)
            install(TARGETS ggml-cuda
                RUNTIME_DEPENDENCIES
                    DIRECTORIES ${CUDAToolkit_BIN_DIR} ${CUDAToolkit_BIN_DIR}/x64 ${CUDAToolkit_LIBRARY_DIR}
                    PRE_INCLUDE_REGEXES cublas cublasLt cudart
                    PRE_EXCLUDE_REGEXES ".*"
                RUNTIME DESTINATION "${_base_dest}/${OLLAMA_RUNNER_DIR}" COMPONENT llama-server
                LIBRARY DESTINATION "${_base_dest}/${OLLAMA_RUNNER_DIR}" COMPONENT llama-server
            )
        endif()
    endif()
    if(GGML_HIP)
        if(TARGET ggml-hip)
            install(TARGETS ggml-hip
                RUNTIME_DEPENDENCY_SET rocm_deps
                RUNTIME DESTINATION "${_base_dest}/${OLLAMA_RUNNER_DIR}" COMPONENT llama-server
                LIBRARY DESTINATION "${_base_dest}/${OLLAMA_RUNNER_DIR}" COMPONENT llama-server
            )
            install(RUNTIME_DEPENDENCY_SET rocm_deps
                DIRECTORIES ${HIP_BIN_INSTALL_DIR} ${HIP_LIB_INSTALL_DIR}
                PRE_INCLUDE_REGEXES hipblas rocblas amdhip64 rocsolver amd_comgr hsa-runtime64 rocsparse tinfo rocprofiler-register roctx64 rocroller drm drm_amdgpu numa elf
                PRE_EXCLUDE_REGEXES ".*"
                POST_EXCLUDE_REGEXES "system32"
                RUNTIME DESTINATION "${_base_dest}/${OLLAMA_RUNNER_DIR}" COMPONENT llama-server
                LIBRARY DESTINATION "${_base_dest}/${OLLAMA_RUNNER_DIR}" COMPONENT llama-server
            )
            foreach(_hip_dir IN ITEMS ${HIP_BIN_INSTALL_DIR} ${HIP_LIB_INSTALL_DIR})
                if(EXISTS ${_hip_dir}/rocblas)
                    install(DIRECTORY ${_hip_dir}/rocblas
                        DESTINATION "${_base_dest}/${OLLAMA_RUNNER_DIR}"
                        COMPONENT llama-server)
                    break()
                endif()
            endforeach()
        endif()
    endif()
    if(GGML_VULKAN)
        if(TARGET ggml-vulkan)
            install(TARGETS ggml-vulkan
                RUNTIME_DEPENDENCIES
                    PRE_INCLUDE_REGEXES vulkan
                    PRE_EXCLUDE_REGEXES ".*"
                RUNTIME DESTINATION "${_base_dest}/${OLLAMA_RUNNER_DIR}" COMPONENT llama-server
                LIBRARY DESTINATION "${_base_dest}/${OLLAMA_RUNNER_DIR}" COMPONENT llama-server
            )
        endif()
    endif()
else()
    # CPU/base build — install llama-server, llama-quantize + all shared libs + CPU backend modules
    # RUNTIME covers executables and Windows DLLs; LIBRARY covers .so on Linux
    install(TARGETS llama-server llama-quantize
        RUNTIME DESTINATION ${_base_dest} COMPONENT llama-server OPTIONAL)
    install(TARGETS ggml-base ggml llama mtmd
        RUNTIME DESTINATION ${_base_dest} COMPONENT llama-server OPTIONAL
        LIBRARY DESTINATION ${_base_dest} COMPONENT llama-server OPTIONAL)

    # On Windows with /MD, bundle the MSVC CRT DLLs (vcruntime140.dll,
    # msvcp140.dll, vcomp140.dll, etc.) alongside the executables so users
    # don't need the VC++ Redistributable installed separately.
    if(MSVC AND GGML_BACKEND_DL)
        cmake_path(GET CMAKE_CXX_COMPILER PARENT_PATH _msvc_bin_dir)
        # Walk up from the compiler bin dir to find the VC redist directory
        # e.g. .../MSVC/14.44.35207/bin/Hostx64/x64 -> .../MSVC/14.44.35207
        cmake_path(GET _msvc_bin_dir PARENT_PATH _tmp)
        cmake_path(GET _tmp PARENT_PATH _tmp)
        cmake_path(GET _tmp PARENT_PATH _msvc_ver_dir)
        # The redist version may differ from the toolset version, so glob for it
        file(GLOB _vc_redist_dirs "${_msvc_ver_dir}/../../../Redist/MSVC/*/x64/Microsoft.VC143.CRT")
        file(GLOB _vc_openmp_dirs "${_msvc_ver_dir}/../../../Redist/MSVC/*/x64/Microsoft.VC143.OPENMP")
        set(_vc_crt_dlls)
        foreach(_dir ${_vc_redist_dirs} ${_vc_openmp_dirs})
            file(GLOB _dlls "${_dir}/*.dll")
            list(APPEND _vc_crt_dlls ${_dlls})
        endforeach()
        if(_vc_crt_dlls)
            install(FILES ${_vc_crt_dlls}
                DESTINATION ${_base_dest}
                COMPONENT llama-server)
        else()
            message(WARNING "Could not find MSVC CRT redistributable DLLs to bundle")
        endif()
    endif()

    # CPU backend modules (multiple variants from GGML_CPU_ALL_VARIANTS).
    # install(CODE) runs at install time (not configure time) so the glob
    # finds the built files. Handles both single-config (bin/) and
    # multi-config (bin/Release/) generator layouts.
    install(CODE "
        file(GLOB _cpu_backends
            \"${CMAKE_BINARY_DIR}/bin/libggml-cpu*\"
            \"${CMAKE_BINARY_DIR}/bin/libggml-blas*\"
            \"${CMAKE_BINARY_DIR}/bin/Release/ggml-cpu*.dll\"
            \"${CMAKE_BINARY_DIR}/bin/Release/ggml-blas*.dll\"
            \"${CMAKE_BINARY_DIR}/bin/ggml-cpu*.dll\"
            \"${CMAKE_BINARY_DIR}/bin/ggml-blas*.dll\"
        )
        foreach(_f \${_cpu_backends})
            file(INSTALL \${_f} DESTINATION \"\${CMAKE_INSTALL_PREFIX}/${_base_dest}\")
        endforeach()
    " COMPONENT llama-server)
endif()
