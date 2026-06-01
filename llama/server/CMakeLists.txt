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
if(APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path")
    set(CMAKE_BUILD_RPATH "@loader_path")
elseif(NOT WIN32)
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
    set(CMAKE_BUILD_RPATH "$ORIGIN")
endif()

if(GGML_VULKAN AND DEFINED ENV{VULKAN_SDK})
    # LunarG SDK installs SPIRV-Headers package files under SDK-specific
    # directories that CMake does not always find through the default prefix
    # search. Keep this narrow to the SDK layout so future llama.cpp bumps can
    # remove it if upstream's Vulkan dependency discovery becomes sufficient.
    file(TO_CMAKE_PATH "$ENV{VULKAN_SDK}" _ollama_vulkan_sdk)
    list(APPEND CMAKE_PREFIX_PATH "${_ollama_vulkan_sdk}")
    foreach(_ollama_spirv_headers_dir IN ITEMS
            "${_ollama_vulkan_sdk}/Lib/cmake/SPIRV-Headers"
            "${_ollama_vulkan_sdk}/lib/cmake/SPIRV-Headers"
            "${_ollama_vulkan_sdk}/share/cmake/SPIRV-Headers"
            "${_ollama_vulkan_sdk}/Lib/cmake"
            "${_ollama_vulkan_sdk}/lib/cmake"
            "${_ollama_vulkan_sdk}/share/cmake")
        if(EXISTS "${_ollama_spirv_headers_dir}/SPIRV-HeadersConfig.cmake")
            set(SPIRV-Headers_DIR "${_ollama_spirv_headers_dir}" CACHE PATH "SPIRV-Headers package directory")
            break()
        endif()
    endforeach()
endif()

if(WIN32 AND MINGW)
    # cpp-httplib uses Windows 10 APIs such as CreateFile2. Some MinGW
    # toolchains default to an older _WIN32_WINNT unless we set it explicitly.
    add_compile_definitions(_WIN32_WINNT=0x0A00 WINVER=0x0A00)
endif()

function(ollama_set_cache_default name type value doc)
    if(NOT DEFINED ${name} OR "${${name}}" STREQUAL "")
        set(${name} "${value}" CACHE ${type} "${doc}" FORCE)
    endif()
endfunction()

function(ollama_append_cache_flags name)
    set(_flags "${${name}}")
    foreach(_flag IN LISTS ARGN)
        string(FIND " ${_flags} " " ${_flag} " _found)
        if(_found EQUAL -1)
            string(APPEND _flags " ${_flag}")
        endif()
    endforeach()
    string(STRIP "${_flags}" _flags)
    set(${name} "${_flags}" CACHE STRING "" FORCE)
endfunction()

if(GGML_CUDA)
    if(OLLAMA_RUNNER_DIR STREQUAL "cuda_v12")
        ollama_set_cache_default(CMAKE_CUDA_ARCHITECTURES STRING
            "native"
            "CUDA architectures")
        ollama_set_cache_default(CMAKE_CUDA_FLAGS STRING
            "-Wno-deprecated-gpu-targets -t 2"
            "CUDA compiler flags")
    elseif(OLLAMA_RUNNER_DIR STREQUAL "cuda_v13")
        ollama_set_cache_default(CMAKE_CUDA_ARCHITECTURES STRING
            "native" "CUDA architectures")
        ollama_set_cache_default(CMAKE_CUDA_FLAGS STRING
            "-t 4" "CUDA compiler flags")
    elseif(OLLAMA_RUNNER_DIR STREQUAL "cuda_jetpack5")
        ollama_set_cache_default(CMAKE_CUDA_ARCHITECTURES STRING
            "native" "CUDA architectures")
    elseif(OLLAMA_RUNNER_DIR STREQUAL "cuda_jetpack6")
        ollama_set_cache_default(CMAKE_CUDA_ARCHITECTURES STRING
            "native" "CUDA architectures")
    endif()
endif()

if(GGML_HIP AND OLLAMA_RUNNER_DIR MATCHES "^rocm_v")
    ollama_set_cache_default(CMAKE_HIP_FLAGS STRING
        "-parallel-jobs=4" "HIP compiler flags")
    if(WIN32)
        # Windows ROCm split-load needs peer copies disabled for correctness.
        ollama_set_cache_default(GGML_CUDA_NO_PEER_COPY BOOL ON
            "Disable direct peer device copies")
        # HIP on Windows currently emits attributes and deprecated pragma
        # warnings from ROCm headers. Keep the workaround local to Windows ROCm
        # so it can be removed when the ROCm toolchain no longer needs it.
        ollama_append_cache_flags(CMAKE_C_FLAGS
            -parallel-jobs=4 -Wno-ignored-attributes -Wno-deprecated-pragma)
        ollama_append_cache_flags(CMAKE_CXX_FLAGS
            -parallel-jobs=4 -Wno-ignored-attributes -Wno-deprecated-pragma)
    endif()
endif()

include(FetchContent)

# Read pinned llama.cpp commit from version file (shared with Dockerfile)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../../LLAMA_CPP_VERSION" LLAMA_CPP_GIT_TAG)
string(STRIP "${LLAMA_CPP_GIT_TAG}" LLAMA_CPP_GIT_TAG)

# Allow local source override via environment variable (like OLLAMA_MLX_SOURCE)
if(DEFINED ENV{OLLAMA_LLAMA_CPP_SOURCE})
    get_filename_component(_src "$ENV{OLLAMA_LLAMA_CPP_SOURCE}" ABSOLUTE BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    set(FETCHCONTENT_SOURCE_DIR_LLAMA_CPP "${_src}" CACHE PATH "" FORCE)
    message(STATUS "Using local llama.cpp source: ${_src}")
endif()

# Ollama-compat shim: overlays the fetched llama.cpp source with a small
# in-memory translation layer for existing published model blobs.
# See llama/compat/README.md for details.
#
# The patch runs for fetched sources and explicit FetchContent source
# overrides, unless the caller prepared the source already. If
# OLLAMA_LLAMA_CPP_SOURCE is set, leave the developer's tree alone so they can
# apply the patch by hand while iterating on the compat layer.
set(_ollama_compat_patch_cmd "")
include(${CMAKE_CURRENT_SOURCE_DIR}/../compat/compat.cmake)
option(OLLAMA_LLAMA_CPP_SKIP_COMPAT_PATCH
    "Skip llama.cpp compat patch application because the source is already prepared"
    OFF)
set(_ollama_link_compat_sources OFF)
if(OLLAMA_LLAMA_CPP_SKIP_COMPAT_PATCH)
    set(_ollama_link_compat_sources ON)
elseif(NOT DEFINED ENV{OLLAMA_LLAMA_CPP_SOURCE})
    set(_ollama_link_compat_sources ON)
    if(DEFINED FETCHCONTENT_SOURCE_DIR_LLAMA_CPP AND NOT FETCHCONTENT_SOURCE_DIR_LLAMA_CPP STREQUAL "")
        get_filename_component(_llama_cpp_source_override
            "${FETCHCONTENT_SOURCE_DIR_LLAMA_CPP}" ABSOLUTE)
        message(STATUS
            "Applying Ollama llama.cpp compat patch to source override: "
            "${_llama_cpp_source_override}")
        execute_process(
            COMMAND ${OLLAMA_LLAMA_CPP_COMPAT_PATCH_COMMAND}
            WORKING_DIRECTORY "${_llama_cpp_source_override}"
            RESULT_VARIABLE _ollama_compat_patch_result
        )
        if(NOT _ollama_compat_patch_result EQUAL 0)
            message(FATAL_ERROR
                "Failed to apply Ollama llama.cpp compat patch to "
                "${_llama_cpp_source_override}")
        endif()
    else()
        set(_ollama_compat_patch_cmd PATCH_COMMAND ${OLLAMA_LLAMA_CPP_COMPAT_PATCH_COMMAND})
    endif()
endif()

# Configure llama.cpp build options BEFORE FetchContent_MakeAvailable.
# When included via FetchContent, llama.cpp sets LLAMA_STANDALONE=OFF
# so all optional builds default to OFF. We explicitly enable what we need.
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_HTML OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI OFF CACHE BOOL "" FORCE)
set(LLAMA_USE_PREBUILT_UI OFF CACHE BOOL "" FORCE)
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

# Link the Ollama-compat source files into the fetched llama target.
# Kept separate from the hook patch so our .cpp/.h stay
# on-disk in llama/compat/ rather than being copied into _deps/.
if(_ollama_link_compat_sources AND DEFINED OLLAMA_LLAMA_CPP_COMPAT_DIR)
    file(GLOB _compat_sources CONFIGURE_DEPENDS
        ${OLLAMA_LLAMA_CPP_COMPAT_DIR}/*.cpp)
    foreach(_compat_target IN ITEMS llama mtmd)
        if(TARGET ${_compat_target})
            target_sources(${_compat_target} PRIVATE ${_compat_sources})
            target_include_directories(${_compat_target} PRIVATE
                ${OLLAMA_LLAMA_CPP_COMPAT_DIR}
                ${llama_cpp_SOURCE_DIR}/src)
        endif()
    endforeach()
    if(TARGET mtmd)
        target_compile_definitions(mtmd PRIVATE OLLAMA_COMPAT_MTMD_BUILD)
    endif()
endif()

# Find GPU toolkits for runtime dependency bundling.
# The llama.cpp build finds these internally, but we need the
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

set(OLLAMA_LIB_DIR "lib/ollama" CACHE STRING "Install destination for Ollama runtime payloads")
set(_base_dest "${OLLAMA_LIB_DIR}")

function(ollama_append_newest_windows_runtime_family output)
    set(_runtime_dirs)
    foreach(_pattern IN LISTS ARGN)
        file(GLOB _dirs LIST_DIRECTORIES true "${_pattern}")
        list(APPEND _runtime_dirs ${_dirs})
    endforeach()

    if(_runtime_dirs)
        list(REMOVE_DUPLICATES _runtime_dirs)
        # Use one redist version per runtime family so older DLLs do not
        # overwrite newer DLLs with the same names.
        list(SORT _runtime_dirs COMPARE NATURAL ORDER DESCENDING)
        list(GET _runtime_dirs 0 _newest_dir)
        cmake_path(GET _newest_dir PARENT_PATH _newest_parent)

        set(_selected_runtime_dirs)
        foreach(_runtime_dir IN LISTS _runtime_dirs)
            cmake_path(GET _runtime_dir PARENT_PATH _runtime_parent)
            if(_runtime_parent STREQUAL _newest_parent)
                list(APPEND _selected_runtime_dirs "${_runtime_dir}")
            endif()
        endforeach()

        set(${output} ${${output}} ${_selected_runtime_dirs} PARENT_SCOPE)
    endif()
endfunction()

function(ollama_append_mingw_runtime_dlls output)
    cmake_path(GET CMAKE_CXX_COMPILER PARENT_PATH _compiler_bin_dir)
    set(_runtime_dll_dir "${_compiler_bin_dir}")

    cmake_path(GET CMAKE_CXX_COMPILER FILENAME _compiler_name)
    set(_runtime_target_triple)
    if(_compiler_name MATCHES "^([A-Za-z0-9_]+-w64-mingw32)-")
        set(_runtime_target_triple "${CMAKE_MATCH_1}")
    elseif(DEFINED OLLAMA_WINDOWS_RUNTIME_ARCH)
        string(TOLOWER "${OLLAMA_WINDOWS_RUNTIME_ARCH}" _runtime_arch)
        if(_runtime_arch MATCHES "^(arm64|aarch64)$")
            set(_runtime_target_triple "aarch64-w64-mingw32")
        elseif(_runtime_arch MATCHES "^(x64|amd64|x86_64)$")
            set(_runtime_target_triple "x86_64-w64-mingw32")
        elseif(_runtime_arch MATCHES "^(x86|i686)$")
            set(_runtime_target_triple "i686-w64-mingw32")
        endif()
    endif()

    if(_runtime_target_triple)
        cmake_path(GET _compiler_bin_dir PARENT_PATH _toolchain_root)
        set(_target_runtime_dir "${_toolchain_root}/${_runtime_target_triple}/bin")
        # llvm-mingw cross tools keep target runtime DLLs under <triple>/bin;
        # the top-level bin directory contains host-architecture DLLs.
        if(EXISTS "${_target_runtime_dir}")
            set(_runtime_dll_dir "${_target_runtime_dir}")
        elseif(CMAKE_CROSSCOMPILING)
            message(FATAL_ERROR "Could not find MinGW runtime DLL directory ${_target_runtime_dir} for ${_runtime_target_triple}")
        endif()
    endif()

    set(_runtime_dll_patterns
        "${_runtime_dll_dir}/libc++*.dll"
        "${_runtime_dll_dir}/libgcc_s_*.dll"
        "${_runtime_dll_dir}/libstdc++-*.dll"
        "${_runtime_dll_dir}/libunwind*.dll"
        "${_runtime_dll_dir}/libwinpthread-*.dll")
    if(GGML_OPENMP)
        list(APPEND _runtime_dll_patterns
            "${_runtime_dll_dir}/libgomp-[0-9]*.dll"
            "${_runtime_dll_dir}/libomp*.dll")
    endif()

    set(_runtime_dlls)
    foreach(_dll_pattern IN LISTS _runtime_dll_patterns)
        file(GLOB _matched_runtime_dlls
            LIST_DIRECTORIES false
            "${_dll_pattern}")
        list(APPEND _runtime_dlls ${_matched_runtime_dlls})
    endforeach()

    if(_runtime_dlls)
        set(${output} ${${output}} ${_runtime_dlls} PARENT_SCOPE)
    endif()
endfunction()

function(ollama_install_windows_runtime_dlls dest)
    if(NOT WIN32 OR NOT GGML_BACKEND_DL)
        return()
    endif()

    set(_ollama_windows_runtime_arch "x64")
    if(DEFINED OLLAMA_WINDOWS_RUNTIME_ARCH)
        set(_ollama_windows_runtime_arch "${OLLAMA_WINDOWS_RUNTIME_ARCH}")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|arm64|aarch64)$")
        set(_ollama_windows_runtime_arch "arm64")
    endif()

    set(_crt_dir_patterns)
    set(_openmp_dir_patterns)

    if(DEFINED ENV{VCToolsRedistDir})
        file(TO_CMAKE_PATH "$ENV{VCToolsRedistDir}" _vc_tools_redist_dir)
        list(APPEND _crt_dir_patterns
            "${_vc_tools_redist_dir}/${_ollama_windows_runtime_arch}/Microsoft.VC*.CRT")
        list(APPEND _openmp_dir_patterns
            "${_vc_tools_redist_dir}/${_ollama_windows_runtime_arch}/Microsoft.VC*.OpenMP*")
    endif()

    if(CMAKE_GENERATOR_INSTANCE)
        list(APPEND _crt_dir_patterns
            "${CMAKE_GENERATOR_INSTANCE}/VC/Redist/MSVC/*/${_ollama_windows_runtime_arch}/Microsoft.VC*.CRT")
        list(APPEND _openmp_dir_patterns
            "${CMAKE_GENERATOR_INSTANCE}/VC/Redist/MSVC/*/${_ollama_windows_runtime_arch}/Microsoft.VC*.OpenMP*")
    endif()

    if(DEFINED ENV{VSINSTALLDIR})
        file(TO_CMAKE_PATH "$ENV{VSINSTALLDIR}" _vs_install_dir)
        list(APPEND _crt_dir_patterns
            "${_vs_install_dir}/VC/Redist/MSVC/*/${_ollama_windows_runtime_arch}/Microsoft.VC*.CRT")
        list(APPEND _openmp_dir_patterns
            "${_vs_install_dir}/VC/Redist/MSVC/*/${_ollama_windows_runtime_arch}/Microsoft.VC*.OpenMP*")
    endif()

    if(MSVC)
        cmake_path(GET CMAKE_CXX_COMPILER PARENT_PATH _msvc_bin_dir)
        # Walk up from the compiler bin dir to find the VC redist directory
        # e.g. .../MSVC/14.44.35207/bin/Hostx64/x64 -> .../MSVC/14.44.35207
        cmake_path(GET _msvc_bin_dir PARENT_PATH _tmp)
        cmake_path(GET _tmp PARENT_PATH _tmp)
        cmake_path(GET _tmp PARENT_PATH _msvc_ver_dir)
        # The redist version may differ from the toolset version.
        list(APPEND _crt_dir_patterns
            "${_msvc_ver_dir}/../../../Redist/MSVC/*/${_ollama_windows_runtime_arch}/Microsoft.VC*.CRT")
        list(APPEND _openmp_dir_patterns
            "${_msvc_ver_dir}/../../../Redist/MSVC/*/${_ollama_windows_runtime_arch}/Microsoft.VC*.OpenMP*")
    endif()

    set(_runtime_dll_dirs)
    ollama_append_newest_windows_runtime_family(_runtime_dll_dirs ${_crt_dir_patterns})
    if(GGML_OPENMP)
        ollama_append_newest_windows_runtime_family(_runtime_dll_dirs ${_openmp_dir_patterns})
    endif()
    if(_runtime_dll_dirs)
        list(REMOVE_DUPLICATES _runtime_dll_dirs)
    endif()

    set(_runtime_dlls)
    foreach(_dir IN LISTS _runtime_dll_dirs)
        if(EXISTS "${_dir}")
            file(GLOB _dlls "${_dir}/*.dll")
            list(APPEND _runtime_dlls ${_dlls})
        endif()
    endforeach()
    if(MINGW)
        ollama_append_mingw_runtime_dlls(_runtime_dlls)
    endif()

    if(_runtime_dlls)
        list(REMOVE_DUPLICATES _runtime_dlls)
        install(FILES ${_runtime_dlls}
            DESTINATION "${dest}"
            COMPONENT llama-server)
    else()
        message(WARNING "Could not find Windows runtime DLLs to bundle for ${dest}")
    endif()
endfunction()

if(OLLAMA_RUNNER_DIR)
    # GPU backend build: install the GPU backend .so/.dll module.
    # install(CODE) runs at install time so CMAKE_INSTALL_CONFIG_NAME is
    # available for multi-config generators. llama.cpp backend module target
    # names vary by platform and configuration, so glob the final module
    # filenames instead of depending on a single upstream target name.
    install(CODE "
        set(_backend_dirs \"${CMAKE_BINARY_DIR}/bin\")
        if(DEFINED CMAKE_INSTALL_CONFIG_NAME AND NOT CMAKE_INSTALL_CONFIG_NAME STREQUAL \"\")
            list(PREPEND _backend_dirs \"${CMAKE_BINARY_DIR}/bin/\${CMAKE_INSTALL_CONFIG_NAME}\")
        endif()
        foreach(_dir IN LISTS _backend_dirs)
            file(GLOB _dir_gpu_backends
                LIST_DIRECTORIES false
                \"\${_dir}/libggml-${OLLAMA_GPU_BACKEND}*\"
                \"\${_dir}/ggml-${OLLAMA_GPU_BACKEND}*.dll\"
            )
            list(APPEND _gpu_backends \${_dir_gpu_backends})
        endforeach()
        foreach(_f \${_gpu_backends})
            file(INSTALL \${_f} DESTINATION \"\${CMAKE_INSTALL_PREFIX}/${_base_dest}/${OLLAMA_RUNNER_DIR}\")
        endforeach()
    " COMPONENT llama-server)

    ollama_install_windows_runtime_dlls("${_base_dest}/${OLLAMA_RUNNER_DIR}")

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
            if(WIN32)
                # HIP delay-loads COMGR on Windows, so CMake's runtime
                # dependency scan can miss the versioned companion DLL.
                set(_rocm_companion_dlls)
                foreach(_hip_dir IN ITEMS ${HIP_BIN_INSTALL_DIR} ${HIP_LIB_INSTALL_DIR})
                    if(EXISTS "${_hip_dir}")
                        file(GLOB _hip_comgr_dlls
                            LIST_DIRECTORIES false
                            "${_hip_dir}/amd_comgr*.dll")
                        list(APPEND _rocm_companion_dlls ${_hip_comgr_dlls})
                    endif()
                endforeach()
                if(_rocm_companion_dlls)
                    list(REMOVE_DUPLICATES _rocm_companion_dlls)
                    install(FILES ${_rocm_companion_dlls}
                        DESTINATION "${_base_dest}/${OLLAMA_RUNNER_DIR}"
                        COMPONENT llama-server)
                else()
                    message(FATAL_ERROR "Could not find ROCm COMGR runtime DLLs to bundle")
                endif()
            endif()
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
                RUNTIME DESTINATION "${_base_dest}/${OLLAMA_RUNNER_DIR}" COMPONENT llama-server
                LIBRARY DESTINATION "${_base_dest}/${OLLAMA_RUNNER_DIR}" COMPONENT llama-server
            )
            if(WIN32)
                # Bundle the system Vulkan loader, not an arbitrary SDK copy.
                # The loader is supplied by the installed Vulkan runtime or
                # GPU driver; using that copy avoids PATH-dependent SDK drift.
                if(DEFINED ENV{SystemRoot})
                    file(TO_CMAKE_PATH "$ENV{SystemRoot}" _ollama_windows_root)
                else()
                    set(_ollama_windows_root "C:/Windows")
                endif()
                find_file(_vulkan_loader_dll
                    NAMES vulkan-1.dll
                    HINTS "${_ollama_windows_root}/System32"
                    NO_DEFAULT_PATH)
                if(NOT _vulkan_loader_dll)
                    message(FATAL_ERROR "Could not find vulkan-1.dll in the Windows Vulkan runtime. Install the Vulkan runtime from the GPU driver or VULKAN_SDK/Helpers/VulkanRT.exe.")
                endif()
                install(FILES "${_vulkan_loader_dll}"
                    DESTINATION "${_base_dest}/${OLLAMA_RUNNER_DIR}"
                    COMPONENT llama-server)
            endif()
        endif()
    endif()
else()
    # CPU/base build — install llama-server, llama-quantize + all shared libs + CPU backend modules
    # RUNTIME covers executables and Windows DLLs; LIBRARY covers .so on Linux
    install(TARGETS llama-server llama-quantize
        RUNTIME DESTINATION ${_base_dest} COMPONENT llama-server OPTIONAL)
    set(_llama_server_base_libs ggml-base ggml llama mtmd)
    if(TARGET llama-common)
        list(APPEND _llama_server_base_libs llama-common)
    endif()
    if(BUILD_SHARED_LIBS)
        foreach(_impl_lib IN ITEMS llama-server-impl llama-quantize-impl)
            if(TARGET ${_impl_lib})
                list(APPEND _llama_server_base_libs ${_impl_lib})
            endif()
        endforeach()
    endif()

    install(TARGETS ${_llama_server_base_libs}
        RUNTIME DESTINATION ${_base_dest} COMPONENT llama-server OPTIONAL
        LIBRARY DESTINATION ${_base_dest} COMPONENT llama-server OPTIONAL)

    # Bundle Windows CRT DLLs alongside the executables so zip installs
    # do not depend on host-global redistributables.
    ollama_install_windows_runtime_dlls("${_base_dest}")

    # CPU backend modules (multiple variants from GGML_CPU_ALL_VARIANTS).
    # install(CODE) runs at install time so CMAKE_INSTALL_CONFIG_NAME is
    # available for multi-config generators. llama.cpp creates variant module
    # filenames from detected CPU features, so glob the final install payload
    # rather than hard-coding an upstream target list.
    install(CODE "
        set(_backend_dirs \"${CMAKE_BINARY_DIR}/bin\")
        if(DEFINED CMAKE_INSTALL_CONFIG_NAME AND NOT CMAKE_INSTALL_CONFIG_NAME STREQUAL \"\")
            list(PREPEND _backend_dirs \"${CMAKE_BINARY_DIR}/bin/\${CMAKE_INSTALL_CONFIG_NAME}\")
        endif()
        foreach(_dir IN LISTS _backend_dirs)
            file(GLOB _dir_cpu_backends
                LIST_DIRECTORIES false
                \"\${_dir}/libggml-cpu*\"
                \"\${_dir}/libggml-blas*\"
                \"\${_dir}/ggml-cpu*.dll\"
                \"\${_dir}/ggml-blas*.dll\"
            )
            list(APPEND _cpu_backends \${_dir_cpu_backends})
        endforeach()
        foreach(_f \${_cpu_backends})
            file(INSTALL \${_f} DESTINATION \"\${CMAKE_INSTALL_PREFIX}/${_base_dest}\")
        endforeach()
    " COMPONENT llama-server)
endif()
