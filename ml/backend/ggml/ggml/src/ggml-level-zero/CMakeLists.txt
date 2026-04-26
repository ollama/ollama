# SPDX-License-Identifier: MIT
# CMakeLists.txt — Intel Level Zero GGML backend
#
# This file is only processed when the parent CMakeLists.txt passes through
# GGML_LEVEL_ZERO=ON via add_subdirectory().  All targets defined here are
# conditional on LevelZero headers being present.
#
# Constraints enforced:
#   - NEVER pass LevelZero_LIBRARIES to target_link_libraries (INFO-1 from
#     consensus-agent). ze_loader is loaded at runtime via dlopen/LoadLibrary.
#   - Only LevelZero_INCLUDE_DIRS is used for compile-time header access.
#   - GGML_BACKEND_DL=ON produces a shared library loaded by the GGML dynamic
#     backend loader (libggml-level-zero.so/.dll).
#   - AOT SPIR-V compilation via clang -target spir64 with JIT fallback in the
#     C++ source (see ADR-L0-003).
#
cmake_minimum_required(VERSION 3.21)

# ---------------------------------------------------------------------------
# 1. Find Level Zero headers (compile-time only — do NOT link ze_loader).
# ---------------------------------------------------------------------------
find_package(LevelZero QUIET)
if (NOT LevelZero_FOUND)
    # Also try the upstream cmake module name used by Intel oneAPI SDK.
    find_package(ze_loader QUIET)
    if (ze_loader_FOUND)
        set(LevelZero_FOUND TRUE)
        set(LevelZero_INCLUDE_DIRS ${ze_loader_INCLUDE_DIRS})
    endif()
endif()

if (NOT LevelZero_FOUND)
    message(FATAL_ERROR
        "Intel Level Zero development headers not found.\n"
        "On Debian/Ubuntu: apt-get install level-zero-dev\n"
        "On RHEL/Fedora:   dnf install intel-oneapi-level-zero-devel\n"
        "On Windows:       set ONEAPI_ROOT and re-run cmake.\n"
        "Or set: -DLevelZero_INCLUDE_DIR=<path/to/include>\n"
        "Build with -DGGML_LEVEL_ZERO=OFF to suppress this error."
    )
endif()

message(STATUS "Found Level Zero headers: ${LevelZero_INCLUDE_DIRS}")

# ---------------------------------------------------------------------------
# 2. Locate clang with SPIR-V target for AOT kernel compilation (ADR-L0-003).
#    JIT fallback strings are embedded in ggml-level-zero.cpp so the build
#    still works if clang-spirv is absent — a WARNING is issued.
# ---------------------------------------------------------------------------
find_program(CLANG_SPIRV_EXE NAMES clang clang-16 clang-17 clang-18 clang-19
             DOC "Clang with SPIR-V target (Intel oneAPI SDK ships one)")

set(SPIRV_KERNELS_AVAILABLE FALSE)
if (CLANG_SPIRV_EXE)
    execute_process(
        COMMAND ${CLANG_SPIRV_EXE} -target spir64 --version
        RESULT_VARIABLE _spv_result
        OUTPUT_QUIET ERROR_QUIET
    )
    if (_spv_result EQUAL 0)
        set(SPIRV_KERNELS_AVAILABLE TRUE)
        message(STATUS "SPIR-V AOT kernel compilation enabled via ${CLANG_SPIRV_EXE}")
    endif()
endif()

if (NOT SPIRV_KERNELS_AVAILABLE)
    message(WARNING
        "clang with SPIR-V target not found — SPIR-V kernels will NOT be AOT-compiled.\n"
        "The runtime JIT fallback path in ggml-level-zero.cpp will be used instead.\n"
        "For optimal performance, install the Intel oneAPI Base Toolkit which ships\n"
        "a clang build with -target spir64 support."
    )
endif()

# Fallback: try Intel ocloc (oneAPI offline compiler) if clang-spirv not found
if(NOT SPIRV_KERNELS_AVAILABLE AND WIN32)
    find_program(OCLOC_EXE NAMES ocloc ocloc.exe
                 PATHS
                     "$ENV{ONEAPI_ROOT}/compiler/latest/bin"
                     "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/bin"
                     "C:/Program Files/Intel/oneAPI/compiler/latest/bin"
                     "$ENV{ONEAPI_ROOT}/ocloc/latest/bin"
                     "C:/Program Files (x86)/Intel/oneAPI/ocloc/2025.2/bin"
                     "C:/Program Files (x86)/Intel/oneAPI/ocloc/2025.3/bin"
                     "C:/Program Files/Intel/oneAPI/ocloc/2025.2/bin"
                     "C:/Program Files/Intel/oneAPI/ocloc/2025.3/bin"
                 DOC "Intel ocloc offline compiler for OpenCL C to SPIR-V")
    if(OCLOC_EXE)
        set(SPIRV_KERNELS_AVAILABLE TRUE)
        set(USE_OCLOC TRUE)
        message(STATUS "L0 SPIR-V AOT: using ocloc at ${OCLOC_EXE}")
    endif()
endif()

# ---------------------------------------------------------------------------
# 3. Kernel source list.
# ---------------------------------------------------------------------------
set(L0_KERNEL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/rms_norm.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/rope.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/softmax.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/attention.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/kv_cache.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/gelu_silu.cl
)

# ---------------------------------------------------------------------------
# 4. AOT SPIR-V compilation rules (one .spv per .cl).
# ---------------------------------------------------------------------------
set(L0_SPIRV_OBJECTS "")
set(L0_SPIRV_BLOBS   "")

if (SPIRV_KERNELS_AVAILABLE)
    foreach(_cl_src ${L0_KERNEL_SOURCES})
        get_filename_component(_kern_name ${_cl_src} NAME_WE)
        set(_spv_out "${CMAKE_CURRENT_BINARY_DIR}/kernels/${_kern_name}.spv")

        if(USE_OCLOC)
            # ocloc -output_no_suffix suppresses device-specific suffixes but still
            # appends ".spv".  Pass the stem (no extension) so the final file is
            # <stem>.spv, matching ${_spv_out}.
            set(_spv_stem "${CMAKE_CURRENT_BINARY_DIR}/kernels/${_kern_name}")
            add_custom_command(
                OUTPUT "${_spv_out}"
                COMMAND ${CMAKE_COMMAND} -E make_directory
                        "${CMAKE_CURRENT_BINARY_DIR}/kernels"
                COMMAND ${OCLOC_EXE} compile -file ${_cl_src} -spv_only -output_no_suffix -output ${_spv_stem}
                DEPENDS "${_cl_src}"
                COMMENT "Compiling ${_cl_src} -> ${_spv_out} via ocloc"
            )
        else()
            add_custom_command(
                OUTPUT  ${_spv_out}
                COMMAND ${CMAKE_COMMAND} -E make_directory
                        "${CMAKE_CURRENT_BINARY_DIR}/kernels"
                COMMAND ${CLANG_SPIRV_EXE}
                        -target spir64
                        -x cl
                        -cl-std=CL2.0
                        -fno-rtti
                        -O3
                        -Xclang -finclude-default-header
                        -o ${_spv_out}
                        ${_cl_src}
                DEPENDS ${_cl_src}
                COMMENT "Compiling ${_kern_name}.cl -> ${_kern_name}.spv (AOT SPIR-V)"
                VERBATIM
            )
        endif()

        list(APPEND L0_SPIRV_BLOBS ${_spv_out})

        # On Linux embed via ld -r -b binary; on Windows use .rc resource.
        if (UNIX AND NOT APPLE)
            set(_obj_out "${CMAKE_CURRENT_BINARY_DIR}/kernels/${_kern_name}_spv.o")
            add_custom_command(
                OUTPUT  ${_obj_out}
                COMMAND ld -r -b binary
                        -o ${_obj_out}
                        ${_spv_out}
                DEPENDS ${_spv_out}
                COMMENT "Embedding ${_kern_name}.spv into object section"
                VERBATIM
            )
            list(APPEND L0_SPIRV_OBJECTS ${_obj_out})
        endif()
    endforeach()
endif()

# -- Windows SPIR-V embedding: generate C headers via bin2h.cmake --
# The add_custom_command OUTPUT rules are registered here; the target_* calls
# that attach these headers are deferred until after the target is created below.
if (WIN32 AND SPIRV_KERNELS_AVAILABLE)
    foreach(_spv_out IN LISTS L0_SPIRV_BLOBS)
        get_filename_component(_kern_name "${_spv_out}" NAME_WE)
        set(_header_out "${CMAKE_CURRENT_BINARY_DIR}/kernels/${_kern_name}_spv_blob.h")
        add_custom_command(
            OUTPUT "${_header_out}"
            COMMAND ${CMAKE_COMMAND}
                -D "INPUT=${_spv_out}"
                -D "OUTPUT=${_header_out}"
                -D "VARNAME=g_spirv_${_kern_name}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/bin2h.cmake"
            DEPENDS "${_spv_out}"
            COMMENT "Embedding SPIR-V blob: ${_kern_name}"
        )
        list(APPEND L0_SPIRV_HEADERS "${_header_out}")
    endforeach()
    # NOTE: target_sources / target_compile_definitions / target_include_directories
    # for L0_SPIRV_HEADERS are applied after the ggml-level-zero target is created.
endif()

# ---------------------------------------------------------------------------
# 5. Main shared library target.
# ---------------------------------------------------------------------------
set(L0_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml-level-zero.cpp
    ${L0_SPIRV_OBJECTS}
)

# Use the GGML helper macro (defined in common.cmake) to create a backend lib.
# This sets up GGML_BACKEND_DL registration and installs the target correctly.
if (COMMAND ggml_add_backend_library)
    ggml_add_backend_library(ggml-level-zero ${L0_SOURCES})
else()
    # Fallback for builds that do not have ggml_add_backend_library.
    add_library(ggml-level-zero SHARED ${L0_SOURCES})
    target_compile_definitions(ggml-level-zero PRIVATE GGML_BACKEND_DL)
endif()

# -- Attach Windows SPIR-V embedded headers now that the target exists --
# These calls were deferred from the bin2h.cmake block above to avoid
# "target not built by this project" errors when target_* runs before
# ggml_add_backend_library / add_library creates the target.
if (WIN32 AND SPIRV_KERNELS_AVAILABLE AND L0_SPIRV_HEADERS)
    target_sources(ggml-level-zero PRIVATE ${L0_SPIRV_HEADERS})
    target_compile_definitions(ggml-level-zero PRIVATE GGML_L0_SPIRV_EMBEDDED_HEADERS=1)
    target_compile_definitions(ggml-level-zero PRIVATE GGML_L0_SPIRV_AOT_AVAILABLE=1)
    # Source uses #include "kernels/<name>_spv_blob.h", so the parent of kernels/
    # must be on the include path, not kernels/ itself.
    target_include_directories(ggml-level-zero PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
endif()

# CRITICAL (INFO-1): Use ONLY target_include_directories — NEVER
# target_link_libraries with LevelZero_LIBRARIES. ze_loader is dlopen'd at
# runtime in ze_ollama_init(); static/dynamic linking would violate ADR-L0-006.
target_include_directories(ggml-level-zero PRIVATE
    ${LevelZero_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}          # ze_ollama.h, ze_*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../..    # ggml.h (relative from src/)
    ${CMAKE_CURRENT_BINARY_DIR}/kernels  # generated header stubs if any
)

# C++20 required for std::bit_ceil (ze_buffer.hpp) and std::countr_zero.
target_compile_features(ggml-level-zero PRIVATE cxx_std_20)
set_target_properties(ggml-level-zero PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)

target_compile_definitions(ggml-level-zero PRIVATE
    ZE_OLLAMA_VERSION_STR="1.0.0"
)
target_compile_definitions(ggml-level-zero PRIVATE GGML_BACKEND_SHARED GGML_BACKEND_BUILD)

if (SPIRV_KERNELS_AVAILABLE)
    target_compile_definitions(ggml-level-zero PRIVATE
        GGML_L0_SPIRV_AOT_AVAILABLE=1
    )
endif()

# ---------------------------------------------------------------------------
# Debug build option: verbose L0 backend internal logging.
# Default OFF — production binary is unaffected.
# ---------------------------------------------------------------------------
option(GGML_L0_DEBUG "Enable Level Zero backend diagnostic logging" OFF)
if (GGML_L0_DEBUG)
    target_compile_definitions(ggml-level-zero PRIVATE GGML_L0_DEBUG=1)
    message(STATUS "L0 backend: diagnostic logging ENABLED (GGML_L0_DEBUG=ON)")
endif()

# ---------------------------------------------------------------------------
# Debug build option: per-tensor I/O addressability assertions and trace logs.
# Required: enables Test #4 (pointer addressability) in the QA CI matrix.
# Default OFF — production binary is unaffected.
# Note: The CI matrix variant that sets this ON is qa-testing-agent's
# responsibility in D-phase (.github/workflows/ci-intel.yaml). This option
# only wires the compile-time flag so the CI can use it.
# ---------------------------------------------------------------------------
option(OLLAMA_L0_DEBUG_TENSOR_IO
       "Enable per-tensor I/O addressability assertions and trace logging in L0 backend"
       OFF)
if (OLLAMA_L0_DEBUG_TENSOR_IO)
    target_compile_definitions(ggml-level-zero PRIVATE OLLAMA_L0_DEBUG_TENSOR_IO=1)
    message(STATUS "L0 backend: tensor I/O debug trace ENABLED (OLLAMA_L0_DEBUG_TENSOR_IO=ON)")
endif()

# Visibility: hide all symbols except the GGML_BACKEND_DL registration entry.
if (NOT MSVC)
    target_compile_options(ggml-level-zero PRIVATE
        -fvisibility=hidden
        -fvisibility-inlines-hidden
    )
    target_link_options(ggml-level-zero PRIVATE
        -Wl,--exclude-libs,ALL
    )
endif()

if(MSVC)
    target_link_options(ggml-level-zero PRIVATE /OPT:REF /OPT:ICF)
endif()

if (WIN32)
    target_compile_definitions(ggml-level-zero PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        _CRT_SECURE_NO_WARNINGS
    )
endif()

# Link dl on Linux for dlopen used inside ze_ollama_init().
if (UNIX AND NOT APPLE)
    target_link_libraries(ggml-level-zero PRIVATE dl)
    if (SPIRV_KERNELS_AVAILABLE)
        # Link the binary-embedded SPIR-V object files.
        # target_sources cannot handle raw .o files portably so we use
        # target_link_libraries with the object paths directly.
        target_link_libraries(ggml-level-zero PRIVATE ${L0_SPIRV_OBJECTS})
    endif()
endif()

# On Linux use OpenSSL for SHA-256 (MIT-compatible); on Windows use BCrypt.
if (UNIX AND NOT APPLE)
    find_package(OpenSSL QUIET)
    if (OpenSSL_FOUND)
        target_link_libraries(ggml-level-zero PRIVATE OpenSSL::Crypto)
        target_compile_definitions(ggml-level-zero PRIVATE ZE_OLLAMA_SHA256_OPENSSL=1)
    else()
        message(WARNING "OpenSSL not found — SHA-256 kernel cache key will use a built-in fallback implementation.")
        target_compile_definitions(ggml-level-zero PRIVATE ZE_OLLAMA_SHA256_BUILTIN=1)
    endif()
elseif (WIN32)
    target_link_libraries(ggml-level-zero PRIVATE bcrypt)
    target_compile_definitions(ggml-level-zero PRIVATE ZE_OLLAMA_SHA256_BCRYPT=1)
endif()

# ---------------------------------------------------------------------------
# 6. Public GGML include header installation.
# ---------------------------------------------------------------------------
install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../include/ggml-level-zero.h
    DESTINATION include
    COMPONENT   Development
)

install(
    TARGETS ggml-level-zero
    LIBRARY DESTINATION lib/ollama
    RUNTIME DESTINATION lib/ollama
)

# ---------------------------------------------------------------------------
# 7. Register with parent CMakeLists so ggml links backend discovery symbols.
#    The actual registration macro call lives in the parent src/CMakeLists.txt
#    as: ggml_add_backend(LevelZero)   (appended after Vulkan, append-only).
# ---------------------------------------------------------------------------
message(STATUS "Intel Level Zero GGML backend configured.")
