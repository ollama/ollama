cmake_minimum_required(VERSION 3.24)

project(Ollama C CXX)

# Handle cross-compilation on macOS: when CMAKE_OSX_ARCHITECTURES is set to a
# single architecture different from the host, override CMAKE_SYSTEM_PROCESSOR
# to match. This is necessary because CMAKE_SYSTEM_PROCESSOR defaults to the
# host architecture, but downstream projects (like MLX) use it to detect the
# target architecture.
if(CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_OSX_ARCHITECTURES MATCHES ";")
    # Single architecture specified
    if(CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64" AND NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        message(STATUS "Cross-compiling for x86_64: overriding CMAKE_SYSTEM_PROCESSOR from ${CMAKE_SYSTEM_PROCESSOR} to x86_64")
        set(CMAKE_SYSTEM_PROCESSOR "x86_64")
    elseif(CMAKE_OSX_ARCHITECTURES STREQUAL "arm64" AND NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        message(STATUS "Cross-compiling for arm64: overriding CMAKE_SYSTEM_PROCESSOR from ${CMAKE_SYSTEM_PROCESSOR} to arm64")
        set(CMAKE_SYSTEM_PROCESSOR "arm64")
    endif()
endif()

include(CheckLanguage)
include(GNUInstallDirs)

find_package(Threads REQUIRED)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# These defaults can be overridden by presets (e.g., for static macOS llama-server builds)
if(NOT DEFINED BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON) # Recent versions of MLX require gnu++17 extensions to compile properly

# GGML backend for inference is provided by llama-server (built separately via
# llama/server/CMakeLists.txt using FetchContent from the pinned llama.cpp source).
# The root CMake project is the orchestration entrypoint; backend-specific
# build rules live in subprojects under cmake/.

if(APPLE)
    set(CMAKE_BUILD_RPATH "@loader_path")
    set(CMAKE_INSTALL_RPATH "@loader_path")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
endif()

set(OLLAMA_BUILD_DIR ${CMAKE_BINARY_DIR}/lib/ollama)
set(OLLAMA_LIB_DIR "lib/ollama" CACHE STRING "Install destination for Ollama runtime payloads")
set(OLLAMA_INSTALL_DIR ${OLLAMA_LIB_DIR}/${OLLAMA_RUNNER_DIR})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY         ${OLLAMA_BUILD_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${OLLAMA_BUILD_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${OLLAMA_BUILD_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY         ${OLLAMA_BUILD_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG   ${OLLAMA_BUILD_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${OLLAMA_BUILD_DIR})

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/llama/server/CMakeLists.txt")
    set(OLLAMA_HAVE_LLAMA_SERVER TRUE)
else()
    set(OLLAMA_HAVE_LLAMA_SERVER FALSE)
endif()

# ---------------------------------------------------------------------------
# s390x / IBM Z & LinuxONE feature flags
#
# These options are declared here (root project) so that:
#   -DOLLAMA_S390X_* flags on the top-level cmake invocation are recognized.
# The resolved values are forwarded into the llama/server ExternalProject
# via cmake/local.cmake.  The actual default-value logic still lives in
# llama/server/CMakeLists.txt for standalone sub-project builds.
#
# OLLAMA_S390X_BIGENDIAN  – byte-swap little-endian GGUF scalar fields
#                           (auto ON for s390x; required for all z-series builds)
#
# OLLAMA_S390X_VXE        – VX/VXE/VXE2 SIMD acceleration via GGML_VXE
#                           Available on IBM z15 / LinuxONE 3 or later.
#                           Enabled by default on s390x; set OFF only for
#                           debugging or when targeting older z14 hardware.
#
# OLLAMA_S390X_ZDNN       – IBM zDNN / zAIU co-processor acceleration via GGML_ZDNN
#                           Available on IBM z17 / LinuxONE 5 or later.
#                           Requires the IBM zDNN library installed on the build host.
#                           OFF by default even on s390x; opt-in explicitly.
# ---------------------------------------------------------------------------
if(CMAKE_SYSTEM_PROCESSOR MATCHES "s390x")
    option(OLLAMA_S390X_BIGENDIAN "Enable big-endian GGUF byte-swap (auto ON for s390x)" ON)
    option(OLLAMA_S390X_VXE
        "Enable VX/VXE/VXE2 SIMD acceleration (GGML_VXE=ON; z15+ required; auto ON for s390x)" ON)
    option(OLLAMA_S390X_ZDNN
        "Enable IBM zDNN/zAIU co-processor acceleration (GGML_ZDNN=ON; z17+ and zDNN library required)" OFF)
else()
    option(OLLAMA_S390X_BIGENDIAN "Enable big-endian GGUF byte-swap (auto ON for s390x)" OFF)
    option(OLLAMA_S390X_VXE
        "Enable VX/VXE/VXE2 SIMD acceleration (GGML_VXE=ON; z15+ required; auto ON for s390x)" OFF)
    option(OLLAMA_S390X_ZDNN
        "Enable IBM zDNN/zAIU co-processor acceleration (GGML_ZDNN=ON; z17+ and zDNN library required)" OFF)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "s390x")
    message(STATUS "s390x target detected")
    message(STATUS "  OLLAMA_S390X_BIGENDIAN = ${OLLAMA_S390X_BIGENDIAN}")
    message(STATUS "  OLLAMA_S390X_VXE       = ${OLLAMA_S390X_VXE}  (GGML_VXE; z15+ SIMD)")
    message(STATUS "  OLLAMA_S390X_ZDNN      = ${OLLAMA_S390X_ZDNN}  (GGML_ZDNN; z17+ zAIU co-processor)")
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/local.cmake)
