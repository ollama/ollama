# llama.cpp compatibility shim — CMake integration
#
# Include this file BEFORE calling FetchContent_Declare(llama_cpp ...) to
# patch the fetched llama.cpp with Ollama's in-process compatibility
# layer. Example usage:
#
#     include(${CMAKE_CURRENT_SOURCE_DIR}/../compat/compat.cmake)
#
#     FetchContent_Declare(
#         llama_cpp
#         GIT_REPOSITORY ...
#         GIT_TAG        ${LLAMA_CPP_GIT_TAG}
#         GIT_SHALLOW    TRUE
#         PATCH_COMMAND  ${OLLAMA_LLAMA_CPP_COMPAT_PATCH_COMMAND}
#         UPDATE_DISCONNECTED TRUE
#     )
#
# The compat layer consists of:
#   1. Ollama-owned compat source files linked into the fetched llama.cpp
#      targets from this directory.
#   2. A small patch file that adds call-sites in llama.cpp loaders.

set(_compat_dir ${CMAKE_CURRENT_LIST_DIR})

# Expose a single variable the main CMakeLists passes into FetchContent's
# PATCH_COMMAND. The patch is applied via a small CMake script so the step
# is idempotent — re-configuring or rebuilding won't fail with "already
# applied".
#
# The compat source files are NOT copied into the fetched tree.
# Instead, llama/server/CMakeLists.txt does target_sources() on the llama
# target after FetchContent_MakeAvailable. That keeps Ollama's code in
# Ollama's tree and makes the patch pure call-site insertions.
set(OLLAMA_LLAMA_CPP_COMPAT_PATCH_COMMAND
    ${CMAKE_COMMAND}
        -DPATCH_FILE=${_compat_dir}/llama-cpp-hooks.patch
        -P ${_compat_dir}/apply-patch.cmake
    CACHE INTERNAL "llama.cpp compat patch command for FetchContent")

# Where the compat source files live, so the main CMakeLists can wire them
# into the llama.cpp targets that need the hooks.
set(OLLAMA_LLAMA_CPP_COMPAT_DIR
    "${_compat_dir}"
    CACHE INTERNAL "Directory holding llama.cpp compat sources")

# Also export the individual paths in case callers want to do something
# custom (e.g. emit a dependency on the patch so reconfigures re-apply).
set(OLLAMA_LLAMA_CPP_COMPAT_PATCH_FILE
    "${_compat_dir}/llama-cpp-hooks.patch"
    CACHE INTERNAL "Path to the llama.cpp compat patch")

set(OLLAMA_LLAMA_CPP_COMPAT_SOURCES
    "${_compat_dir}/llama-ollama-compat.h"
    "${_compat_dir}/llama-ollama-compat.cpp"
    "${_compat_dir}/llama-ollama-compat-util.h"
    "${_compat_dir}/llama-ollama-compat-util.cpp"
    CACHE INTERNAL "Source files linked into llama.cpp targets")
