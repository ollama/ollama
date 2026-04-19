# llama.cpp compatibility shim — CMake integration
#
# Include this file BEFORE calling FetchContent_Declare(llama_cpp ...) to
# patch the fetched upstream llama.cpp with Ollama's in-process compat
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
#   1. Two new source files dropped into the fetched tree's src/
#      (llama-ollama-compat.{h,cpp}) — Ollama-owned.
#   2. A small patch (upstream-edits.patch) that wires the new files into
#      the build and adds call-sites in upstream loaders.

set(_compat_dir ${CMAKE_CURRENT_LIST_DIR})

# Expose a single variable the main CMakeLists passes into FetchContent's
# PATCH_COMMAND. The patch is applied via a small CMake script so the step
# is idempotent — re-configuring or rebuilding won't fail with "already
# applied".
#
# The compat source files (.h, .cpp) are NOT copied into the fetched tree.
# Instead, llama/server/CMakeLists.txt does target_sources() on the llama
# target after FetchContent_MakeAvailable. That keeps Ollama's code in
# Ollama's tree and makes the patch pure call-site insertions.
set(OLLAMA_LLAMA_CPP_COMPAT_PATCH_COMMAND
    ${CMAKE_COMMAND}
        -DPATCH_FILE=${_compat_dir}/upstream-edits.patch
        -P ${_compat_dir}/apply-patch.cmake
    CACHE INTERNAL "llama.cpp compat patch command for FetchContent")

# Where the compat source files live, so the main CMakeLists can wire them
# into the llama target.
set(OLLAMA_LLAMA_CPP_COMPAT_DIR
    "${_compat_dir}"
    CACHE INTERNAL "Directory holding llama-ollama-compat.{h,cpp}")

# Also export the individual paths in case callers want to do something
# custom (e.g. emit a dependency on the patch so reconfigures re-apply).
set(OLLAMA_LLAMA_CPP_COMPAT_PATCH_FILE
    "${_compat_dir}/upstream-edits.patch"
    CACHE INTERNAL "Path to the llama.cpp compat patch")

set(OLLAMA_LLAMA_CPP_COMPAT_SOURCES
    "${_compat_dir}/llama-ollama-compat.h"
    "${_compat_dir}/llama-ollama-compat.cpp"
    CACHE INTERNAL "Source files copied into llama.cpp's src/ dir")
