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
# PATCH_COMMAND. Uses CMake's own `-E copy` so it's cross-platform; uses
# `git apply` because the patch is in unified git-diff format (same as what
# `git diff` produces — regeneration is one command, see README).
set(OLLAMA_LLAMA_CPP_COMPAT_PATCH_COMMAND
    ${CMAKE_COMMAND} -E copy
        "${_compat_dir}/llama-ollama-compat.h"
        "src/llama-ollama-compat.h"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${_compat_dir}/llama-ollama-compat.cpp"
        "src/llama-ollama-compat.cpp"
    COMMAND git apply --whitespace=nowarn
        "${_compat_dir}/upstream-edits.patch"
    CACHE INTERNAL "llama.cpp compat patch command for FetchContent")

# Also export the individual paths in case callers want to do something
# custom (e.g. emit a dependency on the patch so reconfigures re-apply).
set(OLLAMA_LLAMA_CPP_COMPAT_PATCH_FILE
    "${_compat_dir}/upstream-edits.patch"
    CACHE INTERNAL "Path to the llama.cpp compat patch")

set(OLLAMA_LLAMA_CPP_COMPAT_SOURCES
    "${_compat_dir}/llama-ollama-compat.h"
    "${_compat_dir}/llama-ollama-compat.cpp"
    CACHE INTERNAL "Source files copied into llama.cpp's src/ dir")
