# Idempotent patch applier used by compat.cmake and the new-architecture
# framework in llama/models/.
#
# Invocation (from a CMake PATCH_COMMAND):
#   cmake -DPATCH_FILE=<abs path> [-DPATCH_DIR=<dir of *.patch>] -P apply-patch.cmake
#
# Applies PATCH_FILE first (if set), then every <PATCH_DIR>/*.patch in sorted
# order (if set). At least one of PATCH_FILE / PATCH_DIR must be provided.
#
# Patches are applied in the current working directory, which FetchContent /
# ExternalProject sets to the fetched source's SOURCE_DIR. A patch that can be
# REVERSED cleanly is treated as already applied and skipped, so re-configuring
# or rebuilding is safe.

if(NOT DEFINED PATCH_FILE AND NOT DEFINED PATCH_DIR)
    message(FATAL_ERROR "apply-patch.cmake: set PATCH_FILE and/or PATCH_DIR")
endif()

find_package(Git QUIET REQUIRED)

get_filename_component(_patch_workdir "." ABSOLUTE)
get_filename_component(_git_ceiling "${_patch_workdir}" DIRECTORY)
set(_git_apply_env GIT_CEILING_DIRECTORIES=${_git_ceiling})

function(_ollama_apply_patch patch_file)
    if(NOT EXISTS "${patch_file}")
        message(FATAL_ERROR "apply-patch.cmake: patch does not exist: ${patch_file}")
    endif()

    # If the patch can be REVERSED cleanly, it's already applied. Skip.
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env ${_git_apply_env}
            ${GIT_EXECUTABLE} apply --reverse --check "${patch_file}"
        RESULT_VARIABLE _reverse_check
        OUTPUT_QUIET ERROR_QUIET
    )
    if(_reverse_check EQUAL 0)
        message(STATUS "llama patch: already applied, skipping ${patch_file}")
        return()
    endif()

    # Otherwise, apply forward.
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env ${_git_apply_env}
            ${GIT_EXECUTABLE} apply --whitespace=nowarn "${patch_file}"
        RESULT_VARIABLE _apply_result
    )
    if(NOT _apply_result EQUAL 0)
        message(FATAL_ERROR
            "llama patch: failed to apply ${patch_file}\n"
            "This usually means the pinned llama.cpp source has changed. "
            "Regenerate the patch (see the matching README) against the "
            "pinned LLAMA_CPP_VERSION and retry.")
    endif()

    message(STATUS "llama patch: applied ${patch_file}")
endfunction()

if(DEFINED PATCH_FILE AND NOT PATCH_FILE STREQUAL "")
    _ollama_apply_patch("${PATCH_FILE}")
endif()

if(DEFINED PATCH_DIR AND NOT PATCH_DIR STREQUAL "")
    file(GLOB _ollama_patches "${PATCH_DIR}/*.patch")
    list(SORT _ollama_patches)
    foreach(_ollama_patch IN LISTS _ollama_patches)
        _ollama_apply_patch("${_ollama_patch}")
    endforeach()
endif()
