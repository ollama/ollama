# Idempotent patch applier used by compat.cmake.
#
# Invocation (from a CMake PATCH_COMMAND):
#   cmake -DPATCH_DIR=<dir of *.patch> -P apply-patch.cmake
#
# Every *.patch under PATCH_DIR is applied in the current working directory
# (which ExternalProject / FetchContent sets to the fetched source's
# SOURCE_DIR). A patch already applied — detected via `git apply --reverse
# --check` — is skipped. This makes re-configuring and re-building safe.

if(NOT DEFINED PATCH_DIR)
    message(FATAL_ERROR "apply-patch.cmake: PATCH_DIR not set")
endif()

find_package(Git QUIET REQUIRED)

get_filename_component(_patch_workdir "." ABSOLUTE)
get_filename_component(_git_ceiling "${_patch_workdir}" DIRECTORY)
set(_git_apply_env GIT_CEILING_DIRECTORIES=${_git_ceiling})

file(GLOB_RECURSE _patches "${PATCH_DIR}/*.patch")
list(SORT _patches)
foreach(PATCH_FILE IN LISTS _patches)
    # If the patch can be REVERSED cleanly, it's already applied. Skip.
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env ${_git_apply_env}
            ${GIT_EXECUTABLE} apply --reverse --check "${PATCH_FILE}"
        RESULT_VARIABLE _reverse_check
        OUTPUT_QUIET ERROR_QUIET
    )
    if(_reverse_check EQUAL 0)
        message(STATUS "llama/compat: patch already applied, skipping")
        continue()
    endif()

    # Otherwise, apply forward.
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env ${_git_apply_env}
            ${GIT_EXECUTABLE} apply --whitespace=nowarn "${PATCH_FILE}"
        RESULT_VARIABLE _apply_result
    )
    if(NOT _apply_result EQUAL 0)
        message(FATAL_ERROR
            "llama/compat: failed to apply ${PATCH_FILE}\n"
            "This usually means the pinned llama.cpp source has changed. "
            "Regenerate the patch against the pinned LLAMA_CPP_VERSION and retry.")
    endif()

    message(STATUS "llama/compat: applied patch")
endforeach()
