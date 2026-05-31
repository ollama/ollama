# Idempotent patch applier used by compat.cmake.
#
# Invocation (from a CMake PATCH_COMMAND):
#   cmake -DPATCH_FILE=<abs path> -P apply-patch.cmake
#
# The patch is applied in the current working directory (which ExternalProject
# / FetchContent sets to the fetched source's SOURCE_DIR). If the patch is
# already applied — detected via `git apply --reverse --check` — this script
# is a no-op. This makes re-configuring and re-building the project safe.

if(NOT DEFINED PATCH_FILE)
    message(FATAL_ERROR "apply-patch.cmake: PATCH_FILE not set")
endif()
if(NOT EXISTS "${PATCH_FILE}")
    message(FATAL_ERROR "apply-patch.cmake: PATCH_FILE does not exist: ${PATCH_FILE}")
endif()

find_package(Git QUIET REQUIRED)

get_filename_component(_patch_workdir "." ABSOLUTE)
get_filename_component(_git_ceiling "${_patch_workdir}" DIRECTORY)
set(_git_apply_env GIT_CEILING_DIRECTORIES=${_git_ceiling})

# If the patch can be REVERSED cleanly, it's already applied. Skip.
execute_process(
    COMMAND ${CMAKE_COMMAND} -E env ${_git_apply_env}
        ${GIT_EXECUTABLE} apply --reverse --check "${PATCH_FILE}"
    RESULT_VARIABLE _reverse_check
    OUTPUT_QUIET ERROR_QUIET
)
if(_reverse_check EQUAL 0)
    message(STATUS "llama/compat: patch already applied, skipping")
    return()
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
        "Regenerate the patch (see llama/compat/README.md) against the "
        "pinned LLAMA_CPP_VERSION and retry.")
endif()

message(STATUS "llama/compat: applied patch")
