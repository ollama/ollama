# Idempotent external-source patch applier.
#
# Invocation (from a CMake PATCH_COMMAND):
#   cmake -DPATCH_DIR=<dir of *.patch> [-DPATCH_LABEL=<name>]
#     -P cmake/apply-git-patches.cmake
#
# Every *.patch under PATCH_DIR is applied in numeric filename order in the
# current working directory (which ExternalProject / FetchContent sets to the
# fetched source's SOURCE_DIR). A patch already applied — detected via
# `git apply --reverse --check` — is skipped. This makes re-configuring and
# re-building safe when the source directory outlives its CMake patch stamp.
# A sibling <patch>.upgrade may migrate a retained source tree from a previous
# version of a patch. It is only used when the full patch cannot apply.

if(NOT DEFINED PATCH_DIR)
    message(FATAL_ERROR "apply-git-patches.cmake: PATCH_DIR not set")
endif()

if(NOT DEFINED PATCH_LABEL)
    set(PATCH_LABEL "external patches")
endif()

find_package(Git QUIET REQUIRED)

get_filename_component(_patch_workdir "." ABSOLUTE)
get_filename_component(_git_ceiling "${_patch_workdir}" DIRECTORY)
set(_git_apply_env GIT_CEILING_DIRECTORIES=${_git_ceiling})

file(GLOB_RECURSE _patches "${PATCH_DIR}/*.patch")
set(_patch_entries)
foreach(PATCH_FILE IN LISTS _patches)
    get_filename_component(_patch_name "${PATCH_FILE}" NAME)
    list(APPEND _patch_entries "${_patch_name}|${PATCH_FILE}")
endforeach()

list(SORT _patch_entries)
foreach(_patch_entry IN LISTS _patch_entries)
    string(REGEX REPLACE "^[^|]*\\|" "" PATCH_FILE "${_patch_entry}")
    file(RELATIVE_PATH _patch_rel "${PATCH_DIR}" "${PATCH_FILE}")

    # If the patch can be REVERSED cleanly, it's already applied. Skip.
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env ${_git_apply_env}
            ${GIT_EXECUTABLE} apply --reverse --check "${PATCH_FILE}"
        RESULT_VARIABLE _reverse_check
        OUTPUT_QUIET ERROR_QUIET
    )
    if(_reverse_check EQUAL 0)
        message(STATUS "${PATCH_LABEL}: ${_patch_rel} already applied, skipping")
        continue()
    endif()

    # Otherwise, apply forward, or migrate a recognized older patched tree.
    set(_apply_patch "${PATCH_FILE}")
    if(EXISTS "${PATCH_FILE}.upgrade")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E env ${_git_apply_env}
                ${GIT_EXECUTABLE} apply --check "${PATCH_FILE}"
            RESULT_VARIABLE _forward_check
            OUTPUT_QUIET ERROR_QUIET
        )
        if(NOT _forward_check EQUAL 0)
            set(_apply_patch "${PATCH_FILE}.upgrade")
        endif()
    endif()
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env ${_git_apply_env}
            ${GIT_EXECUTABLE} apply --whitespace=nowarn "${_apply_patch}"
        RESULT_VARIABLE _apply_result
    )
    if(NOT _apply_result EQUAL 0)
        message(FATAL_ERROR
            "${PATCH_LABEL}: failed to apply ${_patch_rel}\n"
            "The pinned source is neither clean nor compatible with this patch. "
            "Remove the retained source directory (${_patch_workdir}), or "
            "regenerate the patch against the pinned source before retrying.")
    endif()

    if(NOT _apply_patch STREQUAL PATCH_FILE)
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E env ${_git_apply_env}
                ${GIT_EXECUTABLE} apply --reverse --check "${PATCH_FILE}"
            RESULT_VARIABLE _upgraded_check
            OUTPUT_QUIET ERROR_QUIET
        )
        if(NOT _upgraded_check EQUAL 0)
            execute_process(
                COMMAND ${CMAKE_COMMAND} -E env ${_git_apply_env}
                    ${GIT_EXECUTABLE} apply --reverse "${_apply_patch}"
                COMMAND_ERROR_IS_FATAL ANY
            )
            message(FATAL_ERROR "${PATCH_LABEL}: upgrade did not produce the expected ${_patch_rel}; rolled back")
        endif()
    endif()

    message(STATUS "${PATCH_LABEL}: applied ${_patch_rel}")
endforeach()
