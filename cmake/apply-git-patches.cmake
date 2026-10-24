# Idempotent external-source patch applier.
#
# Invocation (from a CMake PATCH_COMMAND):
#   cmake -DPATCH_DIR=<dir of *.patch> [-DPATCH_LABEL=<name>]
#     -P cmake/apply-git-patches.cmake
#
# Every *.patch under PATCH_DIR is applied in numeric filename order in the
# current working directory (which ExternalProject / FetchContent sets to the
# fetched source's SOURCE_DIR). The longest prefix of the series that is
# already applied — detected via `git apply --reverse --check` on the prefix
# as a whole — is skipped. This makes re-configuring and re-building safe when
# the source directory outlives its CMake patch stamp.

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
set(_patch_files)
foreach(_patch_entry IN LISTS _patch_entries)
    string(REGEX REPLACE "^[^|]*\\|" "" PATCH_FILE "${_patch_entry}")
    list(APPEND _patch_files "${PATCH_FILE}")
endforeach()

# A later patch may edit lines an earlier one added, so a patch alone can stop
# reversing once its successors are applied; check each prefix as a unit. git
# chains patches only within a single input, and --reverse undoes them in
# reverse order, so the prefix is piped in as one stream in forward order.
set(_applied_count 0)
set(_prefix)
foreach(PATCH_FILE IN LISTS _patch_files)
    list(APPEND _prefix "${PATCH_FILE}")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E cat ${_prefix}
        COMMAND ${CMAKE_COMMAND} -E env ${_git_apply_env}
            ${GIT_EXECUTABLE} apply --reverse --check
        RESULT_VARIABLE _reverse_check
        OUTPUT_QUIET ERROR_QUIET
    )
    if(_reverse_check EQUAL 0)
        list(LENGTH _prefix _applied_count)
    endif()
endforeach()

set(_patch_index 0)
foreach(PATCH_FILE IN LISTS _patch_files)
    math(EXPR _patch_index "${_patch_index} + 1")
    file(RELATIVE_PATH _patch_rel "${PATCH_DIR}" "${PATCH_FILE}")

    if(_patch_index LESS_EQUAL _applied_count)
        message(STATUS "${PATCH_LABEL}: ${_patch_rel} already applied, skipping")
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
            "${PATCH_LABEL}: failed to apply ${_patch_rel}\n"
            "The pinned source is neither clean nor compatible with this patch. "
            "Remove the retained source directory (${_patch_workdir}), or "
            "regenerate the patch against the pinned source before retrying.")
    endif()

    message(STATUS "${PATCH_LABEL}: applied ${_patch_rel}")
endforeach()
