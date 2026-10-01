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

# Later patches can change lines introduced by earlier patches. Check the
# complete stack in reverse order on temporary copies before testing patches
# individually, so reconfiguration does not reject an already-applied stack.
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef _patch_check_id)
set(_patch_check_parent "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles")
set(_patch_check_dir "${_patch_check_parent}/patch-check-${_patch_check_id}")
file(MAKE_DIRECTORY "${_patch_check_dir}")
foreach(_patch_entry IN LISTS _patch_entries)
    string(REGEX REPLACE "^[^|]*\\|" "" PATCH_FILE "${_patch_entry}")
    file(STRINGS "${PATCH_FILE}" _patch_paths REGEX "^(--- a/|[+][+][+] b/)")
    foreach(_patch_path IN LISTS _patch_paths)
        string(SUBSTRING "${_patch_path}" 6 -1 _patch_path)
        if(EXISTS "${_patch_workdir}/${_patch_path}")
            get_filename_component(_patch_parent "${_patch_check_dir}/${_patch_path}" DIRECTORY)
            file(MAKE_DIRECTORY "${_patch_parent}")
            file(COPY_FILE "${_patch_workdir}/${_patch_path}" "${_patch_check_dir}/${_patch_path}")
        endif()
    endforeach()
endforeach()
set(_reverse_entries ${_patch_entries})
list(REVERSE _reverse_entries)
set(_stack_applied TRUE)
foreach(_patch_entry IN LISTS _reverse_entries)
    string(REGEX REPLACE "^[^|]*\\|" "" PATCH_FILE "${_patch_entry}")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env "GIT_CEILING_DIRECTORIES=${_patch_check_parent}"
            ${GIT_EXECUTABLE} apply --reverse "${PATCH_FILE}"
        WORKING_DIRECTORY "${_patch_check_dir}"
        RESULT_VARIABLE _reverse_result
        OUTPUT_QUIET ERROR_QUIET
    )
    if(NOT _reverse_result EQUAL 0)
        set(_stack_applied FALSE)
        break()
    endif()
endforeach()
file(REMOVE_RECURSE "${_patch_check_dir}")
if(_stack_applied)
    message(STATUS "${PATCH_LABEL}: patch stack already applied, skipping")
    return()
endif()

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
