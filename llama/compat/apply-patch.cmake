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

set(_ui_archive "")
set(_llama_cpp_version_file "${PATCH_DIR}/../../LLAMA_CPP_VERSION")
if(EXISTS "${_llama_cpp_version_file}")
    file(READ "${_llama_cpp_version_file}" _llama_cpp_tag)
    string(STRIP "${_llama_cpp_tag}" _llama_cpp_tag)
    set(_ui_archive "${PATCH_DIR}/ui/llama-${_llama_cpp_tag}-ui.tar.gz")
endif()

if(_ui_archive AND EXISTS "${_ui_archive}")
    set(_ui_dist_dir "${_patch_workdir}/tools/ui/dist")

    if(EXISTS "${_ui_dist_dir}/index.html")
        message(STATUS "llama/compat: bundled UI assets already present, skipping")
    else()
        set(_ui_extract_dir "${_patch_workdir}/tools/ui/.ollama-ui-extract")
        file(REMOVE_RECURSE "${_ui_extract_dir}")
        file(MAKE_DIRECTORY "${_ui_dist_dir}" "${_ui_extract_dir}")
        file(ARCHIVE_EXTRACT INPUT "${_ui_archive}" DESTINATION "${_ui_extract_dir}")

        file(GLOB _ui_roots RELATIVE "${_ui_extract_dir}" "${_ui_extract_dir}/*")
        list(LENGTH _ui_roots _ui_root_count)
        if(NOT _ui_root_count EQUAL 1)
            message(FATAL_ERROR
                "llama/compat: expected one top-level directory in ${_ui_archive}")
        endif()

        list(GET _ui_roots 0 _ui_root)
        if(NOT EXISTS "${_ui_extract_dir}/${_ui_root}/index.html")
            message(FATAL_ERROR
                "llama/compat: ${_ui_archive} does not contain UI assets")
        endif()

        file(COPY "${_ui_extract_dir}/${_ui_root}/" DESTINATION "${_ui_dist_dir}")
        file(REMOVE_RECURSE "${_ui_extract_dir}")
        message(STATUS "llama/compat: extracted bundled UI assets from ${_ui_archive}")
    endif()
endif()
