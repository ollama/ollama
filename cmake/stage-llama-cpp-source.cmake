# Prepare a FETCHCONTENT_SOURCE_DIR_LLAMA_CPP override for the llama-server
# builds, which are configured with OLLAMA_LLAMA_CPP_SKIP_COMPAT_PATCH=ON and
# therefore expect an already patched tree.
#
# By default the override is staged (copied) into the build directory and the
# llama/compat patches are applied to the copy, so the caller's source tree is
# never modified. The copy skips the top-level .git and build* directories.
# It is refreshed only when the override's identity changes: git HEAD plus a
# hash of `git diff HEAD` (tracked changes) when the override is a git
# checkout, and the patch file contents. Without git it is recopied on every
# configure. Re-run the CMake configure step after editing the override.
#
# Set OLLAMA_LLAMA_CPP_OVERRIDE_PATCH_IN_PLACE=ON to patch the override tree
# itself instead (idempotent: git apply --reverse --check skips patches that
# are already applied; a tree that conflicts with the patch aborts configure).

function(ollama_prepare_llama_cpp_override src out_var)
    if(OLLAMA_LLAMA_CPP_OVERRIDE_PATCH_IN_PLACE)
        message(STATUS "llama/compat: patching source override in place: ${src}")
        set(_dst "${src}")
    else()
        set(_dst "${CMAKE_BINARY_DIR}/_deps/llama_cpp-override-src")
        set(_stamp "${CMAKE_BINARY_DIR}/_deps/llama_cpp-override.stamp")

        find_package(Git QUIET)
        set(_identity "src=${src}")
        if(GIT_FOUND AND EXISTS "${src}/.git")
            execute_process(COMMAND ${GIT_EXECUTABLE} -C "${src}" rev-parse HEAD
                OUTPUT_VARIABLE _head OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE _head_rc ERROR_QUIET)
            execute_process(COMMAND ${GIT_EXECUTABLE} -C "${src}" diff HEAD --binary
                OUTPUT_VARIABLE _diff RESULT_VARIABLE _diff_rc ERROR_QUIET)
            if(_head_rc EQUAL 0 AND _diff_rc EQUAL 0)
                string(SHA256 _diff_hash "${_diff}")
                # The copy has no .git; record the commit for build info
                # (read by llama/server/CMakeLists.txt).
                string(SUBSTRING "${_head}" 0 9 _short_head)
                set(_commit_marker "${_short_head}")
                if(NOT _diff STREQUAL "")
                    string(APPEND _commit_marker "-dirty")
                endif()
                string(APPEND _identity ";head=${_head};diff=${_diff_hash}")
            else()
                string(TIMESTAMP _now "%s")
                string(APPEND _identity ";nogit=${_now}")
            endif()
        else()
            string(TIMESTAMP _now "%s")
            string(APPEND _identity ";nogit=${_now}")
        endif()
        file(GLOB _patches "${OLLAMA_LLAMA_CPP_COMPAT_DIR}/*.patch")
        list(SORT _patches)
        foreach(_patch IN LISTS _patches)
            file(SHA256 "${_patch}" _patch_hash)
            string(APPEND _identity ";patch=${_patch_hash}")
        endforeach()

        set(_old_identity "")
        if(EXISTS "${_stamp}")
            file(READ "${_stamp}" _old_identity)
        endif()
        if(EXISTS "${_dst}" AND _old_identity STREQUAL _identity)
            message(STATUS "llama/compat: staged source override is current: ${_dst}")
        else()
            message(STATUS "llama/compat: staging source override ${src} -> ${_dst}")
            file(REMOVE_RECURSE "${_dst}")
            file(MAKE_DIRECTORY "${_dst}")
            file(GLOB _entries LIST_DIRECTORIES true RELATIVE "${src}" "${src}/*" "${src}/.*")
            foreach(_entry IN LISTS _entries)
                if(_entry MATCHES "^(\\.|\\.\\.|\\.git|build[^/]*)$")
                    continue()
                endif()
                file(COPY "${src}/${_entry}" DESTINATION "${_dst}")
            endforeach()
            if(DEFINED _commit_marker)
                file(WRITE "${_dst}/.ollama-llama-cpp-commit" "${_commit_marker}\n")
            endif()
            # Written only after the patch succeeds (below) so a failed patch
            # forces a fresh copy next time.
            set(_write_stamp TRUE)
        endif()
    endif()

    execute_process(
        COMMAND ${OLLAMA_LLAMA_CPP_COMPAT_PATCH_COMMAND}
        WORKING_DIRECTORY "${_dst}"
        RESULT_VARIABLE _patch_rc)
    if(NOT _patch_rc EQUAL 0)
        if(DEFINED _stamp)
            file(REMOVE "${_stamp}")
        endif()
        message(FATAL_ERROR "llama/compat: failed to apply patches to ${_dst}")
    endif()
    if(_write_stamp)
        file(WRITE "${_stamp}" "${_identity}")
    endif()

    set(${out_var} "${_dst}" PARENT_SCOPE)
endfunction()
