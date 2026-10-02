# Run with cmake -DTEST_BINARY_DIR=<scratch dir> -P cmake/tests/apply-git-patches.cmake
cmake_minimum_required(VERSION 3.22)

if(NOT DEFINED TEST_BINARY_DIR)
    message(FATAL_ERROR "TEST_BINARY_DIR must name a scratch directory")
endif()
get_filename_component(_applier "${CMAKE_CURRENT_LIST_DIR}/../apply-git-patches.cmake" ABSOLUTE)
file(MAKE_DIRECTORY "${TEST_BINARY_DIR}/patches")
file(WRITE "${TEST_BINARY_DIR}/patches/001-feature.patch" [=[--- a/feature.txt
+++ b/feature.txt
@@ -1,3 +1,3 @@
-base
+current
 anchor
 unchanged
]=])
file(WRITE "${TEST_BINARY_DIR}/patches/001-feature.patch.upgrade" [=[--- a/feature.txt
+++ b/feature.txt
@@ -1,2 +1,2 @@
-previous
+current
 anchor
]=])

foreach(_state IN ITEMS base previous current incompatible incomplete)
    set(_source "${TEST_BINARY_DIR}/${_state}")
    file(MAKE_DIRECTORY "${_source}")
    set(_contents "${_state}\nanchor\nunchanged\n")
    if(_state STREQUAL "incomplete")
        # The upgrade can apply, but the full patch's shared context is absent.
        set(_contents "previous\nanchor\nmodified\n")
    endif()
    file(WRITE "${_source}/feature.txt" "${_contents}")
    foreach(_run RANGE 1 2)
        execute_process(
            COMMAND "${CMAKE_COMMAND}" "-DPATCH_DIR=${TEST_BINARY_DIR}/patches" -P "${_applier}"
            WORKING_DIRECTORY "${_source}"
            RESULT_VARIABLE _result OUTPUT_VARIABLE _out ERROR_VARIABLE _err
        )
        if(_state STREQUAL "incompatible" OR _state STREQUAL "incomplete")
            if(_result EQUAL 0)
                message(FATAL_ERROR "Accepted ${_state} source")
            endif()
            set(_expected "${_contents}")
        else()
            if(NOT _result EQUAL 0)
                message(FATAL_ERROR "${_state}, run ${_run}: ${_out}${_err}")
            endif()
            set(_expected "current\nanchor\nunchanged\n")
        endif()
        file(READ "${_source}/feature.txt" _actual)
        if(NOT _actual STREQUAL _expected)
            message(FATAL_ERROR "Unexpected contents for ${_state}, run ${_run}: ${_actual}")
        endif()
    endforeach()
endforeach()
message(STATUS "Patch application, migration, repeat application, and rollback tests passed")
