foreach(_variable IN ITEMS GO_EXECUTABLE SOURCE_DIR BINARY_DIR OUTPUT_DIR TARGETS)
    if(NOT DEFINED ${_variable})
        message(FATAL_ERROR "${_variable} is required")
    endif()
endforeach()
if(NOT TARGETS)
    message(FATAL_ERROR "At least one GOOS/GOARCH target is required")
endif()

execute_process(
    COMMAND "${GO_EXECUTABLE}" tool dist list
    OUTPUT_VARIABLE _supported_targets
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
string(REPLACE "\r" "" _supported_targets "${_supported_targets}")
string(REPLACE "\n" ";" _supported_targets "${_supported_targets}")

set(_version v2.0.1)
set(_tool_dir "${BINARY_DIR}/go-licenses-${_version}")
set(_tool "${_tool_dir}/go-licenses")
if(CMAKE_HOST_WIN32)
    string(APPEND _tool ".exe")
endif()

if(NOT EXISTS "${_tool}")
    file(MAKE_DIRECTORY "${_tool_dir}")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
            --unset=GOOS --unset=GOARCH --unset=GOARM --unset=GOAMD64
            "GOBIN=${_tool_dir}" "GOFLAGS="
            "${GO_EXECUTABLE}" install "github.com/google/go-licenses/v2@${_version}"
        WORKING_DIRECTORY "${SOURCE_DIR}"
        COMMAND_ERROR_IS_FATAL ANY)
endif()

set(_staging_dir "${BINARY_DIR}/go-license-files")
file(REMOVE_RECURSE "${_staging_dir}")

foreach(_target IN LISTS TARGETS)
    list(FIND _supported_targets "${_target}" _target_index)
    if(_target_index EQUAL -1)
        message(FATAL_ERROR "Unsupported Go license target '${_target}'; expected a GOOS/GOARCH from 'go tool dist list'")
    endif()
    string(REPLACE "/" ";" _target_parts "${_target}")
    list(GET _target_parts 0 _goos)
    list(GET _target_parts 1 _goarch)

    set(_packages .)
    if(_goos STREQUAL "darwin" OR _goos STREQUAL "windows")
        list(APPEND _packages ./app/cmd/app)
    endif()

    set(_target_staging_dir "${BINARY_DIR}/go-license-files-${_goos}-${_goarch}")
    message(STATUS "Collecting Go licenses for ${_target}")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
            "GOOS=${_goos}" "GOARCH=${_goarch}" "CGO_ENABLED=1"
            "${_tool}" save ${_packages}
            --save_path "${_target_staging_dir}" --force
            --ignore github.com/apache/arrow/go/arrow
        WORKING_DIRECTORY "${SOURCE_DIR}"
        COMMAND_ERROR_IS_FATAL ANY)

    file(COPY "${_target_staging_dir}/" DESTINATION "${_staging_dir}")
endforeach()

# Arrow's aggregate license includes a license that go-licenses cannot classify.
execute_process(
    COMMAND "${GO_EXECUTABLE}" list -m -f "{{.Dir}}" github.com/apache/arrow/go/arrow
    WORKING_DIRECTORY "${SOURCE_DIR}"
    OUTPUT_VARIABLE _arrow_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
if(NOT EXISTS "${_arrow_dir}/LICENSE.txt")
    message(FATAL_ERROR "failed to locate the Apache Arrow license")
endif()

set(_arrow_output_dir "${_staging_dir}/github.com/apache/arrow/go/arrow")
file(MAKE_DIRECTORY "${_arrow_output_dir}")
file(COPY "${_arrow_dir}/LICENSE.txt" DESTINATION "${_arrow_output_dir}")

file(GLOB_RECURSE _license_files
    LIST_DIRECTORIES FALSE
    RELATIVE "${_staging_dir}"
    "${_staging_dir}/*")
list(SORT _license_files)

file(MAKE_DIRECTORY "${OUTPUT_DIR}")
set(_output "${OUTPUT_DIR}/GO_LICENSE")
file(WRITE "${_output}" "Go licenses for Ollama and its dependencies.\n")
foreach(_license_file IN LISTS _license_files)
    file(READ "${_staging_dir}/${_license_file}" _license_text)
    file(APPEND "${_output}"
        "\n================================================================================\n"
        "${_license_file}\n"
        "================================================================================\n"
        "${_license_text}\n")
endforeach()
