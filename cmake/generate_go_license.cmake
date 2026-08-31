foreach(_variable IN ITEMS GO_EXECUTABLE SOURCE_DIR BINARY_DIR OUTPUT_DIR)
    if(NOT DEFINED ${_variable})
        message(FATAL_ERROR "${_variable} is required")
    endif()
endforeach()

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

set(_packages .)
if(INCLUDE_APP)
    list(APPEND _packages ./app/cmd/app)
endif()

set(_staging_dir "${BINARY_DIR}/go-license-files")
# Arrow's aggregate license includes a license that go-licenses cannot classify.
execute_process(
    COMMAND "${_tool}" save ${_packages}
        --save_path "${_staging_dir}" --force
        --ignore github.com/apache/arrow/go/arrow
    WORKING_DIRECTORY "${SOURCE_DIR}"
    COMMAND_ERROR_IS_FATAL ANY)

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
