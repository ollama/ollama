cmake_minimum_required(VERSION 3.21)

project(Ollama C CXX)

set(CMAKE_BUILD_TYPE Release)
set(BUILD_SHARED_LIBS ON)

function(set_target_output_directory _target)
    if(TARGET ${_target})
        set_target_properties(${_target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        )
    endif()
endfunction()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml)

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/mlx)
endif()
