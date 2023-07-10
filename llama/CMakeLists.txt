cmake_minimum_required(VERSION 3.12)
project(binding)

include(FetchContent)

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
    GIT_TAG        1d16309
)

FetchContent_MakeAvailable(llama_cpp)

file(COPY ${llama_cpp_SOURCE_DIR}/llama.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${llama_cpp_SOURCE_DIR}/ggml.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

if (LLAMA_METAL)
    file(COPY ${llama_cpp_SOURCE_DIR}/ggml-metal.metal DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../..)
endif()

add_custom_target(copy_libllama ALL COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:llama> ${CMAKE_CURRENT_BINARY_DIR})
add_custom_target(copy_libggml_static ALL COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:ggml_static> ${CMAKE_CURRENT_BINARY_DIR})