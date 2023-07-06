cmake_minimum_required(VERSION 3.12)
project(binding)

include(FetchContent)

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
    GIT_TAG        55dbb91
)

FetchContent_MakeAvailable(llama_cpp)

add_library(binding ${CMAKE_CURRENT_SOURCE_DIR}/binding/binding.cpp ${llama_cpp_SOURCE_DIR}/examples/common.cpp)
target_include_directories(binding PRIVATE ${llama_cpp_SOURCE_DIR}/examples)
target_link_libraries(binding llama ggml_static)

if (LLAMA_METAL)
    configure_file(${llama_cpp_SOURCE_DIR}/ggml-metal.metal ${CMAKE_CURRENT_BINARY_DIR}/../../ggml-metal.metal COPYONLY)
endif()

add_custom_target(copy_libllama ALL COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:llama> ${CMAKE_CURRENT_BINARY_DIR})
add_custom_target(copy_libggml_static ALL COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:ggml_static> ${CMAKE_CURRENT_BINARY_DIR})
