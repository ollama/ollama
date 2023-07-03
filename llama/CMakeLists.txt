cmake_minimum_required(VERSION 3.10) 
include(FetchContent)

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
    GIT_TAG        master
)

FetchContent_MakeAvailable(llama_cpp)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(LLAMA_METAL ON)
    add_compile_definitions(GGML_USE_METAL)
endif()

project(binding)

add_library(binding ${CMAKE_CURRENT_SOURCE_DIR}/binding/binding.cpp ${llama_cpp_SOURCE_DIR}/examples/common.cpp)
target_compile_features(binding PRIVATE cxx_std_11)
target_include_directories(binding PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(binding PRIVATE ${llama_cpp_SOURCE_DIR})
target_include_directories(binding PRIVATE ${llama_cpp_SOURCE_DIR}/examples)
target_link_libraries(binding llama ggml_static)

configure_file(${llama_cpp_SOURCE_DIR}/ggml-metal.metal ${CMAKE_CURRENT_BINARY_DIR}/ggml-metal.metal COPYONLY)
add_custom_target(copy_libllama ALL COMMAND ${CMAKE_COMMAND} -E copy_if_different ${llama_cpp_BINARY_DIR}/libllama.a ${CMAKE_CURRENT_BINARY_DIR})
add_custom_target(copy_libggml_static ALL COMMAND ${CMAKE_COMMAND} -E copy_if_different ${llama_cpp_BINARY_DIR}/libggml_static.a ${CMAKE_CURRENT_BINARY_DIR})