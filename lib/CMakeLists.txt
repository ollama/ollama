cmake_minimum_required(VERSION 3.10) 
include(FetchContent)

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
    GIT_TAG        master
)

FetchContent_MakeAvailable(llama_cpp)

project(binding)

set(LLAMA_METAL ON CACHE BOOL "Enable Llama Metal by default on macOS")

add_library(binding binding.cpp ${llama_cpp_SOURCE_DIR}/examples/common.cpp)
target_compile_features(binding PRIVATE cxx_std_11)
target_include_directories(binding PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(binding PRIVATE ${llama_cpp_SOURCE_DIR})
target_include_directories(binding PRIVATE ${llama_cpp_SOURCE_DIR}/examples)
target_link_libraries(binding llama ggml_static)
