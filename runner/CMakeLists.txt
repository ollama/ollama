cmake_minimum_required(VERSION 3.12)
project(binding)

include(FetchContent)

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
    GIT_TAG        2d8b76a
)

FetchContent_MakeAvailable(llama_cpp)

set(TARGET runner)
add_executable(${TARGET} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp ${llama_cpp_SOURCE_DIR}/examples/common.cpp ${llama_cpp_SOURCE_DIR}/examples/console.cpp ${llama_cpp_SOURCE_DIR}/examples/grammar-parser.cpp)
target_include_directories(${TARGET} PRIVATE ${llama_cpp_SOURCE_DIR} ${llama_cpp_SOURCE_DIR}/examples)
target_link_libraries(${TARGET} llama ggml_static)

if (LLAMA_METAL)
    configure_file(${llama_cpp_SOURCE_DIR}/ggml-metal.metal ${CMAKE_CURRENT_BINARY_DIR}/../../ggml-metal.metal COPYONLY)
endif()
