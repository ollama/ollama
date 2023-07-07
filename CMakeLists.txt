cmake_minimum_required(VERSION 3.12)
project(ollama)

include(FetchContent)

FetchContent_Declare(
    "llama.cpp"
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
    GIT_TAG        55dbb91
)

FetchContent_MakeAvailable(llama.cpp)

add_custom_target(
    ollama
    ALL
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/ggml-metal.metal
    COMMAND
        ${CMAKE_COMMAND} -E
        env
        CGO_CPPFLAGS='-I${llama.cpp_SOURCE_DIR}'
        CGO_LDFLAGS='-L${llama.cpp_BINARY_DIR} -lllama -lggml_static -lm -lstdc++'
        CGO_CXXFLAGS='-std=c++11'
        --
        go build .
    WORKING_DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_command(
    OUTPUT
        ${CMAKE_CURRENT_SOURCE_DIR}/ggml-metal.metal
    COMMAND
        ${CMAKE_COMMAND} -E
        copy_if_different
        ${llama.cpp_SOURCE_DIR}/ggml-metal.metal
        ${CMAKE_CURRENT_SOURCE_DIR}/ggml-metal.metal
    WORKING_DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}
)

add_dependencies(ollama llama ggml_static)
