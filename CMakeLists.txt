cmake_minimum_required(VERSION 3.14)  # 3.11 or later for FetchContent, but some features might require newer versions

project(llama_cpp)

include(FetchContent)

FetchContent_Declare(
    llama_cpp_gguf
    GIT_REPOSITORY    https://github.com/ggerganov/llama.cpp.git
    GIT_TAG           6381d4e
)

FetchContent_Declare(
    llama_cpp_ggml
    GIT_REPOSITORY    https://github.com/ggerganov/llama.cpp.git
    GIT_TAG           dadbed9
)

FetchContent_MakeAvailable(llama_cpp_ggml)

add_subdirectory(${llama_cpp_ggml_SOURCE_DIR}/examples EXCLUDE_FROM_ALL)
add_executable(llama_cpp ${llama_cpp_ggml_SOURCE_DIR}/examples/server/server.cpp)
include_directories(${llama_cpp_ggml_SOURCE_DIR})
include_directories(${llama_cpp_ggml_SOURCE_DIR}/examples)
target_compile_features(llama_cpp PRIVATE cxx_std_11)
target_link_libraries(llama_cpp PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})

if (APPLE)
    add_executable(llama_cpp_metal ${llama_cpp_ggml_SOURCE_DIR}/examples/server/server.cpp)
    target_compile_options(llama_cpp_metal PRIVATE -DLLAMA_STATIC=ON -DLLAMA_METAL=ON -DGGML_USE_METAL=1)
    target_compile_features(llama_cpp_metal PRIVATE cxx_std_11)
    target_link_libraries(llama_cpp_metal PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
    configure_file(${llama_cpp_SOURCE_DIR}/ggml-metal.metal ${CMAKE_BINARY_DIR}/ggml-metal.metal COPYONLY)
else()
    add_executable(llama_cpp_cublas ${llama_cpp_ggml_SOURCE_DIR}/examples/server/server.cpp)
    target_compile_definitions(llama_cpp_cublas PRIVATE -DLLAMA_STATIC=ON -DLLAMA_CUBLAS=ON)
    target_compile_options(llama_cpp_cublas PRIVATE -DLLAMA_CUBLAS=ON -DLLAMA_STATIC=ON)
    target_compile_features(llama_cpp_cublas PRIVATE cxx_std_11)
    target_link_libraries(llama_cpp_cublas PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
endif()
