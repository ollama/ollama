# Ollama specific CMakefile to include in llama.cpp/examples/server

set(TARGET ext_server)
option(LLAMA_SERVER_VERBOSE "Build verbose logging option for Server" ON)
add_library(${TARGET} STATIC ../../../ext_server/ext_server.cpp)
target_include_directories(${TARGET} PRIVATE ../../common)
target_include_directories(${TARGET} PRIVATE ../..)
target_include_directories(${TARGET} PRIVATE ../../..)
target_compile_features(${TARGET} PRIVATE cxx_std_11)
target_compile_definitions(${TARGET} PUBLIC LLAMA_SERVER_LIBRARY=1)
target_link_libraries(${TARGET} PRIVATE common llama llava ${CMAKE_THREAD_LIBS_INIT})
target_compile_definitions(${TARGET} PRIVATE
    SERVER_VERBOSE=$<BOOL:${LLAMA_SERVER_VERBOSE}>
)

if (BUILD_SHARED_LIBS)
    set_target_properties(ext_server PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(ext_server PRIVATE LLAMA_SHARED LLAMA_BUILD)
    add_library(ext_server_shared SHARED $<TARGET_OBJECTS:ext_server>)
    target_link_libraries(ext_server_shared PRIVATE ggml llama llava common ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS ext_server_shared LIBRARY)
endif()

if (CUDAToolkit_FOUND)
    target_include_directories(${TARGET} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    if (WIN32)
        target_link_libraries(ext_server_shared PRIVATE nvml)
    endif()
endif()