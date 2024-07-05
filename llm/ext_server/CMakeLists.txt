
set(TARGET ollama_llama_server)
option(LLAMA_SERVER_VERBOSE "Build verbose logging option for Server" ON)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_executable(${TARGET} server.cpp utils.hpp json.hpp httplib.h)
target_compile_definitions(${TARGET} PRIVATE
    SERVER_VERBOSE=$<BOOL:${LLAMA_SERVER_VERBOSE}>
)
target_link_libraries(${TARGET} PRIVATE ggml llama common llava ${CMAKE_THREAD_LIBS_INIT})
install(TARGETS ollama_llama_server ggml llama
        RUNTIME DESTINATION "${CMAKE_BINARY_DIR}/bin"
        LIBRARY DESTINATION "${CMAKE_BINARY_DIR}/bin"
        COMPONENT ollama_llama_server)
if (WIN32)
    TARGET_LINK_LIBRARIES(${TARGET} PRIVATE ws2_32)
endif()
target_compile_features(${TARGET} PRIVATE cxx_std_11)