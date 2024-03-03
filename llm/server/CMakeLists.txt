cmake_minimum_required(VERSION 3.14)

project(llm)

include(FetchContent)

set(add_token_patch
    git apply ${CMAKE_CURRENT_SOURCE_DIR}/patches/add_token.patch
)

set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/build/llama.cpp")

FetchContent_Declare(
  llama_cpp
  GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
  GIT_TAG        c29af7e2252d288f2ea58a7d437c1cb7c0abf160

  # this could be risky if the patch doesn't apply
  PATCH_COMMAND ${add_token_patch} || true
)

FetchContent_MakeAvailable(llama_cpp)
add_subdirectory(${llama_cpp_SOURCE_DIR}/examples/llava)

# code signing
function(sign target)
    if(APPLE)
        if(DEFINED ENV{APPLE_IDENTITY})
            add_custom_command(TARGET ${target} POST_BUILD
                COMMAND codesign
                        -f 
                        --timestamp
                        --deep
                        --options=runtime
                        --sign "$ENV{APPLE_IDENTITY}"
                        --identifier ai.ollama.ollama
                        $<TARGET_FILE:${target}>
                COMMENT "Signing macOS binary: ${target}"
            )
        endif()
    elseif(WIN32)
        find_program(SIGNTOOL_EXE NAMES signtool PATHS "C:\\Program Files (x86)\\Windows Kits\\8.1\\bin\\x64" NO_DEFAULT_PATH)
        set(KEY_CONTAINER "$ENV{KEY_CONTAINER}")
        set(OLLAMA_CERT "$ENV{OLLAMA_CERT}")

        if(SIGNTOOL_EXE AND KEY_CONTAINER AND OLLAMA_CERT)
            add_custom_command(TARGET ${target} POST_BUILD
                COMMAND "${SIGNTOOL_EXE}"
                        "sign"
                        "/v"
                        "/fd" "sha256"
                        "/t" "http://timestamp.digicert.com"
                        "/f" "${OLLAMA_CERT}"
                        "/csp" "Google Cloud KMS Provider"
                        "/kc" "${KEY_CONTAINER}"
                        "$<TARGET_FILE:${target}>"
                COMMENT "Signing Windows binary: ${target}"
            )
        endif()
    endif()
endfunction()

set(CMAKE_CUDA_ARCHITECTURES "50;52;61;70;75;80")

function(gzip target)
    set(gzip_target "gzip_${target}")
    add_custom_target(${gzip_target} ALL
        COMMAND gzip -k -f ${target}
        COMMENT "Gzipping ${target}"
        VERBATIM
    )
    add_dependencies(${gzip_target} ${target})
endfunction()

function(link_windows_libraries target)
    if (WIN32)
        target_link_libraries(${target} PRIVATE ws2_32)
    endif()
endfunction()

add_executable(server ${llama_cpp_SOURCE_DIR}/examples/server/server.cpp ${llama_cpp_SOURCE_DIR})
target_compile_definitions(server PRIVATE)
target_link_libraries(server PRIVATE common llava ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(server PRIVATE cxx_std_17)
link_windows_libraries(server)
sign(server)
gzip(server)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    configure_file(${llama_cpp_SOURCE_DIR}/ggml-metal.metal ${CMAKE_BINARY_DIR}/ggml-metal.metal COPYONLY)
endif()

# TODO: ROCm
