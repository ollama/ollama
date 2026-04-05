set(TARGET cpp-httplib)

find_package(Threads REQUIRED)

add_library(${TARGET} STATIC httplib.cpp httplib.h)
if (NOT MSVC)
    # disable warnings in 3rd party code
    target_compile_options(${TARGET} PRIVATE -w)
endif()

target_link_libraries  (${TARGET} PRIVATE Threads::Threads)

if (WIN32 AND NOT MSVC)
    target_link_libraries(${TARGET} PRIVATE ws2_32)
endif()

target_compile_features(${TARGET} PRIVATE cxx_std_17)

target_compile_definitions(${TARGET} PRIVATE
    # increase max payload length to allow use of larger context size
    CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH=1048576
    # increase backlog size to avoid connection resets for >> 1 slots
    CPPHTTPLIB_LISTEN_BACKLOG=512
    # increase max URI length to handle longer prompts in query string
    CPPHTTPLIB_REQUEST_URI_MAX_LENGTH=32768
    # disable Nagle's algorithm
    CPPHTTPLIB_TCP_NODELAY=1
)

set(OPENSSL_NO_ASM ON CACHE BOOL "Disable OpenSSL ASM code when building BoringSSL or LibreSSL")

if (LLAMA_BUILD_BORINGSSL)
    set(FIPS OFF CACHE BOOL "Enable FIPS (BoringSSL)")

    set(BORINGSSL_GIT "https://boringssl.googlesource.com/boringssl" CACHE STRING "BoringSSL git repository")
    set(BORINGSSL_VERSION "0.20251002.0" CACHE STRING "BoringSSL version")

    message(STATUS "Fetching BoringSSL version ${BORINGSSL_VERSION}")

    set(BORINGSSL_ARGS
        GIT_REPOSITORY ${BORINGSSL_GIT}
        GIT_TAG        ${BORINGSSL_VERSION}
    )
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.28)
        list(APPEND BORINGSSL_ARGS EXCLUDE_FROM_ALL)
    endif()

    include(FetchContent)
    FetchContent_Declare(boringssl ${BORINGSSL_ARGS})

    set(SAVED_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
    set(SAVED_BUILD_TESTING ${BUILD_TESTING})

    set(BUILD_SHARED_LIBS OFF)
    set(BUILD_TESTING OFF)

    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.28)
        FetchContent_MakeAvailable(boringssl)
    else()
        FetchContent_GetProperties(boringssl)
        if(NOT boringssl_POPULATED)
            FetchContent_Populate(boringssl)
            add_subdirectory(${boringssl_SOURCE_DIR} ${boringssl_BINARY_DIR} EXCLUDE_FROM_ALL)
        endif()
    endif()

    set(BUILD_SHARED_LIBS ${SAVED_BUILD_SHARED_LIBS})
    set(BUILD_TESTING ${SAVED_BUILD_TESTING})

    set(CPPHTTPLIB_OPENSSL_SUPPORT TRUE)
    target_link_libraries(${TARGET} PUBLIC ssl crypto)

elseif (LLAMA_BUILD_LIBRESSL)
    set(LIBRESSL_VERSION "4.2.1" CACHE STRING "LibreSSL version")

    message(STATUS "Fetching LibreSSL version ${LIBRESSL_VERSION}")

    set(LIBRESSL_ARGS
        URL "https://cdn.openbsd.org/pub/OpenBSD/LibreSSL/libressl-${LIBRESSL_VERSION}.tar.gz"
    )
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        list(APPEND LIBRESSL_ARGS DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    endif()

    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.28)
        list(APPEND LIBRESSL_ARGS EXCLUDE_FROM_ALL)
    endif()

    include(FetchContent)
    FetchContent_Declare(libressl ${LIBRESSL_ARGS})

    set(SAVED_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
    set(SAVED_BUILD_TESTING ${BUILD_TESTING})

    set(BUILD_SHARED_LIBS OFF)
    set(BUILD_TESTING OFF)

    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.28)
        FetchContent_MakeAvailable(libressl)
    else()
        FetchContent_GetProperties(libressl)
        if(NOT libressl_POPULATED)
            FetchContent_Populate(libressl)
            add_subdirectory(${libressl_SOURCE_DIR} ${libressl_BINARY_DIR} EXCLUDE_FROM_ALL)
        endif()
    endif()

    set(BUILD_SHARED_LIBS ${SAVED_BUILD_SHARED_LIBS})
    set(BUILD_TESTING ${SAVED_BUILD_TESTING})

    set(CPPHTTPLIB_OPENSSL_SUPPORT TRUE)
    target_link_libraries(${TARGET} PUBLIC ssl crypto)

elseif (LLAMA_OPENSSL)
    find_package(OpenSSL)
    if (OpenSSL_FOUND)
        include(CheckCSourceCompiles)
        set(SAVED_CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES})
        set(CMAKE_REQUIRED_INCLUDES ${OPENSSL_INCLUDE_DIR})
        check_c_source_compiles("
        #include <openssl/opensslv.h>
        #if defined(OPENSSL_IS_BORINGSSL) || defined(LIBRESSL_VERSION_NUMBER)
        #    if OPENSSL_VERSION_NUMBER < 0x1010107f
        #        error bad version
        #    endif
        #else
        #    if OPENSSL_VERSION_NUMBER < 0x30000000L
        #        error bad version
        #    endif
        #endif
        int main() { return 0; }
        " OPENSSL_VERSION_SUPPORTED)
        set(CMAKE_REQUIRED_INCLUDES ${SAVED_CMAKE_REQUIRED_INCLUDES})
        if (OPENSSL_VERSION_SUPPORTED)
            message(STATUS "OpenSSL found: ${OPENSSL_VERSION}")
            set(CPPHTTPLIB_OPENSSL_SUPPORT TRUE)
            target_link_libraries(${TARGET} PUBLIC OpenSSL::SSL OpenSSL::Crypto)
        endif()
    else()
        message(STATUS "OpenSSL not found, SSL support disabled")
    endif()
endif()

if (CPPHTTPLIB_OPENSSL_SUPPORT)
    target_compile_definitions(${TARGET} PUBLIC CPPHTTPLIB_OPENSSL_SUPPORT) # used in server.cpp
    if (APPLE AND CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        target_compile_definitions(${TARGET} PRIVATE CPPHTTPLIB_USE_CERTS_FROM_MACOSX_KEYCHAIN)
        find_library(CORE_FOUNDATION_FRAMEWORK CoreFoundation REQUIRED)
        find_library(SECURITY_FRAMEWORK Security REQUIRED)
        target_link_libraries(${TARGET} PUBLIC ${CORE_FOUNDATION_FRAMEWORK} ${SECURITY_FRAMEWORK})
    endif()
    if (WIN32 AND NOT MSVC)
        target_link_libraries(${TARGET} PUBLIC crypt32)
    endif()
endif()
