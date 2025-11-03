function(ggml_get_flags CCID CCVER)
    set(C_FLAGS "")
    set(CXX_FLAGS "")

    if (CCID MATCHES "Clang")
        set(C_FLAGS   -Wunreachable-code-break -Wunreachable-code-return)
        set(CXX_FLAGS -Wunreachable-code-break -Wunreachable-code-return -Wmissing-prototypes -Wextra-semi)

        if (
            (CCID STREQUAL "Clang"      AND CCVER VERSION_GREATER_EQUAL 3.8.0) OR
            (CCID STREQUAL "AppleClang" AND CCVER VERSION_GREATER_EQUAL 7.3.0)
        )
            list(APPEND C_FLAGS -Wdouble-promotion)
        endif()
    elseif (CCID STREQUAL "GNU")
        set(C_FLAGS   -Wdouble-promotion)
        set(CXX_FLAGS -Wno-array-bounds)

        if (CCVER VERSION_GREATER_EQUAL 8.1.0)
            list(APPEND CXX_FLAGS -Wextra-semi)
        endif()
    endif()

    set(GF_C_FLAGS   ${C_FLAGS}   PARENT_SCOPE)
    set(GF_CXX_FLAGS ${CXX_FLAGS} PARENT_SCOPE)
endfunction()

function(ggml_get_system_arch)
    if (CMAKE_OSX_ARCHITECTURES      STREQUAL "arm64" OR
        CMAKE_GENERATOR_PLATFORM_LWR STREQUAL "arm64" OR
        (NOT CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_GENERATOR_PLATFORM_LWR AND
            CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm.*|ARM64)$"))
        set(GGML_SYSTEM_ARCH "ARM" PARENT_SCOPE)
    elseif (CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64" OR
            CMAKE_GENERATOR_PLATFORM_LWR MATCHES "^(x86_64|i686|amd64|x64|win32)$" OR
            (NOT CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_GENERATOR_PLATFORM_LWR AND
            CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|i686|AMD64|amd64)$"))
        set(GGML_SYSTEM_ARCH "x86" PARENT_SCOPE)
    elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc|power")
        set(GGML_SYSTEM_ARCH "PowerPC" PARENT_SCOPE)
    elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "loongarch64")
        set(GGML_SYSTEM_ARCH "loongarch64"  PARENT_SCOPE)
    elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "riscv64")
        set(GGML_SYSTEM_ARCH "riscv64" PARENT_SCOPE)
    elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "s390x")
        set(GGML_SYSTEM_ARCH "s390x" PARENT_SCOPE)
    else()
        set(GGML_SYSTEM_ARCH "UNKNOWN" PARENT_SCOPE)
    endif()
endfunction()
