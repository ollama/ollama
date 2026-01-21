include(CheckCXXCompilerFlag)
include("../cmake/common.cmake")

add_compile_definitions(GGML_SCHED_MAX_COPIES=${GGML_SCHED_MAX_COPIES})

# enable libstdc++ assertions for debug builds
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    add_compile_definitions($<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>)
endif()

if (NOT MSVC)
    if (GGML_SANITIZE_THREAD)
        add_compile_options(-fsanitize=thread)
        link_libraries     (-fsanitize=thread)
    endif()

    if (GGML_SANITIZE_ADDRESS)
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        link_libraries     (-fsanitize=address)
    endif()

    if (GGML_SANITIZE_UNDEFINED)
        add_compile_options(-fsanitize=undefined)
        link_libraries     (-fsanitize=undefined)
    endif()
endif()

if (GGML_FATAL_WARNINGS)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        list(APPEND C_FLAGS   -Werror)
        list(APPEND CXX_FLAGS -Werror)
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        add_compile_options(/WX)
    endif()
endif()

if (GGML_ALL_WARNINGS)
    if (NOT MSVC)
        list(APPEND WARNING_FLAGS -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function)
        list(APPEND C_FLAGS       -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes
                                  -Werror=implicit-int -Werror=implicit-function-declaration)
        list(APPEND CXX_FLAGS     -Wmissing-declarations -Wmissing-noreturn)

        list(APPEND C_FLAGS   ${WARNING_FLAGS})
        list(APPEND CXX_FLAGS ${WARNING_FLAGS})

        ggml_get_flags(${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})

        add_compile_options("$<$<COMPILE_LANGUAGE:C>:${C_FLAGS};${GF_C_FLAGS}>"
                            "$<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS};${GF_CXX_FLAGS}>")
    else()
        # todo : msvc
        set(C_FLAGS   "")
        set(CXX_FLAGS "")
    endif()
endif()

if (GGML_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if (result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "IPO is not supported: ${output}")
    endif()
endif()

if (GGML_CCACHE AND NOT CMAKE_C_COMPILER_LAUNCHER AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
    find_program(GGML_CCACHE_FOUND ccache)
    find_program(GGML_SCCACHE_FOUND sccache)

    if (GGML_CCACHE_FOUND OR GGML_SCCACHE_FOUND)
        if(GGML_CCACHE_FOUND)
            set(GGML_CCACHE_VARIANT ccache)
        else()
            set(GGML_CCACHE_VARIANT sccache)
        endif()
        # TODO: should not be set globally
        if (GGML_SYCL AND GGML_CCACHE_FOUND AND WIN32)
            set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "ccache compiler_type=icl")
        else ()
            set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${GGML_CCACHE_VARIANT}")
        endif ()
        set(ENV{CCACHE_SLOPPINESS} time_macros)
        message(STATUS "${GGML_CCACHE_VARIANT} found, compilation results will be cached. Disable with GGML_CCACHE=OFF.")
    else()
        message(STATUS "Warning: ccache not found - consider installing it for faster compilation or disable this warning with GGML_CCACHE=OFF")
    endif ()
endif()

# this version of Apple ld64 is buggy
execute_process(
    COMMAND ${CMAKE_C_COMPILER} ${CMAKE_EXE_LINKER_FLAGS} -Wl,-v
    ERROR_VARIABLE output
    OUTPUT_QUIET
)

if (output MATCHES "dyld-1015\.7")
    add_compile_definitions(HAVE_BUGGY_APPLE_LINKER)
endif()

# architecture specific
# TODO: probably these flags need to be tweaked on some architectures
#       feel free to update the Makefile for your architecture and send a pull request or issue
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if (MSVC)
    string(TOLOWER "${CMAKE_GENERATOR_PLATFORM}" CMAKE_GENERATOR_PLATFORM_LWR)
    message(STATUS "CMAKE_GENERATOR_PLATFORM: ${CMAKE_GENERATOR_PLATFORM}")
else ()
    set(CMAKE_GENERATOR_PLATFORM_LWR "")
endif ()
ggml_get_system_arch()
message(STATUS "GGML_SYSTEM_ARCH: ${GGML_SYSTEM_ARCH}")

if (NOT MSVC)
    if (GGML_STATIC)
        if (UNIX AND NOT APPLE)
            set(CMAKE_FIND_LIBRARY_SUFFIXES ".a;.so")
        endif()
        add_link_options(-static)
        if (MINGW)
            add_link_options(-static-libgcc -static-libstdc++)
        endif()
    endif()
    if (GGML_GPROF)
        add_compile_options(-pg)
    endif()
endif()

#
# POSIX conformance
#

# clock_gettime came in POSIX.1b (1993)
# CLOCK_MONOTONIC came in POSIX.1-2001 / SUSv3 as optional
# posix_memalign came in POSIX.1-2001 / SUSv3
# M_PI is an XSI extension since POSIX.1-2001 / SUSv3, came in XPG1 (1985)

# Somehow in OpenBSD whenever POSIX conformance is specified
# some string functions rely on locale_t availability,
# which was introduced in POSIX.1-2008, forcing us to go higher
if (CMAKE_SYSTEM_NAME MATCHES "OpenBSD")
    add_compile_definitions(_XOPEN_SOURCE=700)
elseif (CMAKE_SYSTEM_NAME MATCHES "AIX")
    # Don't define _XOPEN_SOURCE.  We need _ALL_SOURCE, which is the default,
    # in order to define _SC_PHYS_PAGES.
else()
    add_compile_definitions(_XOPEN_SOURCE=600)
endif()

# Data types, macros and functions related to controlling CPU affinity and
# some memory allocation are available on Linux through GNU extensions in libc
if (CMAKE_SYSTEM_NAME MATCHES "Linux" OR CMAKE_SYSTEM_NAME MATCHES "Android")
    add_compile_definitions(_GNU_SOURCE)
endif()

# RLIMIT_MEMLOCK came in BSD, is not specified in POSIX.1,
# and on macOS its availability depends on enabling Darwin extensions
# similarly on DragonFly, enabling BSD extensions is necessary
if (
    CMAKE_SYSTEM_NAME MATCHES "Darwin" OR
    CMAKE_SYSTEM_NAME MATCHES "iOS"    OR
    CMAKE_SYSTEM_NAME MATCHES "tvOS"   OR
    CMAKE_SYSTEM_NAME MATCHES "DragonFly"
)
    add_compile_definitions(_DARWIN_C_SOURCE)
endif()

# alloca is a non-standard interface that is not visible on BSDs when
# POSIX conformance is specified, but not all of them provide a clean way
# to enable it in such cases
if (CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
    add_compile_definitions(__BSD_VISIBLE)
endif()
if (CMAKE_SYSTEM_NAME MATCHES "NetBSD")
    add_compile_definitions(_NETBSD_SOURCE)
endif()
if (CMAKE_SYSTEM_NAME MATCHES "OpenBSD")
    add_compile_definitions(_BSD_SOURCE)
endif()

if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

# ggml

if (GGML_BACKEND_DL AND NOT BUILD_SHARED_LIBS)
    message(FATAL_ERROR "GGML_BACKEND_DL requires BUILD_SHARED_LIBS")
endif()

add_library(ggml-base
            ../include/ggml.h
            ../include/ggml-alloc.h
            ../include/ggml-backend.h
            ../include/ggml-cpp.h
            ../include/ggml-opt.h
            ../include/gguf.h
            ggml.c
            ggml.cpp
            ggml-alloc.c
            ggml-backend.cpp
            ggml-opt.cpp
            ggml-threading.cpp
            ggml-threading.h
            ggml-quants.c
            ggml-quants.h
            mem_hip.cpp
            mem_nvml.cpp
            mem_dxgi_pdh.cpp
            gguf.cpp)

set_target_properties(ggml-base PROPERTIES
    VERSION ${GGML_VERSION}
    SOVERSION ${GGML_VERSION_MAJOR}
)

target_include_directories(ggml-base PRIVATE .)
if (GGML_BACKEND_DL)
    target_compile_definitions(ggml-base PUBLIC GGML_BACKEND_DL)
endif()

if (GGML_SCHED_NO_REALLOC)
    target_compile_definitions(ggml-base PUBLIC GGML_SCHED_NO_REALLOC)
endif()

add_library(ggml
            ggml-backend-reg.cpp)
add_library(ggml::ggml ALIAS ggml)

set_target_properties(ggml PROPERTIES
    VERSION ${GGML_VERSION}
    SOVERSION ${GGML_VERSION_MAJOR}
)

if (GGML_BACKEND_DIR)
    if (NOT GGML_BACKEND_DL)
        message(FATAL_ERROR "GGML_BACKEND_DIR requires GGML_BACKEND_DL")
    endif()
    target_compile_definitions(ggml PUBLIC GGML_BACKEND_DIR="${GGML_BACKEND_DIR}")
endif()

target_link_libraries(ggml PUBLIC ggml-base)

if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_libraries(ggml PRIVATE dl)
endif()

function(ggml_add_backend_library backend)
    if (GGML_BACKEND_DL)
        add_library(${backend} MODULE ${ARGN})
        # write the shared library to the output directory
        set_target_properties(${backend} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
        target_compile_definitions(${backend} PRIVATE GGML_BACKEND_DL)
        add_dependencies(ggml ${backend})
        if (GGML_BACKEND_DIR)
            install(TARGETS ${backend} LIBRARY DESTINATION ${GGML_BACKEND_DIR})
        else()
            install(TARGETS ${backend} LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR})
        endif()
    else()
        add_library(${backend} ${ARGN})
        target_link_libraries(ggml PUBLIC ${backend})
        install(TARGETS ${backend} LIBRARY)
    endif()

    target_link_libraries(${backend} PRIVATE ggml-base)
    target_include_directories(${backend} PRIVATE ..)

    if (${BUILD_SHARED_LIBS})
        target_compile_definitions(${backend} PRIVATE GGML_BACKEND_BUILD)
        target_compile_definitions(${backend} PUBLIC  GGML_BACKEND_SHARED)
    endif()

    # Set versioning properties for all backend libraries
    # Building a MODULE library with a version is not supported on macOS (https://gitlab.kitware.com/cmake/cmake/-/issues/20782)
    if (NOT (APPLE AND GGML_BACKEND_DL))
        set_target_properties(${backend} PROPERTIES
            VERSION ${GGML_VERSION}
            SOVERSION ${GGML_VERSION_MAJOR}
        )
    endif()

    if(NOT GGML_AVAILABLE_BACKENDS)
        set(GGML_AVAILABLE_BACKENDS "${backend}"
            CACHE INTERNAL "List of backends for cmake package")
    else()
        list(FIND GGML_AVAILABLE_BACKENDS "${backend}" has_backend)
        if(has_backend EQUAL -1)
            set(GGML_AVAILABLE_BACKENDS "${GGML_AVAILABLE_BACKENDS};${backend}"
                CACHE INTERNAL "List of backends for cmake package")
        endif()
    endif()
endfunction()

function(ggml_add_backend backend)
    string(TOUPPER "GGML_${backend}" backend_id)
    if (${backend_id})
        string(TOLOWER "ggml-${backend}" backend_target)
        add_subdirectory(${backend_target})
        message(STATUS "Including ${backend} backend")
        if (NOT GGML_BACKEND_DL)
            string(TOUPPER "GGML_USE_${backend}" backend_use)
            target_compile_definitions(ggml PUBLIC ${backend_use})
        endif()
    endif()
endfunction()

function(ggml_add_cpu_backend_variant tag_name)
    set(GGML_CPU_TAG_NAME ${tag_name})
    # other: OPENMP LLAMAFILE CPU_HBM
    if (GGML_SYSTEM_ARCH STREQUAL "x86")
        foreach (feat NATIVE
                      SSE42
                      AVX AVX2 BMI2 AVX_VNNI FMA F16C
                      AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16
                      AMX_TILE AMX_INT8 AMX_BF16)
            set(GGML_${feat} OFF)
        endforeach()

        foreach (feat ${ARGN})
            set(GGML_${feat} ON)
        endforeach()
    elseif (GGML_SYSTEM_ARCH STREQUAL "ARM")
        foreach (feat ${ARGN})
            set(GGML_INTERNAL_${feat} ON)
        endforeach()
    elseif (GGML_SYSTEM_ARCH STREQUAL "PowerPC")
        foreach (feat ${ARGN})
            set(GGML_INTERNAL_${feat} ON)
        endforeach()
    elseif (GGML_SYSTEM_ARCH STREQUAL "s390x")
        foreach (feat VXE2 NNPA)
            set(GGML_INTERNAL_${feat} OFF)
        endforeach()

        foreach (feat ${ARGN})
            set(GGML_INTERNAL_${feat} ON)
        endforeach()
    elseif (GGML_SYSTEM_ARCH STREQUAL "riscv64")
        foreach (feat RVV)
            set(GGML_INTERNAL_${feat} OFF)
        endforeach()

        foreach (feat ${ARGN})
            set(GGML_INTERNAL_${feat} ON)
        endforeach()
    endif()

    ggml_add_cpu_backend_variant_impl(${tag_name})
    add_dependencies(ggml-cpu ggml-cpu-${tag_name})
endfunction()

ggml_add_backend(CPU)

if (GGML_CPU_ALL_VARIANTS)
    if (NOT GGML_BACKEND_DL)
        message(FATAL_ERROR "GGML_CPU_ALL_VARIANTS requires GGML_BACKEND_DL")
    elseif (GGML_CPU_ARM_ARCH)
        message(FATAL_ERROR "Cannot use both GGML_CPU_ARM_ARCH and GGML_CPU_ALL_VARIANTS")
    endif()
    add_custom_target(ggml-cpu)
    if (GGML_SYSTEM_ARCH STREQUAL "x86")
        ggml_add_cpu_backend_variant(x64)
        ggml_add_cpu_backend_variant(sse42        SSE42)
        ggml_add_cpu_backend_variant(sandybridge  SSE42 AVX)
        ggml_add_cpu_backend_variant(haswell      SSE42 AVX F16C AVX2 BMI2 FMA)
        ggml_add_cpu_backend_variant(skylakex     SSE42 AVX F16C AVX2 BMI2 FMA AVX512)
        ggml_add_cpu_backend_variant(icelake      SSE42 AVX F16C AVX2 BMI2 FMA AVX512 AVX512_VBMI AVX512_VNNI)
        ggml_add_cpu_backend_variant(alderlake    SSE42 AVX F16C AVX2 BMI2 FMA AVX_VNNI)
    elseif(GGML_SYSTEM_ARCH STREQUAL "ARM")
        if (CMAKE_SYSTEM_NAME MATCHES "Linux")
            # Many of these features are optional so we build versions with popular
            # combinations and name the backends based on the version they were
            # first released with
            ggml_add_cpu_backend_variant(armv8.0_1)
            ggml_add_cpu_backend_variant(armv8.2_1    DOTPROD)
            ggml_add_cpu_backend_variant(armv8.2_2    DOTPROD FP16_VECTOR_ARITHMETIC)
            ggml_add_cpu_backend_variant(armv8.2_3    DOTPROD FP16_VECTOR_ARITHMETIC SVE)
            ggml_add_cpu_backend_variant(armv8.6_1    DOTPROD FP16_VECTOR_ARITHMETIC SVE MATMUL_INT8)
            ggml_add_cpu_backend_variant(armv8.6_2    DOTPROD FP16_VECTOR_ARITHMETIC SVE MATMUL_INT8 SVE2)
            ggml_add_cpu_backend_variant(armv9.2_1    DOTPROD FP16_VECTOR_ARITHMETIC SVE MATMUL_INT8 SME)
            ggml_add_cpu_backend_variant(armv9.2_2    DOTPROD FP16_VECTOR_ARITHMETIC SVE MATMUL_INT8 SVE2 SME)
        elseif (CMAKE_SYSTEM_NAME MATCHES "Android")
            # Android-specific backends with SoC-compatible feature sets
            ggml_add_cpu_backend_variant(android_armv8.0_1)
            ggml_add_cpu_backend_variant(android_armv8.2_1    DOTPROD)
            ggml_add_cpu_backend_variant(android_armv8.2_2    DOTPROD FP16_VECTOR_ARITHMETIC)
            ggml_add_cpu_backend_variant(android_armv8.6_1    DOTPROD FP16_VECTOR_ARITHMETIC MATMUL_INT8)
        elseif (APPLE)
            ggml_add_cpu_backend_variant(apple_m1             DOTPROD)
            ggml_add_cpu_backend_variant(apple_m2_m3          DOTPROD MATMUL_INT8)
            ggml_add_cpu_backend_variant(apple_m4             DOTPROD MATMUL_INT8 NOSVE SME)
        else()
            message(FATAL_ERROR "Unsupported ARM target OS: ${CMAKE_SYSTEM_NAME}")
        endif()
    elseif (GGML_SYSTEM_ARCH STREQUAL "PowerPC")
        if (CMAKE_SYSTEM_NAME MATCHES "Linux")
            ggml_add_cpu_backend_variant(power0)
            ggml_add_cpu_backend_variant(power7_1       POWER7)
            ggml_add_cpu_backend_variant(power7_2       POWER7  VSX)
            ggml_add_cpu_backend_variant(power8_1       POWER8)
            ggml_add_cpu_backend_variant(power8_2       POWER8  VSX)
            ggml_add_cpu_backend_variant(power9         POWER9  VSX)
            ggml_add_cpu_backend_variant(power10        POWER10 VSX)
            ggml_add_cpu_backend_variant(power11        POWER11 VSX)
        else()
            message(FATAL_ERROR "Unsupported PowerPC target OS: ${CMAKE_SYSTEM_NAME}")
        endif()
    elseif (GGML_SYSTEM_ARCH STREQUAL "s390x")
        if (CMAKE_SYSTEM_NAME MATCHES "Linux")
            ggml_add_cpu_backend_variant(z15    Z15 VXE2)
            ggml_add_cpu_backend_variant(z16    Z16 VXE2 NNPA)
        else()
            message(FATAL_ERROR "Unsupported s390x target OS: ${CMAKE_SYSTEM_NAME}")
        endif()
    elseif (GGML_SYSTEM_ARCH STREQUAL "riscv64")
        if (CMAKE_SYSTEM_NAME MATCHES "Linux")
            ggml_add_cpu_backend_variant(riscv64_0)
            ggml_add_cpu_backend_variant(riscv64_v   RVV)
        else()
            message(FATAL_ERROR "Unsupported RISC-V target OS: ${CMAKE_SYSTEM_NAME}")
        endif()
    else()
        message(FATAL_ERROR "GGML_CPU_ALL_VARIANTS not yet supported with ${GGML_SYSTEM_ARCH} on ${CMAKE_SYSTEM_NAME}")
    endif()
elseif (GGML_CPU)
    ggml_add_cpu_backend_variant_impl("")
endif()

ggml_add_backend(BLAS)
ggml_add_backend(CANN)
ggml_add_backend(CUDA)
ggml_add_backend(HIP)
ggml_add_backend(METAL)
ggml_add_backend(MUSA)
ggml_add_backend(RPC)
ggml_add_backend(SYCL)
ggml_add_backend(Vulkan)
ggml_add_backend(WebGPU)
ggml_add_backend(zDNN)
ggml_add_backend(OpenCL)
ggml_add_backend(Hexagon)
ggml_add_backend(ZenDNN)

foreach (target ggml-base ggml)
    target_include_directories(${target} PUBLIC    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include> $<INSTALL_INTERFACE:include>)
    target_compile_features   (${target} PRIVATE c_std_11 cxx_std_17) # don't bump
endforeach()

target_link_libraries(ggml-base PRIVATE Threads::Threads)

find_library(MATH_LIBRARY m)
if (MATH_LIBRARY)
    if (NOT WIN32 OR NOT DEFINED ENV{ONEAPI_ROOT})
        target_link_libraries(ggml-base PRIVATE m)
    endif()
endif()

if (CMAKE_SYSTEM_NAME MATCHES "Android")
    target_link_libraries(ggml-base PRIVATE dl)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "visionOS")
    target_compile_definitions(ggml-base PUBLIC _DARWIN_C_SOURCE)
endif()

if (BUILD_SHARED_LIBS)
    foreach (target ggml-base ggml)
        set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_compile_definitions(${target} PRIVATE GGML_BUILD)
        target_compile_definitions(${target} PUBLIC  GGML_SHARED)
    endforeach()
endif()
