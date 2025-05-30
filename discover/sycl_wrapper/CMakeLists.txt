cmake_minimum_required(VERSION 3.21)
project(sycl_wrapper CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find SYCL
if (DEFINED ENV{ONEAPI_ROOT})
    message(STATUS "Using oneAPI Release SYCL compiler (icpx).")
else()
    check_cxx_compiler_flag("-fsycl" SUPPORTS_SYCL)
    if(NOT SUPPORTS_SYCL)
        message(FATAL_ERROR "C++ compiler lacks SYCL support.")
    endif()
    message(STATUS "Using open-source SYCL compiler (clang++).")
endif()

# Create the shared library
add_library(sycl_wrapper SHARED sycl_wrapper.cpp)

# Add SYCL compile options
target_compile_options(sycl_wrapper PRIVATE "-fsycl")
target_link_options(sycl_wrapper PRIVATE "-fsycl")

# Optional: Enable DPC++ compatibility if needed
# Uncomment to enable Intel DPC++ compatibility headers
target_compile_definitions(sycl_wrapper PRIVATE DPCT_COMPATIBILITY_TEMP)

# Set output name
set_target_properties(sycl_wrapper PROPERTIES
    OUTPUT_NAME "sycl_wrapper"
    PREFIX "lib"
)

    # Install the library
    install(TARGETS sycl_wrapper
        LIBRARY DESTINATION ${OLLAMA_INSTALL_DIR}
        RUNTIME DESTINATION ${OLLAMA_INSTALL_DIR}
    )
