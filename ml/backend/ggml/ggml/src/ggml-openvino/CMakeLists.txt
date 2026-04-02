find_package(OpenVINO REQUIRED)
find_package(OpenCL REQUIRED)

include("${OpenVINO_DIR}/../3rdparty/tbb/lib/cmake/TBB/TBBConfig.cmake")

file(GLOB_RECURSE GGML_HEADERS_OPENVINO "*.h" "*.hpp")
file(GLOB_RECURSE GGML_SOURCES_OPENVINO "*.cpp")

ggml_add_backend_library(ggml-openvino
    ${GGML_SOURCES_OPENVINO}
    ${GGML_HEADERS_OPENVINO}
)

target_link_libraries(ggml-openvino PRIVATE openvino::runtime TBB::tbb OpenCL::OpenCL)

if (GGML_OPENVINO)
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "amd64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64")
    else()
        message(FATAL_ERROR "OpenVINO: OpenVINO toolkit supports x86-64 and arm64 but not ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
endif()
