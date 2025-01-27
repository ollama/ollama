cmake_minimum_required(VERSION 3.14) # for add_link_options and implicit target directories.
# let cmake could find C++ compiler successfully 
cmake_policy(SET CMP0057 NEW)

project("llama.cann.kernel")

file(GLOB SRC_FILES
    get_row_f32.cpp
    get_row_f16.cpp
    get_row_q4_0.cpp
    get_row_q8_0.cpp
    quantize_f32_q8_0.cpp
    quantize_f16_q8_0.cpp
    quantize_float_to_q4_0.cpp
    dup.cpp
)

set(ASCEND_CANN_PACKAGE_PATH ${CANN_INSTALL_DIR})
set(RUN_MODE "npu" CACHE STRING "run mode: npu/sim")

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the compiler package is installed.")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

ascendc_library(ascendc_kernels SHARED
    ${SRC_FILES}
)

message(STATUS "CANN: compile ascend kernels witch SOC_TYPE:${SOC_TYPE}, SOC_VERSION:${SOC_VERSION}, compile macro:-D${SOC_TYPE_COMPILE_OPTION}.")
ascendc_compile_definitions(ascendc_kernels PRIVATE "-D${SOC_TYPE_COMPILE_OPTION}")
# ascendc_compile_definitions(ascendc_kernels PRIVATE -DASCENDC_DUMP)
