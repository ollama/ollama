cmake_minimum_required(VERSION 3.19)
cmake_policy(SET CMP0114 NEW)
cmake_policy(SET CMP0116 NEW)
if (POLICY CMP0147)
    # Parallel build custom build steps
    cmake_policy(SET CMP0147 NEW)
endif()

find_package(Vulkan COMPONENTS glslc REQUIRED)

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Parallel build object files
    add_definitions(/MP)
endif()

function(detect_host_compiler)
    if (CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
        find_program(HOST_C_COMPILER NAMES cl gcc clang NO_CMAKE_FIND_ROOT_PATH)
        find_program(HOST_CXX_COMPILER NAMES cl g++ clang++ NO_CMAKE_FIND_ROOT_PATH)
    else()
        find_program(HOST_C_COMPILER NAMES gcc clang NO_CMAKE_FIND_ROOT_PATH)
        find_program(HOST_CXX_COMPILER NAMES g++ clang++ NO_CMAKE_FIND_ROOT_PATH)
    endif()
    set(HOST_C_COMPILER "${HOST_C_COMPILER}" PARENT_SCOPE)
    set(HOST_CXX_COMPILER "${HOST_CXX_COMPILER}" PARENT_SCOPE)
endfunction()

# Function to test shader extension support
# Parameters:
#  EXTENSION_NAME - Name of the extension to test (e.g., "GL_EXT_integer_dot_product")
#  TEST_SHADER_FILE - Path to the test shader file
#  RESULT_VARIABLE - Name of the variable to set (ON/OFF) based on test result
function(test_shader_extension_support EXTENSION_NAME TEST_SHADER_FILE RESULT_VARIABLE)
    execute_process(
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} -o - -fshader-stage=compute --target-env=vulkan1.3 "${TEST_SHADER_FILE}"
        OUTPUT_VARIABLE glslc_output
        ERROR_VARIABLE glslc_error
    )

    if (${glslc_error} MATCHES ".*extension not supported: ${EXTENSION_NAME}.*")
        message(STATUS "${EXTENSION_NAME} not supported by glslc")
        set(${RESULT_VARIABLE} OFF PARENT_SCOPE)
    else()
        message(STATUS "${EXTENSION_NAME} supported by glslc")
        set(${RESULT_VARIABLE} ON PARENT_SCOPE)
        add_compile_definitions(${RESULT_VARIABLE})

        # Ensure the extension support is forwarded to vulkan-shaders-gen
        list(APPEND VULKAN_SHADER_GEN_CMAKE_ARGS -D${RESULT_VARIABLE}=ON)
        set(VULKAN_SHADER_GEN_CMAKE_ARGS "${VULKAN_SHADER_GEN_CMAKE_ARGS}" PARENT_SCOPE)
    endif()
endfunction()

if (Vulkan_FOUND)
    message(STATUS "Vulkan found")

    ggml_add_backend_library(ggml-vulkan
                             ggml-vulkan.cpp
                             ../../include/ggml-vulkan.h
                            )

    set(VULKAN_SHADER_GEN_CMAKE_ARGS "")

    # Test all shader extensions
    test_shader_extension_support(
        "GL_KHR_cooperative_matrix"
        "${CMAKE_CURRENT_SOURCE_DIR}/vulkan-shaders/feature-tests/coopmat.comp"
        "GGML_VULKAN_COOPMAT_GLSLC_SUPPORT"
    )

    test_shader_extension_support(
        "GL_NV_cooperative_matrix2"
        "${CMAKE_CURRENT_SOURCE_DIR}/vulkan-shaders/feature-tests/coopmat2.comp"
        "GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT"
    )

    test_shader_extension_support(
        "GL_EXT_integer_dot_product"
        "${CMAKE_CURRENT_SOURCE_DIR}/vulkan-shaders/feature-tests/integer_dot.comp"
        "GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT"
    )

    test_shader_extension_support(
        "GL_EXT_bfloat16"
        "${CMAKE_CURRENT_SOURCE_DIR}/vulkan-shaders/feature-tests/bfloat16.comp"
        "GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT"
    )

    target_link_libraries(ggml-vulkan PRIVATE Vulkan::Vulkan)
    target_include_directories(ggml-vulkan PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

    # Workaround to the "can't dereference invalidated vector iterator" bug in clang-cl debug build
    # Posssibly relevant: https://stackoverflow.com/questions/74748276/visual-studio-no-displays-the-correct-length-of-stdvector
    if (MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        add_compile_definitions(_ITERATOR_DEBUG_LEVEL=0)
    endif()

    if (GGML_VULKAN_CHECK_RESULTS)
        add_compile_definitions(GGML_VULKAN_CHECK_RESULTS)
    endif()

    if (GGML_VULKAN_DEBUG)
        add_compile_definitions(GGML_VULKAN_DEBUG)
    endif()

    if (GGML_VULKAN_MEMORY_DEBUG)
        add_compile_definitions(GGML_VULKAN_MEMORY_DEBUG)
    endif()

    if (GGML_VULKAN_SHADER_DEBUG_INFO)
        add_compile_definitions(GGML_VULKAN_SHADER_DEBUG_INFO)
        list(APPEND VULKAN_SHADER_GEN_CMAKE_ARGS -DGGML_VULKAN_SHADER_DEBUG_INFO=ON)
    endif()

    if (GGML_VULKAN_VALIDATE)
        add_compile_definitions(GGML_VULKAN_VALIDATE)
    endif()

    if (GGML_VULKAN_RUN_TESTS)
        add_compile_definitions(GGML_VULKAN_RUN_TESTS)
    endif()

    # Set up toolchain for host compilation whether cross-compiling or not
    if (CMAKE_CROSSCOMPILING)
        if (GGML_VULKAN_SHADERS_GEN_TOOLCHAIN)
            set(HOST_CMAKE_TOOLCHAIN_FILE ${GGML_VULKAN_SHADERS_GEN_TOOLCHAIN})
        else()
            detect_host_compiler()
            if (NOT HOST_C_COMPILER OR NOT HOST_CXX_COMPILER)
                message(FATAL_ERROR "Host compiler not found")
            else()
                message(STATUS "Host compiler: ${HOST_C_COMPILER} ${HOST_CXX_COMPILER}")
            endif()
            configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/host-toolchain.cmake.in ${CMAKE_BINARY_DIR}/host-toolchain.cmake @ONLY)
            set(HOST_CMAKE_TOOLCHAIN_FILE ${CMAKE_BINARY_DIR}/host-toolchain.cmake)
        endif()
    else()
        # For non-cross-compiling, use empty toolchain (use host compiler)
        set(HOST_CMAKE_TOOLCHAIN_FILE "")
    endif()

    include(ExternalProject)

    if (CMAKE_CROSSCOMPILING)
        list(APPEND VULKAN_SHADER_GEN_CMAKE_ARGS -DCMAKE_TOOLCHAIN_FILE=${HOST_CMAKE_TOOLCHAIN_FILE})
        message(STATUS "vulkan-shaders-gen toolchain file: ${HOST_CMAKE_TOOLCHAIN_FILE}")
    endif()

    ExternalProject_Add(
        vulkan-shaders-gen
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vulkan-shaders
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/$<CONFIG>
                   -DCMAKE_INSTALL_BINDIR=.
                   -DCMAKE_BUILD_TYPE=$<CONFIG>
                   ${VULKAN_SHADER_GEN_CMAKE_ARGS}

        BUILD_COMMAND ${CMAKE_COMMAND} --build . --config $<CONFIG>
        BUILD_ALWAYS  TRUE

        # NOTE: When DESTDIR is set using Makefile generators and
        # "make install" triggers the build step, vulkan-shaders-gen
        # would be installed into the DESTDIR prefix, so it is unset
        # to ensure that does not happen.

        INSTALL_COMMAND ${CMAKE_COMMAND} -E env --unset=DESTDIR
                        ${CMAKE_COMMAND} --install . --config $<CONFIG>
    )

    set (_ggml_vk_host_suffix $<IF:$<STREQUAL:${CMAKE_HOST_SYSTEM_NAME},Windows>,.exe,>)
    set (_ggml_vk_genshaders_dir "${CMAKE_BINARY_DIR}/$<CONFIG>")
    set (_ggml_vk_genshaders_cmd "${_ggml_vk_genshaders_dir}/vulkan-shaders-gen${_ggml_vk_host_suffix}")
    set (_ggml_vk_header     "${CMAKE_CURRENT_BINARY_DIR}/ggml-vulkan-shaders.hpp")
    set (_ggml_vk_input_dir  "${CMAKE_CURRENT_SOURCE_DIR}/vulkan-shaders")
    set (_ggml_vk_output_dir "${CMAKE_CURRENT_BINARY_DIR}/vulkan-shaders.spv")

    file(GLOB _ggml_vk_shader_files CONFIGURE_DEPENDS "${_ggml_vk_input_dir}/*.comp")

    # Because external projects do not provide source-level tracking,
    # the vulkan-shaders-gen sources need to be explicitly added to
    # ensure that changes will cascade into shader re-generation.

    file(GLOB _ggml_vk_shaders_gen_sources
              CONFIGURE_DEPENDS "${_ggml_vk_input_dir}/*.cpp"
                                "${_ggml_vk_input_dir}/*.h")

    add_custom_command(
        OUTPUT ${_ggml_vk_header}
        COMMAND ${_ggml_vk_genshaders_cmd}
            --output-dir ${_ggml_vk_output_dir}
            --target-hpp ${_ggml_vk_header}
        DEPENDS ${_ggml_vk_shaders_gen_sources}
                vulkan-shaders-gen
        COMMENT "Generate vulkan shaders header"
    )
    target_sources(ggml-vulkan PRIVATE ${_ggml_vk_header})

    foreach (file_full ${_ggml_vk_shader_files})
        get_filename_component(file ${file_full} NAME)
        set (_ggml_vk_target_cpp "${CMAKE_CURRENT_BINARY_DIR}/${file}.cpp")

        add_custom_command(
            OUTPUT  ${_ggml_vk_target_cpp}
            DEPFILE ${_ggml_vk_target_cpp}.d
            COMMAND ${_ggml_vk_genshaders_cmd}
                --glslc      ${Vulkan_GLSLC_EXECUTABLE}
                --source     ${file_full}
                --output-dir ${_ggml_vk_output_dir}
                --target-hpp ${_ggml_vk_header}
                --target-cpp ${_ggml_vk_target_cpp}
            DEPENDS ${file_full}
                    ${_ggml_vk_shaders_gen_sources}
                    vulkan-shaders-gen
            COMMENT "Generate vulkan shaders for ${file}"
        )
        target_sources(ggml-vulkan PRIVATE ${_ggml_vk_target_cpp})
    endforeach()

else()
    message(WARNING "Vulkan not found")
endif()
