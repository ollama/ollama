find_package(Vulkan COMPONENTS glslc REQUIRED)

if (Vulkan_FOUND)
    message(STATUS "Vulkan found")

    ggml_add_backend_library(ggml-vulkan
                             ggml-vulkan.cpp
                             ../../include/ggml-vulkan.h
                            )

    # Compile a test shader to determine whether GL_NV_cooperative_matrix2 is supported.
    # If it's not, there will be an error to stderr.
    # If it's supported, set a define to indicate that we should compile those shaders
    execute_process(COMMAND ${Vulkan_GLSLC_EXECUTABLE} -o - -fshader-stage=compute --target-env=vulkan1.3 "${CMAKE_CURRENT_SOURCE_DIR}/vulkan-shaders/test_coopmat2_support.comp"
                    OUTPUT_VARIABLE glslc_output
                    ERROR_VARIABLE glslc_error)

    if (${glslc_error} MATCHES ".*extension not supported: GL_NV_cooperative_matrix2.*")
        message(STATUS "GL_NV_cooperative_matrix2 not supported by glslc")
    else()
        message(STATUS "GL_NV_cooperative_matrix2 supported by glslc")
        add_compile_definitions(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    endif()

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
    endif()

    if (GGML_VULKAN_PERF)
        add_compile_definitions(GGML_VULKAN_PERF)
    endif()

    if (GGML_VULKAN_VALIDATE)
        add_compile_definitions(GGML_VULKAN_VALIDATE)
    endif()

    if (GGML_VULKAN_RUN_TESTS)
        add_compile_definitions(GGML_VULKAN_RUN_TESTS)
    endif()

    add_subdirectory(vulkan-shaders)

    set (_ggml_vk_genshaders_cmd vulkan-shaders-gen)
    set (_ggml_vk_header     ${CMAKE_CURRENT_BINARY_DIR}/ggml-vulkan-shaders.hpp)
    set (_ggml_vk_source     ${CMAKE_CURRENT_BINARY_DIR}/ggml-vulkan-shaders.cpp)
    set (_ggml_vk_input_dir  ${CMAKE_CURRENT_SOURCE_DIR}/vulkan-shaders)
    set (_ggml_vk_output_dir ${CMAKE_CURRENT_BINARY_DIR}/vulkan-shaders.spv)

    file(GLOB _ggml_vk_shader_deps "${_ggml_vk_input_dir}/*.comp")

    add_custom_command(
        OUTPUT ${_ggml_vk_header}
                ${_ggml_vk_source}

        COMMAND "$<TARGET_FILE_DIR:vulkan-shaders-gen>/${_ggml_vk_genshaders_cmd}"
            --glslc      ${Vulkan_GLSLC_EXECUTABLE}
            --input-dir  ${_ggml_vk_input_dir}
            --output-dir ${_ggml_vk_output_dir}
            --target-hpp ${_ggml_vk_header}
            --target-cpp ${_ggml_vk_source}
            --no-clean

        DEPENDS ${_ggml_vk_shader_deps} ${_ggml_vk_genshaders_cmd}
        COMMENT "Generate vulkan shaders"
    )

    target_sources(ggml-vulkan PRIVATE ${_ggml_vk_source} ${_ggml_vk_header})

else()
    message(WARNING "Vulkan not found")
endif()
