cmake_minimum_required(VERSION 3.24)

project(OllamaMLXBackendOrderTest NONE)

if(NOT DEFINED TEST_SCENARIO)
    message(FATAL_ERROR "TEST_SCENARIO is required")
endif()

include(${CMAKE_CURRENT_LIST_DIR}/../../mlx-backend-order.cmake)

if(TEST_SCENARIO STREQUAL "both_reversed")
    # Define v4 first to prove the ordering does not depend on list order.
    add_custom_target(ollama-mlx-metal_v4)
    add_custom_target(ollama-mlx-metal_v3)
elseif(TEST_SCENARIO STREQUAL "metal_v3_only")
    add_custom_target(ollama-mlx-metal_v3)
elseif(TEST_SCENARIO STREQUAL "metal_v4_only")
    add_custom_target(ollama-mlx-metal_v4)
else()
    message(FATAL_ERROR "unknown TEST_SCENARIO: ${TEST_SCENARIO}")
endif()

ollama_order_mlx_metal_builds()

if(TEST_SCENARIO STREQUAL "both_reversed")
    get_property(_dependencies
        TARGET ollama-mlx-metal_v4
        PROPERTY MANUALLY_ADDED_DEPENDENCIES)
    if(NOT "ollama-mlx-metal_v3" IN_LIST _dependencies)
        message(FATAL_ERROR
            "metal_v4 does not depend on metal_v3: ${_dependencies}")
    endif()
elseif(TARGET ollama-mlx-metal_v3)
    get_property(_dependencies
        TARGET ollama-mlx-metal_v3
        PROPERTY MANUALLY_ADDED_DEPENDENCIES)
    if(_dependencies)
        message(FATAL_ERROR
            "single metal_v3 target has unexpected dependencies: ${_dependencies}")
    endif()
elseif(TARGET ollama-mlx-metal_v4)
    get_property(_dependencies
        TARGET ollama-mlx-metal_v4
        PROPERTY MANUALLY_ADDED_DEPENDENCIES)
    if(_dependencies)
        message(FATAL_ERROR
            "single metal_v4 target has unexpected dependencies: ${_dependencies}")
    endif()
endif()
