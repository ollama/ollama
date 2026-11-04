if(NOT DEFINED MLX_C_HEADERS_DIR OR NOT IS_DIRECTORY "${MLX_C_HEADERS_DIR}")
    message(FATAL_ERROR "MLX_C_HEADERS_DIR does not exist: ${MLX_C_HEADERS_DIR}")
endif()
if(NOT DEFINED MLX_C_HEADERS_DEST OR "${MLX_C_HEADERS_DEST}" STREQUAL "")
    message(FATAL_ERROR "MLX_C_HEADERS_DEST is required")
endif()

file(GLOB _mlx_c_headers LIST_DIRECTORIES false "${MLX_C_HEADERS_DIR}/*.h")
if(NOT _mlx_c_headers)
    message(FATAL_ERROR "No MLX-C headers found in ${MLX_C_HEADERS_DIR}")
endif()

file(MAKE_DIRECTORY "${MLX_C_HEADERS_DEST}")
file(COPY ${_mlx_c_headers} DESTINATION "${MLX_C_HEADERS_DEST}")
