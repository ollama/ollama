cmake_minimum_required(VERSION 3.5)

project(mlx)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/dist" CACHE PATH "" FORCE)
endif()

set(MLX_BUILD_GGUF OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_SAFETENSORS ON CACHE BOOL "" FORCE)
set(MLX_C_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)

if(APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path")
else()
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()

include(FetchContent)

# Read MLX-C version from top-level file.
file(READ "${CMAKE_CURRENT_LIST_DIR}/../../../MLX_C_VERSION" MLX_C_GIT_TAG)
string(STRIP "${MLX_C_GIT_TAG}" MLX_C_GIT_TAG)

FetchContent_Declare(
  mlx-c
  GIT_REPOSITORY "https://github.com/ml-explore/mlx-c.git"
  GIT_TAG ${MLX_C_GIT_TAG}
)

FetchContent_MakeAvailable(mlx-c)

# Stamp libmlx with the pinned commit's git describe so the built library
# identifies its exact MLX commit, not just mlx/version.h's MAJOR.MINOR.PATCH.
if(TARGET mlx_version AND DEFINED FETCHCONTENT_SOURCE_DIR_MLX)
    execute_process(
        COMMAND git describe --tags --first-parent --abbrev=7 --long --dirty --always
        WORKING_DIRECTORY ${mlx_SOURCE_DIR}
        OUTPUT_VARIABLE _mlx_git_version
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _mlx_git_result
    )
    if(_mlx_git_result EQUAL 0 AND _mlx_git_version)
        # Strip leading "v" prefix for consistency
        string(REGEX REPLACE "^v" "" _mlx_git_version "${_mlx_git_version}")
        get_target_property(_mlx_defs mlx_version COMPILE_DEFINITIONS)
        list(FILTER _mlx_defs EXCLUDE REGEX "^MLX_VERSION=")
        set_target_properties(mlx_version PROPERTIES COMPILE_DEFINITIONS "${_mlx_defs}")
        target_compile_definitions(mlx_version PRIVATE "MLX_VERSION=\"${_mlx_git_version}\"")
        message(STATUS "MLX version: ${_mlx_git_version}")
    endif()
endif()
