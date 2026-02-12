cmake_minimum_required(VERSION 3.5)

project(mlx)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/dist" CACHE PATH "" FORCE)
endif()

set(MLX_BUILD_GGUF OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_SAFETENSORS ON CACHE BOOL "" FORCE)
set(MLX_C_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)

set(CMAKE_INSTALL_RPATH "@loader_path")

include(FetchContent)

set(MLX_C_GIT_TAG "v0.4.1" CACHE STRING "")

FetchContent_Declare(
  mlx-c
  GIT_REPOSITORY "https://github.com/ml-explore/mlx-c.git"
  GIT_TAG ${MLX_C_GIT_TAG}
)

FetchContent_MakeAvailable(mlx-c)
