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

# Read MLX-C version from top-level file (shared with imagegen CMakeLists)
file(READ "${CMAKE_SOURCE_DIR}/MLX_C_VERSION" MLX_C_GIT_TAG)
string(STRIP "${MLX_C_GIT_TAG}" MLX_C_GIT_TAG)

FetchContent_Declare(
  mlx-c
  GIT_REPOSITORY "https://github.com/ml-explore/mlx-c.git"
  GIT_TAG ${MLX_C_GIT_TAG}
)

FetchContent_MakeAvailable(mlx-c)

# Sync vendored headers with fetched version
file(GLOB _mlx_c_hdrs "${mlx-c_SOURCE_DIR}/mlx/c/*.h")
file(COPY ${_mlx_c_hdrs} DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/include/mlx/c/")
