include(FetchContent)

set(MLX_C_BUILD_EXAMPLES OFF)

set(MLX_BUILD_GGUF OFF)
set(MLX_BUILD_SAFETENSORS OFF)

execute_process(
  COMMAND
    zsh "-c"
    "echo \"__METAL_VERSION__\" | xcrun -sdk macosx metal ${XCRUN_FLAGS} -E -x metal -P - | tail -1 | tr -d '\n'"
  OUTPUT_VARIABLE MLX_METAL_VERSION COMMAND_ERROR_IS_FATAL ANY)

if(NOT MLX_METAL_VERSION)
    message(STATUS "`xcrun metal` error. Setting MLX_BUILD_METAL=OFF")
    set(MLX_BUILD_METAL OFF)
endif()

FetchContent_Declare(
  mlx-c
  GIT_REPOSITORY "https://github.com/ml-explore/mlx-c.git"
  GIT_TAG v0.1.0)
FetchContent_MakeAvailable(mlx-c)

set_target_output_directory(mlx)
set_target_output_directory(mlxc)
