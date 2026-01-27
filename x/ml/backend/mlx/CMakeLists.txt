include(FetchContent)

# Read MLX version from top-level file (shared with Dockerfile)
file(READ "${CMAKE_SOURCE_DIR}/MLX_VERSION" MLX_C_GIT_TAG)
string(STRIP "${MLX_C_GIT_TAG}" MLX_C_GIT_TAG)

# Read MLX core version from top-level file
file(READ "${CMAKE_SOURCE_DIR}/MLX_CORE_VERSION" MLX_GIT_TAG)
string(STRIP "${MLX_GIT_TAG}" MLX_GIT_TAG)

set(MLX_C_BUILD_EXAMPLES OFF)

set(MLX_BUILD_GGUF OFF)
set(MLX_BUILD_SAFETENSORS ON)

function(set_target_output_directory _target)
    if(TARGET ${_target})
        set_target_properties(${_target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${OLLAMA_BUILD_DIR}
            LIBRARY_OUTPUT_DIRECTORY ${OLLAMA_BUILD_DIR}
            ARCHIVE_OUTPUT_DIRECTORY ${OLLAMA_BUILD_DIR}
        )
    endif()
endfunction()

# Check for Metal support (macOS only)
if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    execute_process(
      COMMAND
        zsh "-c"
        "echo \"__METAL_VERSION__\" | xcrun -sdk macosx metal ${XCRUN_FLAGS} -E -x metal -P - | tail -1 | tr -d '\n'"
      OUTPUT_VARIABLE MLX_METAL_VERSION COMMAND_ERROR_IS_FATAL ANY)

    if(NOT MLX_METAL_VERSION)
        message(STATUS "`xcrun metal` error. Setting MLX_BUILD_METAL=OFF")
        set(MLX_BUILD_METAL OFF)
    endif()
else()
    # On Linux, disable Metal backend
    message(STATUS "Non-macOS platform detected. Setting MLX_BUILD_METAL=OFF")
    set(MLX_BUILD_METAL OFF)
endif()

# Map CMAKE_CUDA_ARCHITECTURES to MLX_CUDA_ARCHITECTURES if not explicitly set
if(NOT MLX_CUDA_ARCHITECTURES AND CMAKE_CUDA_ARCHITECTURES)
    set(MLX_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
    message(STATUS "Using CMAKE_CUDA_ARCHITECTURES for MLX: ${MLX_CUDA_ARCHITECTURES}")
endif()

# Enable CUDA backend if CUDA architectures are specified and CUDA compiler is available
if(MLX_CUDA_ARCHITECTURES AND CMAKE_CUDA_COMPILER)
    set(MLX_BUILD_CUDA ON CACHE BOOL "Build CUDA backend for MLX" FORCE)
    message(STATUS "Enabling MLX CUDA backend with architectures: ${MLX_CUDA_ARCHITECTURES}")
elseif(MLX_CUDA_ARCHITECTURES)
    message(WARNING "MLX_CUDA_ARCHITECTURES specified but CUDA compiler not found, CUDA backend will be disabled")
endif()

# On Windows, we need to apply patches to MLX for MSVC+NVCC compatibility
# Pre-declare MLX with patches BEFORE mlx-c declares it, so our version takes precedence
if(WIN32)
    find_package(Git REQUIRED)

    # Collect all patch files
    file(GLOB MLX_PATCHES "${CMAKE_CURRENT_SOURCE_DIR}/patches/*.patch")
    list(SORT MLX_PATCHES)

    # Build the patch command - apply each patch in order
    set(MLX_PATCH_COMMAND ${GIT_EXECUTABLE} -c user.name=ollama -c user.email=ollama@ollama.com am --3way)
    foreach(patch ${MLX_PATCHES})
        list(APPEND MLX_PATCH_COMMAND "${patch}")
    endforeach()

    message(STATUS "MLX Windows patches will be applied: ${MLX_PATCHES}")

    FetchContent_Declare(
        mlx
        GIT_REPOSITORY "https://github.com/ml-explore/mlx.git"
        GIT_TAG ${MLX_GIT_TAG}
        PATCH_COMMAND ${MLX_PATCH_COMMAND}
    )
endif()

# On Windows, we need to apply patches to mlx-c for MSVC compatibility
if(WIN32)
    # Collect all mlx-c patch files
    file(GLOB MLX_C_PATCHES "${CMAKE_CURRENT_SOURCE_DIR}/patches-c/*.patch")
    list(SORT MLX_C_PATCHES)

    if(MLX_C_PATCHES)
        # Build the patch command - apply each patch in order
        set(MLX_C_PATCH_COMMAND ${GIT_EXECUTABLE} -c user.name=ollama -c user.email=ollama@ollama.com am --3way)
        foreach(patch ${MLX_C_PATCHES})
            list(APPEND MLX_C_PATCH_COMMAND "${patch}")
        endforeach()

        message(STATUS "MLX-C Windows patches will be applied: ${MLX_C_PATCHES}")

        FetchContent_Declare(
            mlx-c
            GIT_REPOSITORY "https://github.com/ml-explore/mlx-c.git"
            GIT_TAG ${MLX_C_GIT_TAG}
            PATCH_COMMAND ${MLX_C_PATCH_COMMAND}
        )
    else()
        FetchContent_Declare(
            mlx-c
            GIT_REPOSITORY "https://github.com/ml-explore/mlx-c.git"
            GIT_TAG ${MLX_C_GIT_TAG}
        )
    endif()
else()
    FetchContent_Declare(
        mlx-c
        GIT_REPOSITORY "https://github.com/ml-explore/mlx-c.git"
        GIT_TAG ${MLX_C_GIT_TAG}
    )
endif()
FetchContent_MakeAvailable(mlx-c)

set_target_output_directory(mlx)
set_target_output_directory(mlxc)
