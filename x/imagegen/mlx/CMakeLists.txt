include(FetchContent)

# Read MLX-C version from top-level file (shared with Dockerfile)
file(READ "${CMAKE_SOURCE_DIR}/MLX_C_VERSION" MLX_C_GIT_TAG)
string(STRIP "${MLX_C_GIT_TAG}" MLX_C_GIT_TAG)

# Read MLX version from top-level file
file(READ "${CMAKE_SOURCE_DIR}/MLX_VERSION" MLX_GIT_TAG)
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

# Forward cuDNN environment variables to cmake variables so MLX's FindCUDNN.cmake
# can find them via HINTS ${CUDNN_INCLUDE_PATH} / ${CUDNN_LIBRARY_PATH}.
if(DEFINED ENV{CUDNN_INCLUDE_PATH} AND NOT CUDNN_INCLUDE_PATH)
    set(CUDNN_INCLUDE_PATH "$ENV{CUDNN_INCLUDE_PATH}" CACHE PATH "cuDNN include path")
    message(STATUS "Using CUDNN_INCLUDE_PATH from environment: ${CUDNN_INCLUDE_PATH}")
endif()
if(DEFINED ENV{CUDNN_LIBRARY_PATH} AND NOT CUDNN_LIBRARY_PATH)
    set(CUDNN_LIBRARY_PATH "$ENV{CUDNN_LIBRARY_PATH}" CACHE PATH "cuDNN library path")
    message(STATUS "Using CUDNN_LIBRARY_PATH from environment: ${CUDNN_LIBRARY_PATH}")
endif()

# Enable CUDA backend if CUDA architectures are specified and CUDA compiler is available
if(MLX_CUDA_ARCHITECTURES AND CMAKE_CUDA_COMPILER)
    set(MLX_BUILD_CUDA ON CACHE BOOL "Build CUDA backend for MLX" FORCE)
    message(STATUS "Enabling MLX CUDA backend with architectures: ${MLX_CUDA_ARCHITECTURES}")
elseif(MLX_CUDA_ARCHITECTURES)
    message(WARNING "MLX_CUDA_ARCHITECTURES specified but CUDA compiler not found, CUDA backend will be disabled")
endif()

# Allow local source overrides via environment variables
# Resolve to absolute paths so FetchContent doesn't break on relative dirs.
if(DEFINED ENV{OLLAMA_MLX_SOURCE})
    get_filename_component(_mlx_src "$ENV{OLLAMA_MLX_SOURCE}" ABSOLUTE BASE_DIR ${CMAKE_SOURCE_DIR})
    set(FETCHCONTENT_SOURCE_DIR_MLX "${_mlx_src}" CACHE PATH "" FORCE)
    message(STATUS "Using local MLX source: ${_mlx_src}")
endif()
if(DEFINED ENV{OLLAMA_MLX_C_SOURCE})
    get_filename_component(_mlx_c_src "$ENV{OLLAMA_MLX_C_SOURCE}" ABSOLUTE BASE_DIR ${CMAKE_SOURCE_DIR})
    set(FETCHCONTENT_SOURCE_DIR_MLX-C "${_mlx_c_src}" CACHE PATH "" FORCE)
    message(STATUS "Using local MLX-C source: ${_mlx_c_src}")
endif()

# Pre-declare mlx so our pinned version takes precedence over the one
# hardcoded in mlx-c's CMakeLists.txt (first FetchContent_Declare wins).
FetchContent_Declare(
    mlx
    GIT_REPOSITORY "https://github.com/ml-explore/mlx.git"
    GIT_TAG ${MLX_GIT_TAG}
)

FetchContent_Declare(
    mlx-c
    GIT_REPOSITORY "https://github.com/ml-explore/mlx-c.git"
    GIT_TAG ${MLX_C_GIT_TAG}
)
FetchContent_MakeAvailable(mlx-c)

# Sync vendored headers with fetched version
file(GLOB _mlx_c_hdrs "${mlx-c_SOURCE_DIR}/mlx/c/*.h")
file(COPY ${_mlx_c_hdrs} DESTINATION "${CMAKE_SOURCE_DIR}/x/mlxrunner/mlx/include/mlx/c/")

# Regenerate Go/C shim wrappers from the (possibly updated) headers.
# Skip during cross-compilation — the generated files are arch-independent.
if(CMAKE_SYSTEM_PROCESSOR STREQUAL CMAKE_HOST_SYSTEM_PROCESSOR OR NOT APPLE)
    find_program(GO_EXECUTABLE go REQUIRED)
    message(STATUS "Regenerating MLX Go wrappers")

    # CGo's probe compilation is sensitive to CGO_CFLAGS/CGO_CXXFLAGS and CC.
    # Clear them so go generate uses default compiler settings:
    # - On Windows, CC may contain spaces (e.g., "C:/Program Files/.../cl.exe")
    #   which breaks CGo's CC parsing.
    # - On macOS, CGO_CFLAGS with -mmacosx-version-min breaks header search
    #   when cmake also sets CMAKE_OSX_DEPLOYMENT_TARGET.
    set(_SAVE_CC "$ENV{CC}")
    set(_SAVE_CGO_CFLAGS "$ENV{CGO_CFLAGS}")
    set(_SAVE_CGO_CXXFLAGS "$ENV{CGO_CXXFLAGS}")
    set(ENV{CC} "")
    set(ENV{CGO_CFLAGS} "")
    set(ENV{CGO_CXXFLAGS} "")

    execute_process(
        COMMAND ${GO_EXECUTABLE} generate ./x/...
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMAND_ERROR_IS_FATAL ANY
    )

    set(ENV{CC} "${_SAVE_CC}")
    set(ENV{CGO_CFLAGS} "${_SAVE_CGO_CFLAGS}")
    set(ENV{CGO_CXXFLAGS} "${_SAVE_CGO_CXXFLAGS}")
else()
    message(STATUS "Skipping MLX Go wrapper generation (cross-compiling)")
endif()

# For local dev builds, override MLX_VERSION with git describe output
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
        message(STATUS "MLX version (local dev): ${_mlx_git_version}")
    endif()
endif()

set_target_output_directory(mlx)
set_target_output_directory(mlxc)
