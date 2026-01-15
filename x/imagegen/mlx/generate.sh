#!/bin/bash
set -e

# This script generates dynamic loading wrappers for MLX-C
# It uses the MLX-C headers that CMake downloads to ensure version consistency

# Find MLX-C headers from CMake build
CMAKE_MLX_C_PATH="../../../build/_deps/mlx-c-src/mlx/c"

if [ ! -d "$CMAKE_MLX_C_PATH" ]; then
    echo "WARNING: MLX-C headers not found at $CMAKE_MLX_C_PATH"
    echo ""
    echo "The generated mlx_wrappers.h is checked into git, so this is only needed"
    echo "when updating to a new MLX-C version."
    echo ""
    echo "To regenerate wrappers, run CMake first to download dependencies:"
    echo "  cmake -B build"
    echo ""
    exit 0
fi

echo "Generating wrappers from: $CMAKE_MLX_C_PATH"
python3 generate_wrappers.py "$CMAKE_MLX_C_PATH" mlx_wrappers.h

echo "Done! Generated mlx_wrappers.h successfully."
