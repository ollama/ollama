#!/bin/sh
#
# Deduplicate CUDA libraries across mlx_* and cuda_* directories
# This script finds identical .so* files in mlx_cuda_* directories that exist
# in corresponding cuda_* directories and replaces them with symlinks.
#

set -eu

if [ $# -eq 0 ]; then
    echo "ERROR: No directory specified" >&2
    echo "Usage: $0 <base_directory>" >&2
    exit 1
fi

base_dir="$1"

if [ ! -d "${base_dir}" ]; then
    echo "ERROR: Directory ${base_dir} does not exist" >&2
    exit 1
fi

echo "Deduplicating CUDA libraries in ${base_dir}..."

# Find all mlx_cuda_* directories
for mlx_dir in "${base_dir}"/lib/ollama/mlx_cuda_*; do
    [ -d "${mlx_dir}" ] || continue

    # Extract CUDA version (e.g., v12, v13)
    cuda_version=$(basename "${mlx_dir}" | sed 's/mlx_cuda_//')
    cuda_dir="${base_dir}/lib/ollama/cuda_${cuda_version}"

    # Skip if corresponding cuda_* directory doesn't exist
    [ -d "${cuda_dir}" ] || continue

    echo "  Checking ${mlx_dir} against ${cuda_dir}..."

    # Find all .so* files in mlx directory
    find "${mlx_dir}" -type f -name "*.so*" | while read mlx_file; do
        filename=$(basename "${mlx_file}")
        cuda_file="${cuda_dir}/${filename}"

        # Skip if file doesn't exist in cuda directory
        [ -f "${cuda_file}" ] || continue

        # Compare checksums
        mlx_sum=$(sha256sum "${mlx_file}" | awk '{print $1}')
        cuda_sum=$(sha256sum "${cuda_file}" | awk '{print $1}')

        if [ "${mlx_sum}" = "${cuda_sum}" ]; then
            echo "    Deduplicating ${filename}"
            # Calculate relative path from mlx_dir to cuda_dir
            rel_path="../cuda_${cuda_version}/${filename}"
            rm -f "${mlx_file}"
            ln -s "${rel_path}" "${mlx_file}"
        fi
    done
done

echo "Deduplication complete"
