#!/bin/sh
# Incremental build script - rebuilds only the ollama binary after a full build_linux.sh run
# Use this when you've only changed Go code and don't want to rebuild the native libraries

set -eu

. $(dirname $0)/env.sh

# Ensure a full build has been run first
echo "WARNING: This script is for incremental builds after running build_linux.sh"
echo "It will only replace the ollama binary, not the native libraries"
echo ""

mkdir -p .tmp/build_output

# Build the ollama binary only
docker buildx build \
    --output type=local,dest=./.tmp/build_output/ \
    --platform=${PLATFORM} \
    ${OLLAMA_COMMON_BUILD_ARGS} \
    --target archive-ollama-only \
    -f Dockerfile \
    .

# Copy binaries to correct locations
if echo $PLATFORM | grep "," > /dev/null; then
    # Multi-arch build - subdirectories created by buildx
    for arch_dir in .tmp/build_output/linux_amd64 .tmp/build_output/linux_arm64; do
        if [ -d "${arch_dir}" ]; then
            arch=$(basename ${arch_dir})
            target_dir="./dist/${arch}"
            if [ ! -d "${target_dir}" ]; then
                echo "ERROR: ${target_dir} does not exist. Run build_linux.sh first." >&2
                exit 1
            fi
            cp "${arch_dir}/bin/ollama" "${target_dir}/bin/ollama"
            echo "Updated ${target_dir}/bin/ollama"
        fi
    done
else
    # Single-arch build
    if [ ! -d "./dist" ]; then
        echo "ERROR: ./dist does not exist. Run build_linux.sh first." >&2
        exit 1
    fi
    cp .tmp/build_output/bin/ollama "./dist/bin/ollama"
    echo "Updated ./dist/bin/ollama"
fi

rm -rf .tmp/build_output

echo ""
echo "=========================================="
echo "Incremental build complete!"
echo "=========================================="
echo ""
echo "NOTE: Tar files in ./dist have NOT been updated."
echo "To update the tar files, run build_linux.sh after committing your changes."
echo ""