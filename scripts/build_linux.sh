#!/bin/sh
#
# Mac ARM users, rosetta can be flaky, so to use a remote x86 builder.
# Use the docker-container driver with the bundled buildkit GC config
# for improved cache behavior.
#
# docker context create amd64 --docker host=ssh://mybuildhost
# docker buildx create --name mybuilder \
#     --driver docker-container \
#     --config ./scripts/buildkitd.toml.example \
#     --bootstrap amd64 --platform linux/amd64
# docker buildx create --name mybuilder --append desktop-linux --platform linux/arm64
# docker buildx use mybuilder


set -eu

. $(dirname $0)/env.sh

# Check for required tools
if ! command -v zstd >/dev/null 2>&1; then
    echo "ERROR: zstd is required but not installed." >&2
    echo "Please install zstd:" >&2
    echo "  - macOS: brew install zstd" >&2
    echo "  - Debian/Ubuntu: sudo apt-get install zstd" >&2
    echo "  - RHEL/CentOS/Fedora: sudo dnf install zstd" >&2
    echo "  - Arch: sudo pacman -S zstd" >&2
    exit 1
fi

rm -rf dist/bin dist/lib dist/linux_amd64 dist/linux_arm64
rm -f dist/ollama-linux-*.tar.zst
mkdir -p dist

docker buildx build \
        --output type=local,dest=./dist/ \
        --platform=${PLATFORM} \
        ${OLLAMA_COMMON_BUILD_ARGS} \
        --target archive \
        -f Dockerfile \
        .

# buildx behavior changes for single vs. multiplatform
echo "Compressing linux tar bundles..."
if echo $PLATFORM | grep "," > /dev/null ; then
    $(dirname $0)/package_linux.sh ./dist/linux_amd64 amd64 ./dist
    $(dirname $0)/package_linux.sh ./dist/linux_arm64 arm64 ./dist
elif echo $PLATFORM | grep "arm64" > /dev/null ; then
    $(dirname $0)/package_linux.sh ./dist arm64 ./dist
elif echo $PLATFORM | grep "amd64" > /dev/null ; then
    $(dirname $0)/package_linux.sh ./dist amd64 ./dist
fi
