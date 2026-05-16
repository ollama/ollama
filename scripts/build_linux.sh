#!/bin/sh
#
# Mac ARM users, rosetta can be flaky, so to use a remote x86 builder.
# Use the docker-container driver with the bundled buildkit GC config 
# for improved cache behavior
#
# docker context create amd64 --docker host=ssh://mybuildhost
# docker buildx create --name mybuilder \
#     --driver docker-container \
#     --config ./buildkitd.toml.example \
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

mkdir -p dist

docker buildx build \
        --output type=local,dest=./dist/ \
        --platform=${PLATFORM} \
        ${OLLAMA_COMMON_BUILD_ARGS} \
        --target archive \
        -f Dockerfile \
        .

if echo $PLATFORM | grep "amd64" > /dev/null; then
    outDir="./dist"
    if echo $PLATFORM | grep "," > /dev/null ; then
       outDir="./dist/linux_amd64"
    fi
    docker buildx build \
        --output type=local,dest=${outDir} \
        --platform=linux/amd64 \
        ${OLLAMA_COMMON_BUILD_ARGS} \
        --build-arg FLAVOR=rocm \
        --target archive \
        -f Dockerfile \
        .
fi

# Run deduplication for each platform output directory
if echo $PLATFORM | grep "," > /dev/null ; then
    $(dirname $0)/deduplicate_cuda_libs.sh "./dist/linux_amd64"
    $(dirname $0)/deduplicate_cuda_libs.sh "./dist/linux_arm64"
elif echo $PLATFORM | grep "amd64\|arm64" > /dev/null ; then
    $(dirname $0)/deduplicate_cuda_libs.sh "./dist"
fi

# buildx behavior changes for single vs. multiplatform
echo "Compressing linux tar bundles..."
if echo $PLATFORM | grep "," > /dev/null ; then
        tar c -C ./dist/linux_arm64 --exclude cuda_jetpack5 --exclude cuda_jetpack6 . | zstd -9 -T0 >./dist/ollama-linux-arm64.tar.zst
        tar c -C ./dist/linux_arm64 ./lib/ollama/cuda_jetpack5  | zstd -9 -T0 >./dist/ollama-linux-arm64-jetpack5.tar.zst
        tar c -C ./dist/linux_arm64 ./lib/ollama/cuda_jetpack6  | zstd -9 -T0 >./dist/ollama-linux-arm64-jetpack6.tar.zst
        tar c -C ./dist/linux_amd64 --exclude rocm --exclude 'mlx*' . | zstd -9 -T0 >./dist/ollama-linux-amd64.tar.zst
        tar c -C ./dist/linux_amd64 ./lib/ollama/rocm  | zstd -9 -T0 >./dist/ollama-linux-amd64-rocm.tar.zst
        ( cd ./dist/linux_amd64 && tar c lib/ollama/mlx* ) | zstd -9 -T0 >./dist/ollama-linux-amd64-mlx.tar.zst
elif echo $PLATFORM | grep "arm64" > /dev/null ; then
        tar c -C ./dist/ --exclude cuda_jetpack5 --exclude cuda_jetpack6 bin lib | zstd -9 -T0 >./dist/ollama-linux-arm64.tar.zst
        tar c -C ./dist/ ./lib/ollama/cuda_jetpack5  | zstd -9 -T0 >./dist/ollama-linux-arm64-jetpack5.tar.zst
        tar c -C ./dist/ ./lib/ollama/cuda_jetpack6  | zstd -9 -T0 >./dist/ollama-linux-arm64-jetpack6.tar.zst
elif echo $PLATFORM | grep "amd64" > /dev/null ; then
        tar c -C ./dist/ --exclude rocm --exclude 'mlx*' bin lib | zstd -9 -T0 >./dist/ollama-linux-amd64.tar.zst
        tar c -C ./dist/ ./lib/ollama/rocm  | zstd -9 -T0 >./dist/ollama-linux-amd64-rocm.tar.zst
        ( cd ./dist/ && tar c lib/ollama/mlx* ) | zstd -9 -T0 >./dist/ollama-linux-amd64-mlx.tar.zst
fi

# Warn if any compressed tarball exceeds GitHub's 2 GiB release-asset limit
LIMIT=2147483648
for f in ./dist/ollama-linux-*.tar.zst; do
    [ -f "$f" ] || continue
    size=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f")
    if [ "$size" -gt "$LIMIT" ]; then
        echo "WARNING: $f is $size bytes ($((size - LIMIT)) over the 2 GiB GitHub release-asset limit)" >&2
    fi
done
