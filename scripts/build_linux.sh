#!/bin/sh
#
# Mac ARM users, rosetta can be flaky, so to use a remote x86 builder
#
# docker context create amd64 --docker host=ssh://mybuildhost
# docker buildx create --name mybuilder amd64 --platform linux/amd64
# docker buildx create --name mybuilder --append desktop-linux --platform linux/arm64
# docker buildx use mybuilder


set -eu

. $(dirname $0)/env.sh

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

# Deduplicate CUDA libraries across mlx_* and cuda_* directories
deduplicate_cuda_libs() {
    local base_dir="$1"
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
                rel_path="../../cuda_${cuda_version}/${filename}"
                rm -f "${mlx_file}"
                ln -s "${rel_path}" "${mlx_file}"
            fi
        done
    done
}

# Run deduplication for each platform output directory
if echo $PLATFORM | grep "," > /dev/null ; then
    deduplicate_cuda_libs "./dist/linux_amd64"
    deduplicate_cuda_libs "./dist/linux_arm64"
elif echo $PLATFORM | grep "amd64\|arm64" > /dev/null ; then
    deduplicate_cuda_libs "./dist"
fi

# buildx behavior changes for single vs. multiplatform
echo "Compressing linux tar bundles..."
if echo $PLATFORM | grep "," > /dev/null ; then
        tar c -C ./dist/linux_arm64 --exclude cuda_jetpack5 --exclude cuda_jetpack6 . | pigz -9vc >./dist/ollama-linux-arm64.tgz
        tar c -C ./dist/linux_arm64 ./lib/ollama/cuda_jetpack5  | pigz -9vc >./dist/ollama-linux-arm64-jetpack5.tgz
        tar c -C ./dist/linux_arm64 ./lib/ollama/cuda_jetpack6  | pigz -9vc >./dist/ollama-linux-arm64-jetpack6.tgz
        tar c -C ./dist/linux_amd64 --exclude rocm . | pigz -9vc >./dist/ollama-linux-amd64.tgz
        tar c -C ./dist/linux_amd64 ./lib/ollama/rocm  | pigz -9vc >./dist/ollama-linux-amd64-rocm.tgz
elif echo $PLATFORM | grep "arm64" > /dev/null ; then
        tar c -C ./dist/ --exclude cuda_jetpack5 --exclude cuda_jetpack6 bin lib | pigz -9vc >./dist/ollama-linux-arm64.tgz
        tar c -C ./dist/ ./lib/ollama/cuda_jetpack5  | pigz -9vc >./dist/ollama-linux-arm64-jetpack5.tgz
        tar c -C ./dist/ ./lib/ollama/cuda_jetpack6  | pigz -9vc >./dist/ollama-linux-arm64-jetpack6.tgz
elif echo $PLATFORM | grep "amd64" > /dev/null ; then
        tar c -C ./dist/ --exclude rocm bin lib | pigz -9vc >./dist/ollama-linux-amd64.tgz
        tar c -C ./dist/ ./lib/ollama/rocm  | pigz -9vc >./dist/ollama-linux-amd64-rocm.tgz
fi
