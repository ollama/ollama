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

docker buildx build --progress=plain \
        --output type=local,dest=./dist/ \
        --platform=linux/arm64 \
        ${OLLAMA_COMMON_BUILD_ARGS} \
        --build-arg ASCEND_VERSION="8.1.rc1-910b-openeuler22.03-py3.10" \
        --build-arg ASCEND_PRODUCT_NAME="CANN Atlas 800 A2" \
        --target archive-cann-atlas-a2 \
        -f Dockerfile \
        .

docker buildx build --progress=plain \
        --output type=local,dest=./dist/ \
        --platform=linux/arm64 \
        ${OLLAMA_COMMON_BUILD_ARGS} \
        --build-arg ASCEND_VERSION="8.1.rc1-310p-openeuler22.03-py3.10" \
        --build-arg ASCEND_PRODUCT_NAME="CANN Atlas 300I Duo" \
        --target archive-cann-300i-duo \
        -f Dockerfile \
        .

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

# buildx behavior changes for single vs. multiplatform
echo "Compressing linux tar bundles..."
if echo $PLATFORM | grep "," > /dev/null ; then
        tar c -C ./dist/linux_arm64 --exclude cuda_jetpack5 --exclude cuda_jetpack6 . | pigz -9vc >./dist/ollama-linux-arm64.tgz
        tar c -C ./dist/linux_arm64 ./lib/ollama/cuda_jetpack5  | pigz -9vc >./dist/ollama-linux-arm64-jetpack5.tgz
        tar c -C ./dist/linux_arm64 ./lib/ollama/cuda_jetpack6  | pigz -9vc >./dist/ollama-linux-arm64-jetpack6.tgz
        tar c -C ./dist ./lib/ollama/cann_atlas_a2 | pigz -9vc >./dist/ollama-linux-arm64-cann-atlas-a2.tgz
        tar c -C ./dist ./lib/ollama/cann_300i_duo | pigz -9vc >./dist/ollama-linux-arm64-cann-300i-duo.tgz
        tar c -C ./dist/linux_amd64 --exclude rocm . | pigz -9vc >./dist/ollama-linux-amd64.tgz
        tar c -C ./dist/linux_amd64 ./lib/ollama/rocm  | pigz -9vc >./dist/ollama-linux-amd64-rocm.tgz
elif echo $PLATFORM | grep "arm64" > /dev/null ; then
        tar c -C ./dist/ --exclude cuda_jetpack5 --exclude cuda_jetpack6 bin lib | pigz -9vc >./dist/ollama-linux-arm64.tgz
        tar c -C ./dist/ ./lib/ollama/cuda_jetpack5  | pigz -9vc >./dist/ollama-linux-arm64-jetpack5.tgz
        tar c -C ./dist/ ./lib/ollama/cuda_jetpack6  | pigz -9vc >./dist/ollama-linux-arm64-jetpack6.tgz
        tar c -C ./dist ./lib/ollama/cann_atlas_a2 | pigz -9vc >./dist/ollama-linux-arm64-cann-atlas-a2.tgz
        tar c -C ./dist ./lib/ollama/cann_300i_duo | pigz -9vc >./dist/ollama-linux-arm64-cann-300i-duo.tgz
elif echo $PLATFORM | grep "amd64" > /dev/null ; then
        tar c -C ./dist/ --exclude rocm bin lib | pigz -9vc >./dist/ollama-linux-amd64.tgz
        tar c -C ./dist/ ./lib/ollama/rocm  | pigz -9vc >./dist/ollama-linux-amd64-rocm.tgz
fi
