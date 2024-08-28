#!/bin/sh

set -eu

. $(dirname $0)/env.sh

BUILD_ARCH=${BUILD_ARCH:-"amd64 arm64"}
export AMDGPU_TARGETS=${AMDGPU_TARGETS:=""}
mkdir -p dist

for TARGETARCH in ${BUILD_ARCH}; do
    docker build \
        --platform=linux/$TARGETARCH \
        ${OLLAMA_COMMON_BUILD_ARGS} \
        --target build-$TARGETARCH \
        -f Dockerfile \
        -t builder:$TARGETARCH \
        .
    docker create --platform linux/$TARGETARCH --name builder-$TARGETARCH builder:$TARGETARCH
    docker cp builder-$TARGETARCH:/go/src/github.com/ollama/ollama/dist/ollama-linux-$TARGETARCH.tgz ./dist/
    if echo ${TARGETARCH} | grep "amd64" > /dev/null; then
        docker cp builder-$TARGETARCH:/go/src/github.com/ollama/ollama/dist/ollama-linux-$TARGETARCH-rocm.tgz ./dist/
    fi
    docker rm builder-$TARGETARCH
done