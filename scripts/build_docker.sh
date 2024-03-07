#!/bin/sh

set -eu

export VERSION=${VERSION:-$(git describe --tags --first-parent --abbrev=7 --long --dirty --always | sed -e "s/^v//g")}
export GOFLAGS="'-ldflags=-w -s \"-X=github.com/jmorganca/ollama/version.Version=$VERSION\" \"-X=github.com/jmorganca/ollama/server.mode=release\"'"

IMAGE_NAME=${IMAGE_NAME:-"ollama/ollama"}
BUILD_PLATFORM=${BUILD_PLATFORM:-"linux/arm64,linux/amd64"}
docker build \
    --load \
    --platform=${BUILD_PLATFORM} \
    --build-arg=VERSION \
    --build-arg=GOFLAGS \
    -f Dockerfile \
    -t ${IMAGE_NAME}:$VERSION \
    .

if echo ${BUILD_PLATFORM} | grep "amd64" > /dev/null; then
    docker build \
        --load \
        --platform=linux/amd64 \
        --build-arg=VERSION \
        --build-arg=GOFLAGS \
        --target runtime-rocm \
        -f Dockerfile \
        -t ${IMAGE_NAME}:$VERSION-rocm \
        .
    docker tag ${IMAGE_NAME}:$VERSION-rocm ${IMAGE_NAME}:rocm
fi

docker tag ${IMAGE_NAME}:$VERSION ${IMAGE_NAME}:latest

echo "To release, run:"
echo "  docker push ${IMAGE_NAME}:$VERSION && docker push ${IMAGE_NAME}:latest"
if echo ${BUILD_PLATFORM} | grep "amd64" > /dev/null; then
    echo "  docker push ${IMAGE_NAME}:$VERSION-rocm && docker push ${IMAGE_NAME}:rocm"
fi
