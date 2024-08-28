#!/bin/sh

set -eu

. $(dirname $0)/env.sh

# Override to a single architecture if your environment doesn't support multiarch
PLATFORM=${PLATFORM:-"linux/arm64,linux/amd64"}

echo "Will be pushing ${RELEASE_IMAGE_REPO}:$VERSION"

docker buildx build \
    --push \
    --platform=${PLATFORM} \
    ${OLLAMA_COMMON_BUILD_ARGS} \
    -f Dockerfile \
    -t ${RELEASE_IMAGE_REPO}:$VERSION \
    .

docker buildx build \
    --push \
    --platform=linux/amd64 \
    ${OLLAMA_COMMON_BUILD_ARGS} \
    --target runtime-rocm \
    -f Dockerfile \
    -t ${RELEASE_IMAGE_REPO}:$VERSION-rocm \
    .
