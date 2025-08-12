#!/bin/sh

set -eu

. $(dirname $0)/env.sh

# Set PUSH to a non-empty string to trigger push instead of load
PUSH=${PUSH:-""}

if [ -z "${PUSH}" ] ; then
    echo "Building ${FINAL_IMAGE_REPO}:$VERSION locally.  set PUSH=1 to push"
    LOAD_OR_PUSH="--load"
else
    echo "Will be pushing ${FINAL_IMAGE_REPO}:$VERSION"
    LOAD_OR_PUSH="--push"
fi

if echo "$PLATFORM" | grep "amd64" > /dev/null; then
    FLAVORS="musa"
elif echo "$PLATFORM" | grep "arm64" > /dev/null; then
    FLAVORS="vulkan"
else
    echo "Error: Unsupported platform '$PLATFORM'. FLAVORS cannot be set."
    exit 1
fi

if [ "${DOCKER_ORG}" != "mthreads" ]; then
    docker buildx build \
        ${LOAD_OR_PUSH} \
        --platform=${PLATFORM} \
        ${OLLAMA_COMMON_BUILD_ARGS} \
        -f Dockerfile \
        -t ${FINAL_IMAGE_REPO}:$VERSION \
        .
    FLAVORS="rocm musa"
fi

for FLAVOR in $FLAVORS; do
    docker buildx build \
        ${LOAD_OR_PUSH} \
        --platform=${PLATFORM} \
        ${OLLAMA_COMMON_BUILD_ARGS} \
        --build-arg FLAVOR=${FLAVOR} \
        -f Dockerfile \
        -t ${FINAL_IMAGE_REPO}:$VERSION-${FLAVOR} \
        .
done
