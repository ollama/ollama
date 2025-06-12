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

# Since linux/amd64 Ascend drivers are not currently available, only linux/arm64 is built.
docker buildx build --progress=plain \
    ${LOAD_OR_PUSH} \
    --platform=linux/arm64 \
    ${OLLAMA_COMMON_BUILD_ARGS} \
    --build-arg ASCEND_VERSION="8.1.rc1-910b-openeuler22.03-py3.10" \
    --build-arg ASCEND_PRODUCT_NAME="CANN Atlas 800 A2" \
    --target cann \
    -f Dockerfile \
    -t ${FINAL_IMAGE_REPO}:$VERSION-cann-atlas-a2 \
    .

docker buildx build --progress=plain \
    ${LOAD_OR_PUSH} \
    --platform=linux/arm64 \
    ${OLLAMA_COMMON_BUILD_ARGS} \
    --build-arg ASCEND_VERSION="8.1.rc1-310p-openeuler22.03-py3.10" \
    --build-arg ASCEND_PRODUCT_NAME="CANN Atlas 300I Duo" \
    --target cann \
    -f Dockerfile \
    -t ${FINAL_IMAGE_REPO}:$VERSION-cann-300i-duo \
    .

docker buildx build \
    ${LOAD_OR_PUSH} \
    --platform=${PLATFORM} \
    ${OLLAMA_COMMON_BUILD_ARGS} \
    -f Dockerfile \
    -t ${FINAL_IMAGE_REPO}:$VERSION \
    .

if echo $PLATFORM | grep "amd64" > /dev/null; then
    docker buildx build \
        ${LOAD_OR_PUSH} \
        --platform=linux/amd64 \
        ${OLLAMA_COMMON_BUILD_ARGS} \
        --build-arg FLAVOR=rocm \
        -f Dockerfile \
        -t ${FINAL_IMAGE_REPO}:$VERSION-rocm \
        .
fi