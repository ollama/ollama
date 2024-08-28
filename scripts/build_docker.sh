#!/bin/sh

set -eu

. $(dirname $0)/env.sh

BUILD_ARCH=${BUILD_ARCH:-"amd64 arm64"}

# Set PUSH to a non-empty string to trigger push instead of load
PUSH=${PUSH:-""}

# In CI mode, we break things down
OLLAMA_SKIP_MANIFEST_CREATE=${OLLAMA_SKIP_MANIFEST_CREATE:-""}
OLLAMA_SKIP_IMAGE_BUILD=${OLLAMA_SKIP_IMAGE_BUILD:-""}

if [ -z "${PUSH}" ] ; then
    LOAD_OR_PUSH="--load"
else
    echo "Will be pushing ${RELEASE_IMAGE_REPO}:$VERSION for ${BUILD_ARCH}"
    LOAD_OR_PUSH="--push"
fi

if [ -z "${OLLAMA_SKIP_IMAGE_BUILD}" ]; then
    for TARGETARCH in ${BUILD_ARCH}; do
        docker build \
            ${LOAD_OR_PUSH} \
            --platform=linux/${TARGETARCH} \
            ${OLLAMA_COMMON_BUILD_ARGS} \
            -f Dockerfile \
            -t ${RELEASE_IMAGE_REPO}:$VERSION-${TARGETARCH} \
            .
    done

    if echo ${BUILD_ARCH} | grep "amd64" > /dev/null; then
        docker build \
            ${LOAD_OR_PUSH} \
            --platform=linux/amd64 \
            ${OLLAMA_COMMON_BUILD_ARGS} \
            --target runtime-rocm \
            -f Dockerfile \
            -t ${RELEASE_IMAGE_REPO}:$VERSION-rocm \
            .
    fi
fi

if [ -z "${OLLAMA_SKIP_MANIFEST_CREATE}" ]; then
    if [ -n "${PUSH}" ]; then
        docker manifest create ${FINAL_IMAGE_REPO}:$VERSION \
            ${RELEASE_IMAGE_REPO}:$VERSION-amd64 \
            ${RELEASE_IMAGE_REPO}:$VERSION-arm64
        docker manifest push ${FINAL_IMAGE_REPO}:$VERSION

        # For symmetry, tag/push the rocm image
        if [ "${RELEASE_IMAGE_REPO}" != "${FINAL_IMAGE_REPO}" ]; then
            echo "Tagging and pushing rocm image"
            docker pull ${RELEASE_IMAGE_REPO}:$VERSION-rocm
            docker tag ${RELEASE_IMAGE_REPO}:$VERSION-rocm ${FINAL_IMAGE_REPO}:$VERSION-rocm
            docker push ${FINAL_IMAGE_REPO}:$VERSION-rocm
        fi
    else
        echo "Skipping manifest generation when not pushing images are available locally as "
        echo "  ${RELEASE_IMAGE_REPO}:$VERSION-amd64"
        echo "  ${RELEASE_IMAGE_REPO}:$VERSION-arm64"
        echo "  ${RELEASE_IMAGE_REPO}:$VERSION-rocm"
    fi
fi
