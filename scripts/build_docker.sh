#!/bin/sh

set -eu

export VERSION=${VERSION:-$(git describe --tags --first-parent --abbrev=7 --long --dirty --always | sed -e "s/^v//g")}
export GOFLAGS="'-ldflags=-w -s \"-X=github.com/ollama/ollama/version.Version=$VERSION\" \"-X=github.com/ollama/ollama/server.mode=release\"'"

# We use 2 different image repositories to handle combining architecture images into multiarch manifest
# (The ROCm image is x86 only and is not a multiarch manifest)
# For developers, you can override the DOCKER_ORG to generate multiarch manifests
#  DOCKER_ORG=jdoe PUSH=1 ./scripts/build_docker.sh
DOCKER_ORG=${DOCKER_ORG:-"ollama"}
RELEASE_IMAGE_REPO=${RELEASE_IMAGE_REPO:-"${DOCKER_ORG}/release"}
FINAL_IMAGE_REPO=${FINAL_IMAGE_REPO:-"${DOCKER_ORG}/ollama"}

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
            --build-arg=VERSION \
            --build-arg=GOFLAGS \
            -f Dockerfile \
            -t ${RELEASE_IMAGE_REPO}:$VERSION-${TARGETARCH} \
            .
    done

    if echo ${BUILD_ARCH} | grep "amd64" > /dev/null; then
        docker build \
            ${LOAD_OR_PUSH} \
            --platform=linux/amd64 \
            --build-arg=VERSION \
            --build-arg=GOFLAGS \
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
