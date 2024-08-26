#!/bin/sh

set -eu

export VERSION=${VERSION:-$(git describe --tags --first-parent --abbrev=7 --long --dirty --always | sed -e "s/^v//g")}
export GOFLAGS="'-ldflags=-w -s \"-X=github.com/ollama/ollama/version.Version=$VERSION\" \"-X=github.com/ollama/ollama/server.mode=release\"'"

# For developers, you can override the DOCKER_ORG to generate multiarch manifests
#  DOCKER_ORG=jdoe ./scripts/build_docker.sh
DOCKER_ORG=${DOCKER_ORG:-"ollama"}
RELEASE_IMAGE_REPO=${RELEASE_IMAGE_REPO:-"${DOCKER_ORG}/release"}

# Set to something like "--builder remote"
BUILDER=${BUILDER:-""}

# Override to a single architecture if your environment doesn't support multiarch
PLATFORM=${PLATFORM:-"linux/arm64,linux/amd64"}

echo "Will be pushing ${RELEASE_IMAGE_REPO}:$VERSION"

docker buildx build ${BUILDER} \
    --push \
    --platform=${PLATFORM} \
    --build-arg=VERSION \
    --build-arg=GOFLAGS \
    -f Dockerfile \
    -t ${RELEASE_IMAGE_REPO}:$VERSION \
    .

docker buildx build ${BUILDER} \
    --push \
    --platform=linux/amd64 \
    --build-arg=VERSION \
    --build-arg=GOFLAGS \
    --target runtime-rocm \
    -f Dockerfile \
    -t ${RELEASE_IMAGE_REPO}:$VERSION-rocm \
    .
