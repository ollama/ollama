#!/bin/bash
set -eu

# Set your organization and image name
ORG=${ORG:-""}
IMAGE_NAME=${IMAGE_NAME:-"ollama-cpu"}
VERSION=${VERSION:-"latest"}

# Docker Hub credentials (can be set via environment variables)
DOCKER_USERNAME=${DOCKER_USERNAME:-""}
DOCKER_PASSWORD=${DOCKER_PASSWORD:-""}

# Target platforms - same as Ollama's defaults
PLATFORMS=${PLATFORMS:-"linux/arm64,linux/amd64"}

# Silent login if credentials are provided
if [ -n "$DOCKER_USERNAME" ] && [ -n "$DOCKER_PASSWORD" ]; then
  echo "Logging in to Docker Hub as $DOCKER_USERNAME..."
  echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin >/dev/null 2>&1
  echo "Login successful"

  # If login successful, use the provided username as the org
  if [ "$ORG" = "yourorg" ]; then
    ORG=$DOCKER_USERNAME
    echo "Using Docker username '$ORG' as organization"
  fi
else
  echo "Docker credentials not provided, assuming you're already logged in"
fi

# Ensure QEMU is installed for cross-platform builds
echo "Setting up QEMU for cross-platform builds..."
docker run --privileged --rm tonistiigi/binfmt --install all

# Set up buildx if needed
BUILDER_NAME="multiarch-builder"
if ! docker buildx inspect ${BUILDER_NAME} &>/dev/null; then
    echo "Creating new buildx builder: ${BUILDER_NAME}"
    docker buildx create --name ${BUILDER_NAME} --driver docker-container --use
else
    docker buildx use ${BUILDER_NAME}
fi
docker buildx inspect --bootstrap

# Set PUSH to a non-empty string to trigger push instead of load
PUSH=${PUSH:-""}
if [ -z "${PUSH}" ] ; then
    echo "Building ${ORG}/${IMAGE_NAME}:${VERSION} locally. Set PUSH=1 to push"
    # Note: --load only works for single platform, so if building locally, adjust PLATFORMS
    if [[ "${PLATFORMS}" == *","* ]]; then
        echo "WARNING: --load only works for single platform. Setting platform to linux/$(uname -m | sed 's/x86_64/amd64/' | sed 's/aarch64/arm64/')"
        PLATFORMS="linux/$(uname -m | sed 's/x86_64/amd64/' | sed 's/aarch64/arm64/')"
    fi
    LOAD_OR_PUSH="--load"
else
    echo "Will be pushing ${ORG}/${IMAGE_NAME}:${VERSION}"
    LOAD_OR_PUSH="--push"
fi

# Build and push/load the multi-arch image
echo "Building for platforms: ${PLATFORMS}"
docker buildx build \
    --provenance=true \
    --sbom=true \
    --network=host \
    ${LOAD_OR_PUSH} \
    --platform=${PLATFORMS} \
    -f Dockerfile-cpu \
    -t ${ORG}/${IMAGE_NAME}:${VERSION} \
    .

echo "Build completed successfully!"
if [ -n "${PUSH}" ]; then
    echo "Image pushed to: ${ORG}/${IMAGE_NAME}:${VERSION}"
    echo "To pull: docker pull ${ORG}/${IMAGE_NAME}:${VERSION}"
fi