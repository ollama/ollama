#!/bin/bash
set -eu

# Artifactory configuration
REGISTRY=${REGISTRY:-""}
IMAGE_NAME=${IMAGE_NAME:-"ollama-cpu"}
VERSION=${VERSION:-"latest"}

# Optional Docker repository path within the registry
# If using this path, ensure to add /${DOCKER_REPO} to the image name
# example: FULL_IMAGE_NAME="${REGISTRY}/${DOCKER_REPO}/${IMAGE_NAME}:${VERSION}"
#DOCKER_REPO=${DOCKER_REPO:-""}

# Artifactory credentials (can be set via environment variables)
# Use API Key authentication for Artifactory
ARTIFACTORY_USERNAME=${ARTIFACTORY_USERNAME:-""}
ARTIFACTORY_API_KEY=${ARTIFACTORY_API_KEY:-""}

# Target platforms
PLATFORMS=${PLATFORMS:-"linux/amd64,linux/arm64"}

# Silent login if credentials are provided
if [ -n "$ARTIFACTORY_USERNAME" ] && [ -n "$ARTIFACTORY_API_KEY" ]; then
  echo "Logging in to Artifactory at $REGISTRY as $ARTIFACTORY_USERNAME..."
  echo "$ARTIFACTORY_API_KEY" | docker login -u "$ARTIFACTORY_USERNAME" --password-stdin "$REGISTRY" >/dev/null 2>&1
  echo "Login successful"
else
  echo "Artifactory credentials not provided, assuming you're already logged in"
fi

# Set up buildx if needed
BUILDER_NAME="multiarch-builder"
if ! docker buildx inspect ${BUILDER_NAME} &>/dev/null; then
    echo "Creating new buildx builder: ${BUILDER_NAME}"
    docker buildx create --name ${BUILDER_NAME} --driver docker-container --use
else
    echo "Using existing buildx builder: ${BUILDER_NAME}"
    docker buildx use ${BUILDER_NAME}
fi
docker buildx inspect --bootstrap

# Build and push the multi-arch image
FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${VERSION}"
echo "Building and pushing ${FULL_IMAGE_NAME} for platforms: ${PLATFORMS}"

docker buildx build \
    --push \
    --platform ${PLATFORMS} \
    --output=type=image,push=true,registry.insecure=true \
    --tag ${FULL_IMAGE_NAME} \
    -f Dockerfile-cpu \
    . \
    --no-cache

echo "Build and push completed successfully!"
echo "Image pushed to: ${FULL_IMAGE_NAME}"
