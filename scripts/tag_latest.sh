#!/bin/sh

set -eu

# We use 2 different image repositories to handle combining architecture images into multiarch manifest
# (The ROCm image is x86 only and is not a multiarch manifest)
# For developers, you can override the DOCKER_ORG to generate multiarch manifests
#  DOCKER_ORG=jdoe VERSION=0.1.30 PUSH=1 ./scripts/tag_latest.sh
DOCKER_ORG=${DOCKER_ORG:-"ollama"}
RELEASE_IMAGE_REPO=${RELEASE_IMAGE_REPO:-"${DOCKER_ORG}/release"}
FINAL_IMAGE_REPO=${FINAL_IMAGE_REPO:-"${DOCKER_ORG}/ollama"}

# Set PUSH to a non-empty string to trigger push instead of load
PUSH=${PUSH:-""}

echo "Assembling manifest and tagging latest"
docker manifest rm ${FINAL_IMAGE_REPO}:latest || true
docker manifest create ${FINAL_IMAGE_REPO}:latest \
    ${RELEASE_IMAGE_REPO}:$VERSION-amd64 \
    ${RELEASE_IMAGE_REPO}:$VERSION-arm64

docker pull ${RELEASE_IMAGE_REPO}:$VERSION-rocm
docker tag ${RELEASE_IMAGE_REPO}:$VERSION-rocm ${FINAL_IMAGE_REPO}:rocm

if [ -n "${PUSH}" ]; then
    echo "Pushing latest tags up..."
    docker manifest push ${FINAL_IMAGE_REPO}:latest
    docker push ${FINAL_IMAGE_REPO}:rocm
else
    echo "Not pushing ${FINAL_IMAGE_REPO}:latest and ${FINAL_IMAGE_REPO}:rocm"
fi


