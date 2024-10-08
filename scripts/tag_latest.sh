#!/bin/sh

set -eu

# For developers, you can override the DOCKER_ORG to generate multiarch manifests
#  DOCKER_ORG=jdoe VERSION=0.1.30 ./scripts/tag_latest.sh
DOCKER_ORG=${DOCKER_ORG:-"ollama"}
FINAL_IMAGE_REPO=${FINAL_IMAGE_REPO:-"${DOCKER_ORG}/ollama"}

echo "Updating ${FINAL_IMAGE_REPO}:latest -> ${FINAL_IMAGE_REPO}:${VERSION}"
docker buildx imagetools create -t ${FINAL_IMAGE_REPO}:latest ${FINAL_IMAGE_REPO}:${VERSION}
echo "Updating ${FINAL_IMAGE_REPO}:rocm -> ${FINAL_IMAGE_REPO}:${VERSION}-rocm"
docker buildx imagetools create -t ${FINAL_IMAGE_REPO}:rocm ${FINAL_IMAGE_REPO}:${VERSION}-rocm
