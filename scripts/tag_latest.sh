#!/bin/sh
set -eu

DOCKER_ORG=${DOCKER_ORG:-"ollama"}
FINAL_IMAGE_REPO=${FINAL_IMAGE_REPO:-"${DOCKER_ORG}/ollama"}

echo "Updating ${FINAL_IMAGE_REPO}:latest -> ${FINAL_IMAGE_REPO}:${VERSION}"
docker buildx imagetools create -t ${FINAL_IMAGE_REPO}:latest ${FINAL_IMAGE_REPO}:${VERSION}
echo "Updating ${FINAL_IMAGE_REPO}:rocm -> ${FINAL_IMAGE_REPO}:${VERSION}-rocm"
docker buildx imagetools create -t ${FINAL_IMAGE_REPO}:rocm ${FINAL_IMAGE_REPO}:${VERSION}-rocm
echo "Updating ${FINAL_IMAGE_REPO}:claude -> ${FINAL_IMAGE_REPO}:${VERSION}"
docker buildx imagetools create -t ${FINAL_IMAGE_REPO}:claude ${FINAL_IMAGE_REPO}:${VERSION}
