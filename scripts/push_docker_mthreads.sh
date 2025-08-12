#!/bin/sh

set -eu

. "$(dirname "$0")/env.sh"

case "$PLATFORM" in
    *amd64*)
        FLAVOR="musa"
        TAG_SUFFIX="rc4.2.0-amd64"
        ;;
    *arm64*)
        FLAVOR="vulkan"
        TAG_SUFFIX="arm64"
        ;;
    *)
        echo "Unsupported PLATFORM: $PLATFORM"
        exit 1
        ;;
esac

if [ "$PLATFORM" = "linux/arm64,linux/amd64" ]; then
    docker manifest rm "${FINAL_IMAGE_REPO}:latest" || true
    docker manifest create "${FINAL_IMAGE_REPO}:latest" \
        "${FINAL_IMAGE_REPO}:${VERSION}-musa-rc4.2.0-amd64" \
        "${FINAL_IMAGE_REPO}:${VERSION}-vulkan-arm64"
    docker manifest push "${FINAL_IMAGE_REPO}:latest"
    echo "Pushed ${FINAL_IMAGE_REPO}:latest"
else
    docker tag "${FINAL_IMAGE_REPO}:${VERSION}-${FLAVOR}" \
        "${FINAL_IMAGE_REPO}:${VERSION}-${FLAVOR}-${TAG_SUFFIX}"
    docker push "${FINAL_IMAGE_REPO}:${VERSION}-${FLAVOR}-${TAG_SUFFIX}"
    echo "Pushed ${FINAL_IMAGE_REPO}:${VERSION}-${FLAVOR}-${TAG_SUFFIX}"
fi
