#!/bin/sh

set -eu

. $(dirname $0)/env.sh

for FLAVOR in $FLAVORS; do
    MANIFEST_TAG="${FINAL_IMAGE_REPO}:${VERSION}-${FLAVOR}"
    ARCH=$(basename "${PLATFORM}")
    case "$FLAVOR" in
        musa)
            if [ "$PLATFORM" = "linux/amd64" ]; then
                MUSAVERSION="rc4.2.0"
                BASE_TAG="${MANIFEST_TAG}-${ARCH}"
                VERSIONED_TAG="${MANIFEST_TAG}-${MUSAVERSION}-${ARCH}"
                LATEST_TAG="${FINAL_IMAGE_REPO}:latest"

                echo "Pushing $VERSIONED_TAG and $LATEST_TAG"
                docker tag "$BASE_TAG" "$VERSIONED_TAG"
                docker push "$VERSIONED_TAG"
                docker tag "$BASE_TAG" "$LATEST_TAG"
                docker push "$LATEST_TAG"
            else
                echo "Error: Unsupported PLATFORM for musa flavor: $PLATFORM" >&2
                exit 1
            fi
            ;;
        vulkan)
            if [ "$PLATFORM" = "linux/arm64,linux/amd64" ]; then
                AMD64_TAG="${MANIFEST_TAG}-amd64"
                ARM64_TAG="${MANIFEST_TAG}-arm64"

                echo "Pushing $MANIFEST_TAG"
                docker manifest rm "$MANIFEST_TAG" || true
                docker manifest create "$MANIFEST_TAG" "$AMD64_TAG" "$ARM64_TAG"
                docker manifest push "$MANIFEST_TAG"
            else
                TAG="${MANIFEST_TAG}-${ARCH}"

                echo "Pushing $TAG"
                docker push "$TAG"
            fi
            ;;
        *)
            echo "Error: Unsupported FLAVOR: $FLAVOR" >&2
            exit 1
            ;;
    esac
done
