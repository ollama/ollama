#!/bin/sh

set -eu

. $(dirname $0)/env.sh

docker buildx build \
    --push \
    --platform=${PLATFORM} \
    --build-arg=VERSION \
    --build-arg=GOFLAGS \
    -f Dockerfile \
    -t ${FINAL_IMAGE_REPO}:$VERSION \
    .
