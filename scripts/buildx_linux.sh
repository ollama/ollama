#!/bin/sh

set -eu

. $(dirname $0)/env.sh

# Override to a single architecture if your environment doesn't support multiarch
PLATFORM=${PLATFORM:-"linux/arm64,linux/amd64"}

mkdir -p dist

docker buildx build \
        --output type=local,dest=./dist/ \
        --platform=${PLATFORM} \
        ${OLLAMA_COMMON_BUILD_ARGS} \
        --target dist \
        -f Dockerfile \
        .

# Move the bundles to the expected location
mv -f ./dist/linux_*64/ollama* ./dist/
rmdir ./dist/linux_*64