#!/bin/sh

set -eu

export VERSION=${VERSION:-$(git describe --tags --first-parent --abbrev=7 --long --dirty --always | sed -e "s/^v//g")}
export GOFLAGS="'-ldflags=-w -s \"-X=github.com/ollama/ollama/version.Version=$VERSION\" \"-X=github.com/ollama/ollama/server.mode=release\"'"

# Set to something like "--builder remote"
BUILDER=${BUILDER:-""}

# Override to a single architecture if your environment doesn't support multiarch
PLATFORM=${PLATFORM:-"linux/arm64,linux/amd64"}


mkdir -p dist

docker buildx build ${BUILDER} \
        --output type=local,dest=./dist/ \
        --platform=${PLATFORM} \
        --build-arg=GOFLAGS \
        --build-arg=CGO_CFLAGS \
        --build-arg=OLLAMA_CUSTOM_CPU_DEFS \
        --target dist \
        -f Dockerfile \
        .
