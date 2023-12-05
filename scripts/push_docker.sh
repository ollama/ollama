#!/bin/sh

set -eu

export VERSION=${VERSION:-0.0.0}
export GOFLAGS="'-ldflags=-w -s \"-X=github.com/jmorganca/ollama/version.Version=$VERSION\" \"-X=github.com/jmorganca/ollama/server.mode=release\"'"

docker buildx build \
    --push \
    --platform=linux/arm64,linux/amd64 \
    --build-arg=VERSION \
    --build-arg=GOFLAGS \
    --cache-from type=local,src=.cache \
    -f Dockerfile \
    -t ollama/ollama -t ollama/ollama:$VERSION \
    .
