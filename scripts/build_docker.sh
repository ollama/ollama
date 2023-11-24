#!/bin/sh

set -eu

export VERSION=${VERSION:-0.0.0}
export GOFLAGS="'-ldflags=-w -s \"-X=github.com/jmorganca/ollama/version.Version=$VERSION\" \"-X=github.com/jmorganca/ollama/server.mode=release\"'"

docker buildx build \
    --load \
    --platform=linux/arm64,linux/amd64 \
    --build-arg=VERSION \
    --build-arg=GOFLAGS \
    --cache-from type=local,src=.cache \
    --cache-to type=local,dest=.cache \
    -f Dockerfile \
    -t ollama/ollama:$VERSION \
    .
