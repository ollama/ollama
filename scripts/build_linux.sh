#!/bin/sh

set -eu

export VERSION=${VERSION:-0.0.0}
export GOFLAGS="'-ldflags=-w -s \"-X=github.com/jmorganca/ollama/version.Version=$VERSION\" \"-X=github.com/jmorganca/ollama/server.mode=release\"'"

if docker buildx inspect | grep "^Driver:" | grep "docker-container" >/dev/null; then
    echo "Caching Enabled"
    CACHE_FLAGS="--cache-from type=local,src=.cache --cache-to type=local,dest=.cache"
else
    echo "Caching Disabled"
    CACHE_FLAGS=""
fi

mkdir -p dist

for TARGETARCH in amd64 arm64; do
    docker buildx build --load --platform=linux/$TARGETARCH --build-arg=VERSION --build-arg=GOFLAGS --build-arg=CGO_CFLAGS ${CACHE_FLAGS} -f Dockerfile.build -t builder:$TARGETARCH .
    docker create --platform linux/$TARGETARCH --name builder-$TARGETARCH builder:$TARGETARCH
    docker cp builder-$TARGETARCH:/go/src/github.com/jmorganca/ollama/ollama ./dist/ollama-linux-$TARGETARCH
    docker rm builder-$TARGETARCH
done
