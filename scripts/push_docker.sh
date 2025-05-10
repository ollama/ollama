#!/bin/sh

set -eu

export VERSION=${VERSION:-0.0.0}
export GOFLAGS="'-ldflags=-w -s \"-X=github.com/ollama/ollama/version.Version=$VERSION\" \"-X=github.com/ollama/ollama/server.mode=release\"'"

docker build \
    --push \
    --platform=linux/arm64,linux/amd64 \
    --build-arg=VERSION \
    --build-arg=GOFLAGS \
    -f Dockerfile \
    -t ollama/ollama -t ollama/ollama:$VERSION \
    .

# Extract major and minor version (e.g., 0.5 from 0.5.2, or 1.2 from 1.2.0)
MINOR_TAG=$(echo "$VERSION" | awk -F. '{print $1"."$2}')

# Tag and push the minor version (e.g., 0.5)
if [ -n "$MINOR_TAG" ]; then
    docker tag ollama/ollama:$VERSION ollama/ollama:$MINOR_TAG
    docker push ollama/ollama:$MINOR_TAG
fi
