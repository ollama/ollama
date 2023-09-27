#!/bin/bash

set -e

mkdir -p dist

for ARCH in arm64 amd64; do
    docker buildx build --platform=linux/$ARCH -f Dockerfile.build . -t builder:$ARCH --load
    docker create --platform linux/$ARCH --name builder builder:$ARCH
    docker cp builder:/go/src/github.com/jmorganca/ollama/ollama ./dist/ollama-linux-$ARCH
    docker rm builder
done
