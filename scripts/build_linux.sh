#!/bin/bash

set -e

mkdir -p dist

docker buildx build --platform=linux/amd64 -f Dockerfile.build . -t builder:amd64 --load
docker create --platform linux/amd64 --name builder builder:amd64
docker cp builder:/go/src/github.com/jmorganca/ollama/ollama ./dist/ollama-linux-amd64
docker rm builder

docker buildx build --platform=linux/arm64 -f Dockerfile.build . -t builder:arm64 --load
docker create --platform linux/arm64 --name builder builder:arm64
docker cp builder:/go/src/github.com/jmorganca/ollama/ollama ./dist/ollama-linux-arm64
docker rm builder
