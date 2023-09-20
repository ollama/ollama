#!/bin/bash

mkdir -p dist

docker build --platform=linux/amd64 -f Dockerfile.build . -t builder
docker create --platform linux/amd64 --name builder builder
docker cp builder:/go/src/github.com/jmorganca/ollama/ollama ./dist/ollama-linux-amd64
docker rm builder

docker build --platform=linux/arm64 -f Dockerfile.build . -t builder
docker create --platform linux/arm64 --name builder builder
docker cp builder:/go/src/github.com/jmorganca/ollama/ollama ./dist/ollama-linux-arm64
docker rm builder
