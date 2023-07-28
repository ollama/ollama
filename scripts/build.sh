#!/bin/bash

mkdir -p dist
CGO_ENABLED=1 GOARCH=arm64 go build -o dist/ollama_arm64
CGO_ENABLED=1 GOARCH=amd64 go build -o dist/ollama_amd64
lipo -create -output dist/ollama dist/ollama_arm64 dist/ollama_amd64
npm run --prefix app make:sign
