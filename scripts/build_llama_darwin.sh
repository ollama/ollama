#!/bin/sh

set -e

export VERSION=${VERSION:-$(git describe --tags --first-parent --abbrev=7 --long --dirty --always | sed -e "s/^v//g")}
export GOFLAGS="'-ldflags=-w -s \"-X=github.com/ollama/ollama/version.Version=$VERSION\" \"-X=github.com/ollama/ollama/server.mode=release\"'"

mkdir -p dist

# amd64 runners
export CGO_CFLAGS_ALLOW=-mfma
export CGO_CXXFLAGS_ALLOW=-mfma
CGO_ENABLED=1 GOOS=darwin GOARCH=amd64 go build -x -ldflags="-s -w" -trimpath -o dist/ollama_llama_runner_darwin_amd64 ./runner &
CGO_ENABLED=1 GOOS=darwin GOARCH=amd64 go build -x -ldflags="-s -w" -tags avx -trimpath -o dist/ollama_llama_runner_darwin_amd64_avx ./runner &
CGO_ENABLED=1 GOOS=darwin GOARCH=amd64 go build -x -ldflags="-s -w" -tags avx,avx2 -trimpath -o dist/ollama_llama_runner_darwin_amd64_avx2 ./runner &
wait

# amd64
CGO_ENABLED=1 GOOS=darwin GOARCH=amd64 go build -x -ldflags="-s -w" -o dist/ollama_darwin_amd64 .

# arm64 runner
CGO_ENABLED=1 GOOS=darwin GOARCH=arm64 go build -ldflags="-s -w" -trimpath -o dist/ollama_llama_runner_darwin_arm64 ./runner

# arm64
CGO_ENABLED=1 GOOS=darwin GOARCH=arm64 go build -ldflags="-s -w" -trimpath -o dist/ollama_darwin_arm64 .
