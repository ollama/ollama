#!/bin/sh

set -e

case "$(uname -s)" in
  DragonFly)
    ;;
  FreeBSD)
    ;;
  NetBSD)
    ;;
  OpenBSD)
    ;;
  *)
    echo "$(uname -s) is not supported"
    exit 1
    ;;
esac

export VERSION=${VERSION:-$(git describe --tags --first-parent --abbrev=7 --long --dirty --always | sed -e "s/^v//g")}
export GOFLAGS="'-ldflags=-w -s \"-X=github.com/ollama/ollama/version.Version=$VERSION\" \"-X=github.com/ollama/ollama/server.mode=release\"'"

mkdir -p dist
rm -rf llm/llama.cpp/build

go generate ./...
CGO_ENABLED=1 go build -trimpath -o dist/ollama-bsd
