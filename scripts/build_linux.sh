#!/bin/sh

set -eu

export VERSION=${VERSION:-$(git describe --tags --first-parent --abbrev=7 --long --dirty --always | sed -e "s/^v//g")}
export GOFLAGS="'-ldflags=-w -s \"-X=github.com/ollama/ollama/version.Version=$VERSION\" \"-X=github.com/ollama/ollama/server.mode=release\"'"
GZIP=$(which pigz 2>/dev/null || echo "gzip")

BUILD_ARCH=${BUILD_ARCH:-"amd64 arm64"}
export AMDGPU_TARGETS=${AMDGPU_TARGETS:=""}
mkdir -p dist

for TARGETARCH in ${BUILD_ARCH}; do
    docker build \
        --platform=linux/$TARGETARCH \
        --build-arg=GOFLAGS \
        --build-arg=CGO_CFLAGS \
        --build-arg=OLLAMA_CUSTOM_CPU_DEFS \
        --build-arg=AMDGPU_TARGETS \
        --target build-$TARGETARCH \
        -f Dockerfile \
        -t builder:$TARGETARCH \
        .
    docker create --platform linux/$TARGETARCH --name builder-$TARGETARCH builder:$TARGETARCH
    rm -rf ./dist/linux-$TARGETARCH
    docker cp builder-$TARGETARCH:/go/src/github.com/ollama/ollama/dist/linux-$TARGETARCH ./dist
    if echo ${TARGETARCH} | grep "amd64" > /dev/null; then
        docker cp builder-$TARGETARCH:/go/src/github.com/ollama/ollama/dist/linux-$TARGETARCH-rocm ./dist
    fi
    docker rm builder-$TARGETARCH
    echo "Compressing final linux bundle..."
    rm -f ./dist/ollama-linux-$TARGETARCH.tgz
    (cd dist/linux-$TARGETARCH && tar cf - . | ${GZIP} --best > ../ollama-linux-$TARGETARCH.tgz )
    if [ -d dist/linux-$TARGETARCH-rocm ]; then
        (cd dist/linux-$TARGETARCH-rocm && tar cf - . | ${GZIP} --best > ../ollama-linux-$TARGETARCH-rocm.tgz )
    fi
done
