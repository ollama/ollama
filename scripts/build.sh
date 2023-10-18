#!/bin/sh

set -eu

usage() {
    echo "usage: $(basename $0) VERSION"
    exit 1
}

[ "$#" -eq 1 ] || usage

export VERSION="$1"

# build universal MacOS binary
sh $(dirname $0)/build_darwin.sh

# # build arm64 and amd64 Linux binaries
sh $(dirname $0)/build_linux.sh

# # build arm64 and amd64 Docker images
sh $(dirname $0)/build_docker.sh
