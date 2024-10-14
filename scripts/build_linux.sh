#!/bin/sh
#
# Mac ARM users, rosetta can be flaky, so to use a remote x86 builder
#
# docker context create amd64 --docker host=ssh://mybuildhost
# docker buildx create --name mybuilder amd64 --platform linux/amd64
# docker buildx create --name mybuilder --append desktop-linux --platform linux/arm64
# docker buildx use mybuilder


set -eu

. $(dirname $0)/env.sh

mkdir -p dist

docker buildx build \
        --output type=local,dest=./dist/ \
        --platform=${PLATFORM} \
        ${OLLAMA_COMMON_BUILD_ARGS} \
        --target dist \
        -f ${DOCKERFILE_DIR}Dockerfile \
        .

# buildx behavior changes for single vs. multiplatform
if echo $PLATFORM | grep "," > /dev/null ; then 
        mv -f ./dist/linux_*64/ollama* ./dist/
        rmdir ./dist/linux_*64
fi