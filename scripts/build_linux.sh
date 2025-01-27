#!/bin/sh
#
# Mac ARM users, rosetta can be flaky, so to use a remote x86 builder
#
# docker context create amd64 --docker host=ssh://mybuildhost
# docker buildx create --name mybuilder amd64 --platform linux/amd64
# docker buildx create --name mybuilder --append desktop-linux --platform linux/arm64
# docker buildx use mybuilder

#        --output type=local,dest=./dist/ \
set -eu

. $(dirname $0)/env.sh

mkdir -p dist

HTTP_PROXY=172.17.0.1:7890 HTTPS_PROXY=172.17.0.1:7890 docker --debug buildx build --load --progress=plain \
        --platform=linux/arm64 \
        ${OLLAMA_COMMON_BUILD_ARGS} \
        --target runtime-cann \
        -f Dockerfile \
        -t ollama_ascend_cann:test_v0 \
        .

# buildx behavior changes for single vs. multiplatform
if echo $PLATFORM | grep "," > /dev/null ; then 
        mv -f ./dist/linux_*64/ollama* ./dist/
        # rmdir ./dist/linux_*64
fi