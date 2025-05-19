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

# Enable mDNS support for zero-configuration cluster mode
export OLLAMA_ENABLE_MDNS=${OLLAMA_ENABLE_MDNS:-true}

# Ensure mDNS development libraries are installed
if [ "$OLLAMA_ENABLE_MDNS" = "true" ]; then
    # Check if libavahi-compat-libdns_sd-dev is installed
    if [ -f /etc/debian_version ]; then
        # Debian/Ubuntu
        if ! dpkg -l libavahi-compat-libdnssd-dev > /dev/null 2>&1; then
            echo "Warning: libavahi-compat-libdnssd-dev not found. Zero-configuration cluster mode requires this package."
            echo "Install with: sudo apt-get install libavahi-compat-libdnssd-dev"
        fi
    elif [ -f /etc/redhat-release ]; then
        # RHEL/CentOS/Fedora
        if ! rpm -q avahi-compat-libdns_sd-devel > /dev/null 2>&1; then
            echo "Warning: avahi-compat-libdns_sd-devel not found. Zero-configuration cluster mode requires this package."
            echo "Install with: sudo dnf install avahi-compat-libdns_sd-devel"
        fi
    fi
fi

mkdir -p dist

docker buildx build \
        --output type=local,dest=./dist/ \
        --platform=${PLATFORM} \
        ${OLLAMA_COMMON_BUILD_ARGS} \
        --target archive \
        -f Dockerfile \
        .

if echo $PLATFORM | grep "amd64" > /dev/null; then
    outDir="./dist"
    if echo $PLATFORM | grep "," > /dev/null ; then
       outDir="./dist/linux_amd64"
    fi
    docker buildx build \
        --output type=local,dest=${outDir} \
        --platform=linux/amd64 \
        ${OLLAMA_COMMON_BUILD_ARGS} \
        --build-arg FLAVOR=rocm \
        --target archive \
        -f Dockerfile \
        .
fi

# buildx behavior changes for single vs. multiplatform
echo "Compressing linux tar bundles..."
if echo $PLATFORM | grep "," > /dev/null ; then
        tar c -C ./dist/linux_arm64 --exclude cuda_jetpack5 --exclude cuda_jetpack6 . | pigz -9vc >./dist/ollama-linux-arm64.tgz
        tar c -C ./dist/linux_arm64 ./lib/ollama/cuda_jetpack5  | pigz -9vc >./dist/ollama-linux-arm64-jetpack5.tgz
        tar c -C ./dist/linux_arm64 ./lib/ollama/cuda_jetpack6  | pigz -9vc >./dist/ollama-linux-arm64-jetpack6.tgz
        tar c -C ./dist/linux_amd64 --exclude rocm . | pigz -9vc >./dist/ollama-linux-amd64.tgz
        tar c -C ./dist/linux_amd64 ./lib/ollama/rocm  | pigz -9vc >./dist/ollama-linux-amd64-rocm.tgz
elif echo $PLATFORM | grep "arm64" > /dev/null ; then
        tar c -C ./dist/ --exclude cuda_jetpack5 --exclude cuda_jetpack6 bin lib | pigz -9vc >./dist/ollama-linux-arm64.tgz
        tar c -C ./dist/ ./lib/ollama/cuda_jetpack5  | pigz -9vc >./dist/ollama-linux-arm64-jetpack5.tgz
        tar c -C ./dist/ ./lib/ollama/cuda_jetpack6  | pigz -9vc >./dist/ollama-linux-arm64-jetpack6.tgz
elif echo $PLATFORM | grep "amd64" > /dev/null ; then
        tar c -C ./dist/ --exclude rocm bin lib | pigz -9vc >./dist/ollama-linux-amd64.tgz
        tar c -C ./dist/ ./lib/ollama/rocm  | pigz -9vc >./dist/ollama-linux-amd64-rocm.tgz
fi
