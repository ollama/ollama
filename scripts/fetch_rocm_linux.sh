#!/usr/bin/env bash

set -euo pipefail

VERSION="7.14.0"
URL="https://repo.amd.com/rocm/tarball-multi-arch/therock-dist-linux-multiarch-${VERSION}.tar.gz"

PREFIX=""
if [[ $# -eq 2 && "$1" == "--prefix" ]]; then
    PREFIX="$2"
elif [[ $# -ne 0 ]]; then
    echo "usage: $0 [--prefix DIR]" >&2
    exit 2
fi

if [[ -z "$PREFIX" ]]; then
    PREFIX="$(git rev-parse --show-toplevel 2>/dev/null || pwd)/.cache/rocm"
fi

ARCHIVE="${PREFIX}/archives/therock-dist-linux-multiarch-${VERSION}.tar.gz"
INSTALL="${PREFIX}/linux-multiarch-${VERSION}"
ENV_FILE="${INSTALL}/ollama-rocm-env.sh"

mkdir -p "${PREFIX}/archives"
if [[ ! -f "$ARCHIVE" ]]; then
    curl --location --fail --retry 5 --output "$ARCHIVE" "$URL"
fi

if [[ ! -f "${INSTALL}/.ollama-rocm-ready" ]]; then
    TMP="${PREFIX}/.linux-multiarch-${VERSION}.tmp"
    rm -rf "$TMP"
    mkdir -p "$TMP"
    tar -xzf "$ARCHIVE" -C "$TMP"

    children=()
    while IFS= read -r child; do
        children+=("$child")
    done < <(find "$TMP" -maxdepth 1 -mindepth 1)
    if [[ ${#children[@]} -eq 1 && -d "${children[0]}" ]]; then
        find "${children[0]}" -maxdepth 1 -mindepth 1 -exec mv -t "$TMP" {} +
        rmdir "${children[0]}"
    fi

    rm -rf "$INSTALL"
    mv "$TMP" "$INSTALL"
    date -Iseconds > "${INSTALL}/.ollama-rocm-ready"
fi

LLVM_AMDGCN="${INSTALL}/lib/llvm/amdgcn"
ROCM_AMDGCN="${INSTALL}/amdgcn"
if [[ -d "$LLVM_AMDGCN" && ! -e "$ROCM_AMDGCN" ]]; then
    cp -R "$LLVM_AMDGCN" "$ROCM_AMDGCN"
fi

cat > "$ENV_FILE" <<EOF
export HIP_PATH="$INSTALL"
export HIP_DIR="\$HIP_PATH"
export ROCM_PATH="\$HIP_PATH"
export HIP_PLATFORM="amd"
export HIP_CLANG_PATH="\$HIP_PATH/lib/llvm/bin"
export HIP_DEVICE_LIB_PATH="\$HIP_PATH/amdgcn/bitcode"
if [ ! -d "\$HIP_DEVICE_LIB_PATH" ]; then
    export HIP_DEVICE_LIB_PATH="\$HIP_PATH/lib/llvm/amdgcn/bitcode"
fi
export ROCM_DEVICE_LIB_PATH="\$HIP_DEVICE_LIB_PATH"
export CC="\$HIP_CLANG_PATH/clang"
export CXX="\$HIP_CLANG_PATH/amdclang++"
export HIPCXX="\$CXX"
export CMAKE_PREFIX_PATH="\$HIP_PATH"
export PATH="\$HIP_CLANG_PATH:\$HIP_PATH/bin:\$PATH"
EOF

ln -sfn "$(basename "$INSTALL")" "${PREFIX}/linux-current"

echo "ROCm ${VERSION}: ${INSTALL}"
