#!/usr/bin/env bash

set -euo pipefail

VERSION="10.0.0"
INDEX_URL="https://stable.repo.amd.com/rocm/whl-next/"

default_prefix() {
    local root common

    root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
    common="$(git -C "$root" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
    if [[ "${common##*/}" == ".git" ]]; then
        root="${common%/.git}"
    fi
    printf '%s/.cache/rocm\n' "$root"
}

PREFIX="${ROCM_CACHE_DIR:-}"
if [[ $# -eq 2 && "$1" == "--prefix" ]]; then
    PREFIX="$2"
elif [[ $# -ne 0 ]]; then
    echo "usage: $0 [--prefix DIR]" >&2
    exit 2
fi
if [[ -z "$PREFIX" ]]; then
    PREFIX="$(default_prefix)"
fi

PYTHON="${PYTHON:-python3}"
INSTALL="${PREFIX}/linux-${VERSION}"
VENV="${INSTALL}/.venv"
READY="${INSTALL}/.ollama-rocm-ready"

if [[ ! -f "$READY" ]]; then
    rm -rf "$INSTALL"
    "$PYTHON" -m venv "$VENV"
    "$VENV/bin/python" -m pip install --upgrade pip
    "$VENV/bin/python" -m pip install \
        --index-url "$INDEX_URL" \
        "rocm[libraries,devel,device-all]==${VERSION}"
    "$VENV/bin/rocm-sdk" init
    "$VENV/bin/rocm-sdk" path --root >"${INSTALL}/root.txt"
    touch "$READY"
fi

ROCM_ROOT="$(cat "${INSTALL}/root.txt")"

cat >"${INSTALL}/ollama-rocm-env.sh" <<EOF
export ROCM_PATH="$ROCM_ROOT"
export HIP_PLATFORM="amd"
export CMAKE_PREFIX_PATH="\$ROCM_PATH"
export PATH="\$ROCM_PATH/bin:\$ROCM_PATH/lib/llvm/bin:\$ROCM_PATH/llvm/bin:\$PATH"
EOF

echo "ROCm ${VERSION}: ${ROCM_ROOT}"
echo "Environment: source ${INSTALL}/ollama-rocm-env.sh"
