#!/usr/bin/env bash
# Build ollama on ARM (aarch64).
#
# GGML_CPU_ALL_VARIANTS builds armv9.2 CPU variants that require -march=...+sme.
# ARM SME support only landed in GCC 14, so the system gcc (often 13 on
# Ubuntu 24.04) is too old. We build with gcc-14/g++-14, which supports SME
# natively and ships its own libstdc++.
set -euo pipefail

CC=gcc-14
CXX=g++-14

# --- requirement checks -----------------------------------------------------
missing=0

if ! command -v "$CXX" >/dev/null 2>&1 || ! command -v "$CC" >/dev/null 2>&1; then
    echo "error: $CC/$CXX not found (GCC 14 is required for ARM SME support)." >&2
    echo "       install with: sudo apt install gcc-14 g++-14" >&2
    missing=1
fi

if ! command -v go >/dev/null 2>&1 && [ ! -x /usr/local/go/bin/go ]; then
    echo "error: go not found. Install Go or add it to PATH." >&2
    missing=1
fi

if [ "$missing" -ne 0 ]; then
    exit 1
fi

GO_EXECUTABLE="$(command -v go || echo /usr/local/go/bin/go)"

# --- build ------------------------------------------------------------------
export CC CXX

cmake --preset Default -DGO_EXECUTABLE="$GO_EXECUTABLE"
cmake --build build
