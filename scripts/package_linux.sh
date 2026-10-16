#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <payload-dir> <arch> <output-dir>" >&2
    exit 1
fi

PAYLOAD_DIR=$1
ARCH=$2
OUTPUT_DIR=$3

if [ ! -d "$PAYLOAD_DIR/bin" ] || [ ! -d "$PAYLOAD_DIR/lib/ollama" ]; then
    echo "ERROR: invalid Linux payload directory: $PAYLOAD_DIR" >&2
    exit 1
fi

case "$ARCH" in
    amd64|arm64) ;;
    *)
        echo "ERROR: unsupported Linux architecture: $ARCH" >&2
        exit 1
        ;;
esac

command -v zstd >/dev/null 2>&1 || {
    echo "ERROR: zstd is required to package Linux payloads" >&2
    exit 1
}

mkdir -p "$OUTPUT_DIR"
MANIFEST_DIR=$(mktemp -d)
cleanup() { rm -rf "$MANIFEST_DIR"; }
trap cleanup EXIT

manifest() {
    printf '%s/%s\n' "$MANIFEST_DIR" "$1"
}

add() {
    printf '%s\n' "$2" >>"$(manifest "$1")"
}

find "$PAYLOAD_DIR/bin" "$PAYLOAD_DIR/lib/ollama" -mindepth 1 -maxdepth 1 -print |
    LC_ALL=C sort |
    while IFS= read -r PATHNAME; do
        COMPONENT=${PATHNAME#"$PAYLOAD_DIR"/}
        case "$COMPONENT" in
            bin/*|lib/ollama/*.so*|lib/ollama/llama-server*|lib/ollama/llama-quantize*|lib/ollama/*_LICENSE|lib/ollama/*_LICENSE.*|lib/ollama/*_NOTICE|lib/ollama/*_NOTICE.*)
                add cpu "$COMPONENT"
                ;;
            lib/ollama/cuda_v12)
                add cuda-v12 "$COMPONENT"
                ;;
            lib/ollama/cuda_v13)
                add cuda-v13 "$COMPONENT"
                ;;
            lib/ollama/vulkan*)
                add vulkan "$COMPONENT"
                ;;
            lib/ollama/mlx*|lib/ollama/include)
                add mlx "$COMPONENT"
                ;;
            lib/ollama/cuda_jetpack5)
                add jetpack5 "$COMPONENT"
                ;;
            lib/ollama/cuda_jetpack6)
                add jetpack6 "$COMPONENT"
                ;;
            lib/ollama/rocm_v*|lib/ollama/.kpack)
                add rocm "$COMPONENT"
                ;;
            *)
                echo "ERROR: unclassified Linux payload: $COMPONENT" >&2
                exit 1
                ;;
        esac
    done

CPU_MANIFEST=$(manifest cpu)
if [ ! -s "$CPU_MANIFEST" ]; then
    echo "ERROR: Linux CPU payload is empty" >&2
    exit 1
fi

rm -f "$OUTPUT_DIR/ollama-linux-$ARCH-"*.tar.zst

PIDS=
TAR_OWNER_ARGS=
if tar --help 2>&1 | grep -q -- '--owner'; then
    TAR_OWNER_ARGS='--owner 0 --group 0'
fi
for MANIFEST in "$MANIFEST_DIR"/*; do
    [ -s "$MANIFEST" ] || continue
    NAME=$(basename "$MANIFEST")
    OUTPUT="$OUTPUT_DIR/ollama-linux-$ARCH-$NAME.tar.zst"
    echo "Creating $OUTPUT"
    (
        tar c $TAR_OWNER_ARGS -C "$PAYLOAD_DIR" -T "$MANIFEST" |
            zstd -${OLLAMA_ZSTD_LEVEL:-19} -T0 >"$OUTPUT"
    ) &
    PIDS="$PIDS $!"
done

FAILED=0
for PID in $PIDS; do
    if ! wait "$PID"; then
        FAILED=1
    fi
done
if [ "$FAILED" -ne 0 ]; then
    echo "ERROR: one or more Linux archives failed" >&2
    exit 1
fi

LIMIT=2147483648
for ARCHIVE in "$OUTPUT_DIR/ollama-linux-$ARCH-"*.tar.zst; do
    [ -f "$ARCHIVE" ] || continue
    SIZE=$(stat -f%z "$ARCHIVE" 2>/dev/null || stat -c%s "$ARCHIVE")
    if [ "$SIZE" -gt "$LIMIT" ]; then
        echo "ERROR: $ARCHIVE is $SIZE bytes ($((SIZE - LIMIT)) over the 2 GiB GitHub release-asset limit)" >&2
        exit 1
    fi
done
