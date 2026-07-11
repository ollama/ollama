#!/usr/bin/env bash

set -euo pipefail

# Simple, host-local Hy3 deployment helper.
# Keeps model artifacts in the configured OLLAMA_MODELS directory (default: /srv/hy3)
# and exposes a one-off Ollama instance for terminal verification.

MODEL_CLASS="${1:-auto}"
OLLAMA_BIN="${OLLAMA_BIN:-ollama}"
OLLAMA_MODELS_DIR="${OLLAMA_MODELS:-/srv/hy3}"
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11450}"
MODEL_REPO="hf.co/satgeze/Hy3-1M-GGUF"
PROMPT="${OLLAMA_HY3_PROMPT:-One sentence: what is a language model?}"

LOG_FILE="/tmp/ollama-hy3-serve.log"

declare -A MODEL_GB=(
    [MTP-Q6_K]=245
    [MTP-Q5_K_M]=199
    [MTP-Q4_K_M]=183
    [MTP-Q3_K_M]=145
    [MTP-Q3_XXS]=117
    [MTP-IQ2_M]=100
    [MTP-Q2_K]=111
    [IQ2_M]=96
    [Q2_K]=101
    [IQ1_M]=62
)

MODEL_CLASS_PRIORITY=(MTP-Q6_K MTP-Q5_K_M MTP-Q4_K_M MTP-Q3_K_M MTP-Q3_XXS MTP-IQ2_M MTP-Q2_K IQ2_M Q2_K IQ1_M)

usage() {
    cat <<'EOF'
Usage:
  scripts/deploy-hy3.sh [class|auto]

class: one of Q2_K, IQ2_M, IQ1_M, or any MTP-* variant supported by hf.co/satgeze/Hy3-1M-GGUF

Environment:
  OLLAMA_MODELS   Directory used for cache (default: /srv/hy3)
  OLLAMA_HOST     Host for temporary run (default: 127.0.0.1:11450)
  OLLAMA_BIN      Ollama executable (default: ollama)
  OLLAMA_HY3_PROMPT    Optional prompt to run for verification
EOF
}

if [[ "$MODEL_CLASS" == "-h" || "$MODEL_CLASS" == "--help" ]]; then
    usage
    exit 0
fi

if [[ -n "${MODEL_CLASS}" && "$MODEL_CLASS" != "auto" && -z "${MODEL_GB[$MODEL_CLASS]:-}" ]]; then
    echo "Unsupported class: $MODEL_CLASS"
    usage
    exit 1
fi

get_free_gib() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo 0
        return
    fi

    local total_mib=0
    total_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
        | awk '{total += $1} END {if (NR==0) print 0; else print int(total)}')
    if [[ -z "$total_mib" ]]; then
        echo 0
        return
    fi

    awk -v mib="$total_mib" 'BEGIN { printf "%d", mib / 1024 }'
}

pick_auto_class() {
    local free_gib
    free_gib=$(get_free_gib)
    if (( free_gib == 0 )); then
        echo "Q2_K"
        return
    fi

    # Conservative estimate includes runtime overhead and mmap/kv cache.
    local threshold
    for cls in "${MODEL_CLASS_PRIORITY[@]}"; do
        threshold=$(awk -v size="${MODEL_GB[$cls]}" 'BEGIN { printf "%d", size * 1.35 }')
        if (( free_gib >= threshold )); then
            echo "$cls"
            return
        fi
    done

    echo "Q2_K"
}

if [[ "$MODEL_CLASS" == "auto" ]]; then
    MODEL_CLASS=$(pick_auto_class)
fi

export OLLAMA_MODELS="$OLLAMA_MODELS_DIR"
export OLLAMA_HOST="http://$OLLAMA_HOST"

mkdir -p "$OLLAMA_MODELS_DIR"
MODEL_NAME="${MODEL_REPO}:${MODEL_CLASS}"

echo "Selected model: ${MODEL_NAME}"
echo "Model cache: ${OLLAMA_MODELS_DIR}"
echo "Ollama host: ${OLLAMA_HOST}"

"$OLLAMA_BIN" serve >"$LOG_FILE" 2>&1 &
PID=$!
cleanup() {
    if ps -p "$PID" >/dev/null 2>&1; then
        kill "$PID" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

for _ in {1..90}; do
    if curl -s --max-time 2 "$OLLAMA_HOST/api/version" >/dev/null; then
        break
    fi
    sleep 1
done

if [[ ! -f "$OLLAMA_MODELS_DIR/models/manifests/hf.co/satgeze/Hy3-1M-GGUF/$MODEL_CLASS" ]]; then
    echo "Model ${MODEL_CLASS} not present locally; pulling into ${OLLAMA_MODELS_DIR}..."
    "$OLLAMA_BIN" pull "$MODEL_NAME"
fi

echo "Running verify prompt: ${PROMPT}"
if ! "$OLLAMA_BIN" run "$MODEL_NAME" "$PROMPT"; then
    echo "Run failed. If this is 'unknown model architecture: hy_v3', rebuild Ollama"
    echo "with OLLAMA_LLAMA_CPP_REPOSITORY=https://github.com/satgeze/llama.cpp.git"
    echo "and OLLAMA_LLAMA_CPP_TAG=hy3-mtp before deploying again."
    exit 1
fi
echo "Done"
