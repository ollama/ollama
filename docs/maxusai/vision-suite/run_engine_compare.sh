#!/bin/sh
# Engine-parity campaign: cold server per model, the three-suite run plus the
# fine-text probe, one tag per model. Pair with summarize_engine_compare.py,
# which renders the two comparison tables from the per-tag score files.
#
# Usage:
#   RESTART_CMD='<restart the serving process>' \
#   MODELS="gemma4:12b-nvfp4 gemma4:12b-it-q4_K_M ..." \
#     ./run_engine_compare.sh http://127.0.0.1:11499
#
# RESTART_CMD is the cold-server hook (see README "Method"); without it the
# models share one server process and cross-request leakage caveats apply.
# ENDPOINT/THINK/NUM_PREDICT/... pass through to both harnesses (defaults:
# chat endpoint, think off — the 2026-08-08 MLX-vs-GGUF campaign settings).
#
# macOS + MLX note: a fork server binary must start with the repo root as its
# working directory (the MLX dylib and llama-server payload resolve relative
# to cwd/executable), e.g.
#   RESTART_CMD='pkill -f "ollama serve"; sleep 2; (cd /path/to/repo && \
#     OLLAMA_MODELS=$HOME/.ollama/models-mlx OLLAMA_HOST=127.0.0.1:11499 \
#     ./ollama serve >> /tmp/serve.log 2>&1 &)'
set -eu
HOST="${1:?usage: run_engine_compare.sh <host>}"
DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS="${MODELS:?set MODELS to the space-separated model list}"

for m in $MODELS; do
  tag=$(printf '%s' "$m" | tr ':.' '__')
  echo "##### MODEL $m tag=$tag $(date +%T)"
  if [ -n "${RESTART_CMD:-}" ]; then
    sh -c "$RESTART_CMD"
    i=0
    until curl -sf "$HOST/api/version" >/dev/null 2>&1; do
      i=$((i + 1))
      [ "$i" -ge 60 ] && { echo "SERVER FAILED TO START for $m"; exit 1; }
      sleep 1
    done
  fi
  ENDPOINT="${ENDPOINT:-chat}" THINK="${THINK:-false}" \
    python3 "$DIR/vision_suite.py" "$HOST" "$tag" "$m"
  ENDPOINT="${ENDPOINT:-chat}" THINK="${THINK:-false}" \
    python3 "$DIR/finetext_probe.py" "$HOST" "$tag" "$m"
  echo "##### DONE $m $(date +%T)"
done
echo "ENGINE COMPARE DONE"
