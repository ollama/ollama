#!/usr/bin/env bash
# Model × think-mode grid runner for vision_suite.py.
#
#   RESTART_CMD="docker restart <container>" \
#   MODELS="nemotron3:33b-q4_K_M gemma4:31b-it-q4_K_M" \
#   THINK_MODES="false on" NUM_PREDICT=4000 \
#   ./run_grid.sh <host> <tag-prefix>
#
# One vision_suite.py run per (model, think) cell, cold-restarting the server
# between cells when RESTART_CMD is set (required on payloads with
# cross-request leakage — upstream #17475).
set -u
cd "$(dirname "$0")"
HOST="${1:?usage: run_grid.sh <host> <tag-prefix>}"
PREFIX="${2:?usage: run_grid.sh <host> <tag-prefix>}"
MODELS="${MODELS:-nemotron3:33b-q4_K_M gemma4:31b-it-q4_K_M qwen3.6:35b-a3b-q4_k_m}"
THINK_MODES="${THINK_MODES:-false on}"
export NUM_PREDICT="${NUM_PREDICT:-4000}"

for model in $MODELS; do
  for think in $THINK_MODES; do
    if [ -n "${RESTART_CMD:-}" ]; then
      echo ">>> $RESTART_CMD"
      $RESTART_CMD >/dev/null
      sleep 6
    fi
    slug="${PREFIX}-$(echo "$model" | tr ':/.' '---')-think${think}"
    echo "########## $model think:$think -> $slug ##########"
    date +%H:%M:%S
    THINK="$think" python3 vision_suite.py "$HOST" "$slug" "$model"
  done
done
echo "GRID DONE ($PREFIX)"
