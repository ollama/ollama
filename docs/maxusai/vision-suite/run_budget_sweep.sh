#!/usr/bin/env bash
# Gemma 4 bbox-vs-budget sweep, across model sizes.
#
# Answers: "as the visual token budget rises, what happens to bounding-box
# quality?" — the question ADR 0003's validation addendum answers with a single
# number (~0.12 mean IoU cost) that turned out to be size-dependent.
#
# THE LADDER IS NOT ARBITRARY. Google's Gemma 4 model card defines the supported
# visual token budgets as exactly:
#
#     70, 140, 280, 560, 1120
#
#     https://ai.google.dev/gemma/docs/core/model_card_4
#
# 280 is llama.cpp's default ceiling; 1120 is the fork's raised default and the
# documented maximum. Values off this ladder are not documented as supported —
# note that api.DefaultImageMinTokens is 40, which is BELOW the documented floor
# of 70.
#
# WHY SIZE MATTERS: per the same model card, the 12B is encoder-free, while 26B
# A4B and 31B carry a ~550M vision encoder. These are different vision paths, so
# a bbox result measured on one size does not transfer to another.
#
# POST-004 NOTE (ADR 0008): on a payload carrying llama/compat/004, min is a
# no-op for gemma4 and the ceiling snaps to the ladder — pinning min==max no
# longer changes anything, and a budget-matched CONTROL arm can no longer
# reproduce stock behaviour (stock leaves under-budget images on natural grids;
# 004 fills). Sweep cells remain valid as ladder cells; off-ladder cells are
# unreachable by design.
#
# Each cell pins min == max == budget, which forces exactly that budget rather
# than letting the projector pick within a range — that is what isolates budget
# as the single variable. Budget flags are Runner options (SPEC B3), so every
# cell reloads the model; expect the sweep to be dominated by load time on 31B.
#
# Check prompt_eval_count moves with the budget. If it does not, the flag did not
# bind and the cell is meaningless (SPEC B4).
#
# Usage:
#   HOST=http://127.0.0.1:11435 ./run_budget_sweep.sh <tag-prefix>
#   MODELS="gemma4:12b-it-q4_K_M" BUDGETS="280 1120" ./run_budget_sweep.sh quick
set -u
cd "$(dirname "$0")"

PREFIX="${1:?usage: run_budget_sweep.sh <tag-prefix>}"
HOST="${HOST:-http://127.0.0.1:11435}"
MODELS="${MODELS:-gemma4:12b-it-q4_K_M gemma4:31b-it-q4_K_M}"
BUDGETS="${BUDGETS:-70 140 280 560 1120}"

export ONLY_TESTS="${ONLY_TESTS:-scene_single,document_single}"
export NUM_PREDICT="${NUM_PREDICT:-4000}"
export NUM_CTX="${NUM_CTX:-16384}"
export THINK="${THINK:-false}"

for model in $MODELS; do
  size="$(echo "$model" | sed 's/gemma4://; s/-.*//')"
  for b in $BUDGETS; do
    tag="${PREFIX}-${size}-b${b}"
    echo ""
    echo "########## $model  budget=$b (min=max)  -> $tag ##########"
    date +%H:%M:%S
    IMAGE_MIN_TOKENS="$b" IMAGE_MAX_TOKENS="$b" \
      HTTP_TIMEOUT="${HTTP_TIMEOUT:-1800}" \
      python3 vision_suite.py "$HOST" "$tag" "$model"
  done
done

echo ""
echo "SWEEP DONE ($PREFIX) $(date +%H:%M:%S)"
