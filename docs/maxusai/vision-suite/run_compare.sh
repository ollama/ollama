#!/usr/bin/env bash
# Stock-vs-fork comparison with a budget-matched CONTROL arm.
#
# Three arms per model:
#
#   stock    upstream server, its own defaults
#   fork     fork server, fork defaults          <- the shipped behavior
#   control  fork server, vision budget pinned to upstream's effective defaults
#
# WHY THE CONTROL ARM EXISTS
#
# A naive stock-vs-fork comparison moves TWO variables at once:
#
#   1. our vision token budget (visionServerArgs adds gemma4 / nemotron_h_omni
#      branches that upstream does not have at all), and
#   2. the llama.cpp payload — LLAMA_CPP_VERSION differs whenever the fork is not
#      synced to the release the stock server is running. Measured 2026-08-07:
#      fork b10091 (v0.32.5) vs stock b10242 (v0.32.6), 151 builds apart.
#
# So "fork detects X better" and "fork's bbox IoU is worse" are both
# uninterpretable on their own. The control pins (1) to upstream's values, so any
# delta that SURVIVES is attributable to (2), and any delta that DISAPPEARS was
# ours. Run it whenever the fork and the stock server are on different
# LLAMA_CPP_VERSIONs — which is every comparison until the fork syncs.
#
# CONTROL VALUES — these are per-arch, and wrong values silently invalidate the arm:
#
#   gemma4            min 40  max 280   llama.cpp set_limit_image_tokens(40, 280)
#   nemotron_h_omni   min 256 max 256   unpatched payload is a STRUCTURAL flat 256;
#                                       002 changes it to (256, 3328), so pinning
#                                       both bounds to 256 reproduces stock
#
# The knobs are arch-gated in visionServerArgs, so on any other arch the control
# arm is a no-op and will simply duplicate the fork arm. That is a valid result,
# not a bug — but do not read it as "no budget effect" on an arch that was never
# wired up in the first place.
#
# Usage:
#   STOCK=http://127.0.0.1:11434 FORK=http://127.0.0.1:11435 \
#     MODEL=gemma4:12b-it-q4_K_M CONTROL_MIN=40 CONTROL_MAX=280 \
#     ./run_compare.sh mytag
#
# Add CONTAINER=http://127.0.0.1:11437 to include a CPU container arm. Note that
# a CPU arm's detection metrics may differ from a Metal arm on identical inputs
# and identical prompt_eval_count — greedy sampling diverges on backend floating
# point. Check prompt_eval_count before attributing any such delta to a patch.
set -u
cd "$(dirname "$0")"

PREFIX="${1:?usage: run_compare.sh <tag-prefix>}"
MODEL="${MODEL:?set MODEL, e.g. gemma4:12b-it-q4_K_M}"
STOCK="${STOCK:-}"
FORK="${FORK:-}"
CONTAINER="${CONTAINER:-}"
CONTROL_MIN="${CONTROL_MIN:-}"
CONTROL_MAX="${CONTROL_MAX:-}"

export NUM_PREDICT="${NUM_PREDICT:-4000}"
export NUM_CTX="${NUM_CTX:-16384}"
export THINK="${THINK:-false}"

arm () {  # tag host  (remaining args: env assignments)
  local tag="$1" host="$2"; shift 2
  [ -z "$host" ] && return 0
  echo ""
  echo "########## $tag  ($host)  model=$MODEL think=$THINK $* ##########"
  date +%H:%M:%S
  env "$@" python3 vision_suite.py "$host" "$tag" "$MODEL"
}

arm "${PREFIX}-stock" "$STOCK"
arm "${PREFIX}-fork"  "$FORK"

if [ -n "$FORK" ] && [ -n "$CONTROL_MAX" ]; then
  arm "${PREFIX}-control" "$FORK" \
      IMAGE_MIN_TOKENS="${CONTROL_MIN:-$CONTROL_MAX}" \
      IMAGE_MAX_TOKENS="$CONTROL_MAX"
else
  echo ""
  echo "!!! CONTROL ARM SKIPPED — set CONTROL_MAX (and CONTROL_MIN) to run it."
  echo "!!! Without it, stock-vs-fork deltas cannot be attributed to the budget"
  echo "!!! rather than to the llama.cpp payload difference."
fi

arm "${PREFIX}-container" "$CONTAINER"

echo ""
echo "COMPARE DONE ($PREFIX) $(date +%H:%M:%S)"
