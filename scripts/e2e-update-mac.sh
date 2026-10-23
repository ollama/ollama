#!/bin/bash
# Old-to-new macOS update rehearsal. Run this on the release Mac.
# It Developer ID signs two stub Ollama.app bundles, serves the newer zip,
# and checks that the updater installs it and rejects a tampered zip.
# It does not notarize and it does not touch /Applications.
#
#   bash scripts/e2e-update-mac.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$(uname -s)" != "Darwin" ]; then
  echo "This rehearsal runs on macOS. Linux covers the signature and version tests via: bash scripts/release-fork.sh --self-test" >&2
  exit 1
fi
if [ "${OLLAMA_E2E_ALLOW_UNSIGNED:-}" = "1" ]; then
  echo "This Mac test must be Developer ID signed. Unset OLLAMA_E2E_ALLOW_UNSIGNED." >&2
  exit 1
fi

TEAM="${EXPECTED_TEAM_ID:-6LZ2DS9JPD}"
SIGN_ID="${OLLAMA_SIGN_ID:-}"
if [ -z "$SIGN_ID" ]; then
  SIGN_ID="$(security find-identity -v -p codesigning | awk -v team="$TEAM" 'index($0, team) && /Developer ID Application/ { print $2; exit }')"
fi
if [ -z "$SIGN_ID" ]; then
  echo "No Developer ID Application identity for team $TEAM." >&2
  exit 1
fi
export OLLAMA_SIGN_ID="$SIGN_ID"
exec go test -tags updater_rehearse -count=1 -timeout 20m ./app/updater/ -run 'TestMacUpdateRehearsal$'
