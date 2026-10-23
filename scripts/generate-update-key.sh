#!/bin/bash
# One-time, on the Mac that publishes this fork. Writes an ed25519 private key
# outside the repo and refreshes the embedded public key.
#
#   bash scripts/generate-update-key.sh
#
# Then commit app/updatetrust/public_key.go. Do not commit the private key.
# Override the path with OLLAMA_UPDATE_KEY.
set -euo pipefail
cd "$(dirname "$0")/.."

KEY="${OLLAMA_UPDATE_KEY:-$HOME/.config/ollama-fork/update-private.pem}"
mkdir -p "$(dirname "$KEY")"
if [ -f "$KEY" ]; then
  echo "Update signing key already exists: $KEY"
else
  go run ./scripts/updatesign generate "$KEY"
  chmod 600 "$KEY"
fi
go run ./scripts/updatesign write-public "$KEY"
echo "Commit app/updatetrust/public_key.go. Leave $KEY on this Mac only."
