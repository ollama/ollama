#!/usr/bin/env bash
set -euo pipefail

# build-mac-arm64-app.sh
# Builds the macOS arm64 Ollama backend and packages the Electron app for arm64.
# This mirrors the manual steps we validated: build backend -> placeholder amd64 lib dir -> electron-forge package.
#
# Usage:
#   ./scripts/build-mac-arm64-app.sh [version]
# Example:
#   ./scripts/build-mac-arm64-app.sh 0.0.0-dev
# If version is omitted, defaults to 0.0.0-dev.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
VERSION="${1:-0.0.0-dev}"

cd "$ROOT_DIR"

echo "[1/6] Cleaning dist directory"
rm -rf dist
mkdir -p dist/darwin

echo "[2/6] Building arm64 backend binary (version=${VERSION})"
GOOS=darwin GOARCH=arm64 CGO_ENABLED=1 \
  go build -ldflags "-w -s -X=github.com/ollama/ollama/version.Version=${VERSION} -X=github.com/ollama/ollama/server.mode=release" \
  -o dist/darwin/ollama .

if [[ ! -x dist/darwin/ollama ]]; then
  echo "Error: backend binary not created" >&2
  exit 1
fi

echo "[3/6] Creating placeholder amd64 lib directory for forge config"
mkdir -p dist/darwin-amd64/lib/ollama
: > dist/darwin-amd64/lib/ollama/KEEP

pushd macapp >/dev/null

if [[ ! -d node_modules ]]; then
  echo "[4/6] Installing macapp dependencies (npm ci)"
  npm ci --no-audit --no-fund
else
  echo "[4/6] macapp dependencies already installed (skip npm ci)"
fi

# Ensure the backend binary path matches forge extraResource expectation
if [[ ! -f ../dist/darwin/ollama ]]; then
  echo "Error: expected backend binary at ../dist/darwin/ollama" >&2
  exit 1
fi

OUT_DIR="out/Ollama-darwin-arm64"
rm -rf "$OUT_DIR"

echo "[5/6] Packaging Electron app (arm64)"
npx electron-forge package --arch=arm64

if [[ ! -d "$OUT_DIR" ]]; then
  echo "Error: expected output directory $OUT_DIR not found" >&2
  exit 1
fi

BIN_PATH="$OUT_DIR/Ollama.app/Contents/Resources/ollama"
if ! file "$BIN_PATH" | grep -q 'arm64'; then
  echo "Warning: backend binary inside app is not arm64?" >&2
  file "$BIN_PATH" || true
fi

echo "[6/6] Done. Packaged app located at: $OUT_DIR"
echo "Launch with: open $OUT_DIR/Ollama.app"

popd >/dev/null
