#!/usr/bin/env bash
set -euo pipefail

# go to repo root
cd "$(dirname "$0")/.."

# require dev binary
if [[ ! -x ./bin/ollama ]]; then
  echo "❌ ./bin/ollama not found."
  echo "   Build with: GOFLAGS='' go build -o ./bin/ollama ./cmd/ollama"
  exit 1
fi

# dev models directory (isolated from system service)
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama-dev/models}"
HOST="${OLLAMA_HOST:-127.0.0.1:11436}"
PORT="${HOST##*:}"

# check port in advance
if command -v ss >/dev/null 2>&1; then
  if ss -ltnH "( sport = :${PORT} )" | grep -q .; then
    echo "❌ Port ${PORT} is already in use. Stop the process or run: OLLAMA_HOST=127.0.0.1:<other_port> $0"
    exit 1
  fi
fi

mkdir -p "$OLLAMA_MODELS"

# pass --host only if this build supports it
ARGS=()
if ./bin/ollama serve --help 2>&1 | grep -q -- '--host'; then
  ARGS+=(--host "$HOST")
else
  echo "ℹ️  This build does not support --host; serving on 127.0.0.1:11434."
fi

echo "▶️  Serving DEV on ${HOST} (models: $OLLAMA_MODELS)"
exec ./bin/ollama serve "${ARGS[@]}"
