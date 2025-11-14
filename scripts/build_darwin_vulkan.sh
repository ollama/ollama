#!/bin/sh

set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export OLLAMA_LLAMA_BACKENDS=${OLLAMA_LLAMA_BACKENDS:-vulkan}
export OLLAMA_FETCH_MOLTENVK=ON

exec "$SCRIPT_DIR/build_darwin.sh" "$@"
