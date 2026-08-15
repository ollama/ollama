#!/usr/bin/env bash

set -e

echo "================================"
echo "Patch Training Runtime"
echo "================================"

# 1. llama-server -> python
echo "[1] Rename llama-server"

sed -i \
's/"llama-server"/"python"/g' \
llm/llama_server.go

# 2. lib/ollama -> lib/python3
echo "[2] Change lib/ollama -> lib/python3"

sed -i \
's/"lib", "ollama"/"lib", "python3"/g' \
ml/path.go

sed -i \
's/"lib", "ollama"/"lib", "python3"/g' \
llm/llama_binary.go

# 3. 仅同步相关注释中的路径文字
sed -i \
's|lib/ollama|lib/python3|g' \
ml/path.go

sed -i \
's|lib/ollama|lib/python3|g' \
llm/llama_binary.go

echo ""
echo "Patch complete"

echo ""
echo "Check result:"
grep -R "lib.*python3" ml/path.go llm/llama_binary.go || true
grep -R "python" llm/llama_server.go || true
