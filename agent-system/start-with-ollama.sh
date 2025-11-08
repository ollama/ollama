#!/bin/bash

# Start Ollama and Agent System together
# This script starts Ollama, pulls required models, and then starts the Node server

echo "🚀 Starting Ollama server..."
ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!
echo "✓ Ollama started (PID: $OLLAMA_PID)"

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✓ Ollama is ready!"
        break
    fi
    sleep 1
    if [ $i -eq 30 ]; then
        echo "⚠️  Warning: Ollama may not be ready yet, but continuing..."
    fi
done

# Load model configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/models.config.sh" ]; then
    source "$SCRIPT_DIR/models.config.sh"
    echo "✓ Loaded model configuration from models.config.sh"
else
    # Default models if config file doesn't exist
    REQUIRED_MODELS=(
        "llama3.2:latest"
        "qwen2.5:1.5b"
        "deepseek-r1:1.5b"
        "gemma2:2b"
        "phi3:mini"
        "qwen2.5:0.5b"
    )
    echo "⚠️  Using default model configuration"
fi

# Pull models if not already present
echo ""
echo "📦 Checking and pulling required models..."
echo "════════════════════════════════════════"

for model in "${REQUIRED_MODELS[@]}"; do
    echo -n "Checking $model... "

    # Check if model exists
    if ollama list | grep -q "^${model%:*}"; then
        echo "✓ Already installed"
    else
        echo "⬇️  Pulling..."
        if ollama pull "$model" 2>&1 | grep -v "^pulling"; then
            echo "✓ $model installed successfully"
        else
            echo "⚠️  Warning: Failed to pull $model (continuing anyway)"
        fi
    fi
done

echo ""
echo "✓ Model check complete!"
echo "════════════════════════════════════════"
echo ""

# Start the Node server
echo "🚀 Starting Agent Terminal System..."
node server.js

# Cleanup: Kill Ollama when Node server stops
echo ""
echo "🛑 Shutting down Ollama..."
kill $OLLAMA_PID 2>/dev/null
echo "✓ Cleanup complete"
