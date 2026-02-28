#!/bin/bash

set -e

echo "Uninstalling Ollama..."

# Stop and disable the systemd service
if systemctl list-units --full -all | grep -q 'ollama.service'; then
    echo "Stopping Ollama service..."
    sudo systemctl stop ollama || true
    echo "Disabling Ollama service..."
    sudo systemctl disable ollama || true
    echo "Removing Ollama service file..."
    sudo rm -f /etc/systemd/system/ollama.service
    sudo systemctl daemon-reload
fi

# Remove the binary
OLLAMA_BIN=$(command -v ollama || true)
if [ -n "$OLLAMA_BIN" ]; then
    echo "Removing Ollama binary at $OLLAMA_BIN"
    sudo rm -f "$OLLAMA_BIN"
fi

# Remove shared files
if [ -d /usr/share/ollama ]; then
    echo "Removing /usr/share/ollama..."
    sudo rm -rf /usr/share/ollama
fi

# Remove library files
if [ -d /usr/local/lib/ollama ]; then
    echo "Removing /usr/local/lib/ollama..."
    sudo rm -rf /usr/local/lib/ollama
fi

# Delete user and group if they exist
if id -u ollama >/dev/null 2>&1; then
    echo "Deleting 'ollama' user..."
    sudo userdel ollama
fi

if getent group ollama >/dev/null 2>&1; then
    echo "Deleting 'ollama' group..."
    sudo groupdel ollama
fi

echo "Ollama has been successfully uninstalled."
