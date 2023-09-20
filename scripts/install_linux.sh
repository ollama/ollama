#!/bin/sh
# This script detects the current operating system architecture and installs the appropriate version of Ollama

set -eu

os=$(uname -s)
if [ "$os" != "Linux" ]; then
    echo "This script is intended to run on Linux only."
    exit 1
fi

# Determine the system architecture
ARCH=$(uname -m)

# Map architecture to the possible suffixes/names supported
case $ARCH in
    x86_64)
        ARCH_SUFFIX="amd64"
        ;;
    aarch64|arm64)
        ARCH_SUFFIX="arm64"
        ;;
    *)
        echo "Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

# Check if the user is root
IS_ROOT=0
if [ "$(id -u)" -ne 0 ]; then
    IS_ROOT=1
fi

# Conditionally show the sudo warning and use sudo for the move operation
if [ $IS_ROOT -eq 1 ]; then
    echo "Downloading the ollama executable to the PATH, this will require sudo permissions."
    sudo mkdir -p /usr/bin
    sudo curl https://ollama.ai/download/latest/ollama-linux-$ARCH > /usr/bin/ollama
    # Create a systemd service file to auto-start ollama
    echo "Creating systemd service file for ollama..."
    cat <<EOF | sudo tee /etc/systemd/system/ollama.service >/dev/null
[Unit]
Description=Ollama Service
After=network.target

[Service]
ExecStart=/usr/local/bin/ollama serve
Restart=always
RestartSec=3
Environment="HOME=$HOME"

[Install]
WantedBy=default.target
EOF
    # Reload systemd, enable and start the service
    echo "Reloading systemd and enabling ollama service..."
    sudo systemctl daemon-reload
    sudo systemctl enable ollama
    sudo systemctl restart ollama
else
    mkdir -p /usr/bin
    curl https://ollama.ai/download/latest/ollama-linux-$ARCH > /usr/bin/ollama
    echo "Creating systemd service file for ollama..."
    cat <<EOF | tee /etc/systemd/system/ollama.service >/dev/null
[Unit]
Description=Ollama Service
After=network.target

[Service]
ExecStart=/usr/local/bin/ollama serve
Restart=always
RestartSec=3
Environment="HOME=$HOME"

[Install]
WantedBy=default.target
EOF
    echo "Reloading systemd and enabling ollama service..."
    systemctl daemon-reload
    systemctl enable ollama
    systemctl start ollama
fi

echo "Installation complete. You can now run 'ollama' from the command line."
