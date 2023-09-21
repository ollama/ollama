#!/bin/sh
# This script installs Ollama on Linux.
# It detects the current operating system architecture and installs the appropriate version of Ollama.

set -eu

if [ "$(uname -s)" != "Linux" ]; then
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

SUDO_CMD=""

if [ "$(id -u)" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
        SUDO_CMD="sudo"
        echo "Downloading the ollama executable to the PATH, this will require sudo permissions."
    fi
fi

# Check if CUDA drivers are available
if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VERSION=$(nvidia-smi | grep -o "CUDA Version: [0-9]*\.[0-9]*")
    if [ -z "$CUDA_VERSION" ]; then
        echo "Warning: NVIDIA-SMI is available, but the CUDA version cannot be detected. Installing CUDA drivers..."
        curl https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run | ${SUDO_CMD}sh -s -- --silent --driver
    else
        echo "Detected CUDA version $CUDA_VERSION"
    fi
else
    # Check for the presence of an NVIDIA GPU using lspci
    if lspci | grep -i "nvidia" >/dev/null 2>&1; then
        echo "Warning: NVIDIA GPU detected but NVIDIA-SMI is not available. Installing CUDA drivers..."
        curl https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run | ${SUDO_CMD}sh -s -- --silent --driver
    else
        echo "No NVIDIA GPU detected. Skipping driver installation."
    fi
fi

${SUDO_CMD} mkdir -p /usr/bin
${SUDO_CMD} curl https://ollama.ai/download/latest/ollama-linux-$ARCH > /usr/bin/ollama

# Add ollama to start-up
if command -v systemctl >/dev/null 2>&1; then
    echo "Creating systemd service file for ollama..."
    cat <<EOF | ${SUDO_CMD} tee /etc/systemd/system/ollama.service >/dev/null
[Unit]
Description=Ollama Service
After=network.target

[Service]
ExecStart=/usr/bin/ollama serve
Restart=always
RestartSec=3
Environment="HOME=$HOME"

[Install]
WantedBy=default.target
EOF
    echo "Reloading systemd and enabling ollama service..."
    ${SUDO_CMD} systemctl daemon-reload
    ${SUDO_CMD} systemctl enable ollama
    ${SUDO_CMD} systemctl restart ollama
else
    echo "Installation complete. Run 'ollama serve' from the command line to start the service. Use 'ollama run' to query a model."
    exit 0
fi

echo "Installation complete. You can now run 'ollama' from the command line."
