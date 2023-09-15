#!/bin/sh
# This script detects the current operating system architecture and installs the appropriate version of Ollama

set -eu

# Check for jq, systemd, and systemctl dependencies
for cmd in "jq" "systemd" "systemctl"; do
    if ! command -v $cmd > /dev/null 2>&1; then
        echo "Error: $cmd is not installed, and this script requires it. Please install $cmd and try again."
        exit 1
    fi
done

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

# Fetch the latest release information from GitHub API
RELEASE_INFO=$(curl -s "https://api.github.com/repos/jmorganca/ollama/releases/latest")

# Extract the tag name for the latest release
TAG_NAME=$(echo "$RELEASE_INFO" | jq -r '.tag_name')

# Extract the download URL for the tarball (.tar.gz) using jq
TARBALL_URL=$(echo "$RELEASE_INFO" | jq -r --arg ARCH_SUFFIX "$ARCH_SUFFIX" '.assets[] | select(.name | endswith($ARCH_SUFFIX+".tar.gz")) | .browser_download_url')

# Download the tarball
if [ -z "$TARBALL_URL" ]; then
    echo "Failed to fetch the latest release information."
    exit 1
fi

# Create a temporary directory and clean it up when script exits
TEMP=$(mktemp -d)
cleanup() { rm -rf $TEMP; }
trap cleanup 0

echo "Downloading and unpacking from $TARBALL_URL..."
curl -sSfL "$TARBALL_URL" | tar zx -C $TEMP
echo "Download and unpack complete."

# Check if the user is root
IS_ROOT=0
if [ "$(id -u)" -ne 0 ]; then
    IS_ROOT=1
fi

# Conditionally show the sudo warning and use sudo for the move operation
if [ $IS_ROOT -eq 1 ]; then
    echo "Moving the ollama executable to the PATH, this will require sudo permissions."
    sudo mv $TEMP/ollama /usr/local/bin/
    # Create a systemd service file to auto-start ollama
    echo "Creating systemd service file for ollama..."
    cat <<EOF | sudo tee /etc/systemd/system/ollama.service
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
    sudo systemctl start ollama
else
    mv $TEMP/ollama /usr/local/bin/
    echo "Creating systemd service file for ollama..."
    cat <<EOF | tee /etc/systemd/system/ollama.service
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
