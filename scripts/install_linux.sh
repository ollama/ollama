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

if [ "$(id -u)" -ne 0 ]; then
    sudo_cmd="sudo "
    echo "Downloading the ollama executable to the PATH, this will require sudo permissions."
else
    sudo_cmd=""
fi

${sudo_cmd}mkdir -p /usr/bin
${sudo_cmd}curl https://ollama.ai/download/latest/ollama-linux-$ARCH > /usr/bin/ollama

# Check if CUDA drivers are available
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Warning: NVIDIA CUDA drivers are not available on this system, install them to enable GPU support. For more information see: https://developer.nvidia.com/cuda-downloads"
fi

# Add ollama to start-up
if command -v systemctl >/dev/null 2>&1; then
    echo "Creating systemd service file for ollama..."
    cat <<EOF | ${sudo_cmd}tee /etc/systemd/system/ollama.service >/dev/null
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
    ${sudo_cmd}systemctl daemon-reload
    ${sudo_cmd}systemctl enable ollama
    ${sudo_cmd}systemctl restart ollama
elif [ -d "/etc/init.d" ]; then
    # Create an init.d script
    echo "Creating init.d script for ollama..."
    cat <<'EOF' | ${sudo_cmd}tee /etc/init.d/ollama >/dev/null
#!/bin/sh
### BEGIN INIT INFO
# Provides:          ollama
# Required-Start:    $network $local_fs $remote_fs
# Required-Stop:     $network $local_fs $remote_fs
# Default-Start:     2 3 4 5
# Default-Stop:      0 1 6
# Description:       Ollama service
### END INIT INFO

case "$1" in
  start)
    /usr/local/bin/ollama serve &
    ;;
  stop)
    killall ollama
    ;;
  restart)
    killall ollama
    /usr/local/bin/ollama serve &
    ;;
  *)
    echo "Usage: /etc/init.d/ollama {start|stop|restart}"
    exit 1
    ;;
esac

exit 0
EOF
    ${sudo_cmd}chmod +x /etc/init.d/ollama
    ${sudo_cmd}update-rc.d ollama defaults
    ${sudo_cmd}service ollama start
else
    echo "Installation complete. Run 'ollama serve' from the command line to start the service."
    exit 0
fi

echo "Installation complete. You can now run 'ollama' from the command line."
