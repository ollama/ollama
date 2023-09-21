#!/bin/sh
# This script installs Ollama on Linux.
# It detects the current operating system architecture and installs the appropriate version of Ollama.

set -eu

check_os() {
    if [ "$(uname -s)" != "Linux" ]; then
        echo "This script is intended to run on Linux only."
        exit 1
    fi
}

determine_architecture() {
    ARCH=$(uname -m)
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
}

check_sudo() {
    if [ "$(id -u)" -ne 0 ]; then
        if command -v sudo >/dev/null 2>&1; then
            SUDO_CMD="sudo"
            echo "Downloading the ollama executable to the PATH, this will require sudo permissions."
        else
            echo "Error: sudo is not available. Please run as root or install sudo."
            exit 1
        fi
    else
        SUDO_CMD=""
    fi
}

install_cuda_drivers() {
    local os_name

    if command -v lsb_release >/dev/null 2>&1; then
        os_name=$(lsb_release -is)
    else
        # If lsb_release is not available, fall back to /etc/os-release
        if [ -f "/etc/os-release" ]; then
            . /etc/os-release
            os_name=$ID
        else
            echo "Unable to detect operating system."
            return 1
        fi
    fi

    # based on https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation
    case $os_name in
        CentOS)
            sudo yum install yum-utils
            sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
            sudo yum clean all
            sudo yum -y install nvidia-driver-latest-dkms
            sudo yum -y install cuda-driver
            sudo yum install kernel-devel-$(uname -r) kernel-headers-$(uname -r)
            sudo dkms status | awk -F: '/added/ { print $1 }' | xargs -n1 sudo dkms install
            sudo modprobe nvidia
            ;;
        RedHatEnterprise*|Kylin|Fedora|SLES|openSUSE*|Microsoft|Ubuntu|Debian)
            echo "NVIDIA CUDA drivers may not be installed, you can install them from: https://developer.nvidia.com/cuda-downloads"
            ;;
        *)
            echo "Unsupported or unknown distribution, skipping GPU CUDA driver install: $os_name"
            ;;
    esac
}

check_install_cuda_drivers() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        CUDA_VERSION=$(nvidia-smi | grep -o "CUDA Version: [0-9]*\.[0-9]*")
        if [ -z "$CUDA_VERSION" ]; then
            echo "Warning: NVIDIA-SMI is available, but the CUDA version cannot be detected. Installing CUDA drivers..."
            install_cuda_drivers
        else
            echo "Detected CUDA version $CUDA_VERSION"
        fi
    else
        if lspci | grep -i "nvidia" >/dev/null 2>&1; then
            echo "Warning: NVIDIA GPU detected but NVIDIA-SMI is not available. Installing CUDA drivers..."
            install_cuda_drivers
        else
            echo "No NVIDIA GPU detected. Skipping driver installation."
        fi
    fi
}

download_ollama() {
    ${SUDO_CMD} mkdir -p /usr/bin
    ${SUDO_CMD} curl -fsSL -o /usr/bin/ollama "https://ollama.ai/download/latest/ollama-linux-$ARCH_SUFFIX"
}

configure_systemd() {
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
}

main() {
    check_os
    determine_architecture
    check_sudo
    check_install_cuda_drivers
    download_ollama
    configure_systemd
    echo "Installation complete. You can now run 'ollama' from the command line."
}

main
