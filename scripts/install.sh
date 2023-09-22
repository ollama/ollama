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
    local os_name os_version
    if [ -f "/etc/os-release" ]; then
        . /etc/os-release
        os_name=$ID
        os_version=$VERSION_ID
    else
        echo "Unable to detect operating system. Skipping CUDA installation."
        return 1
    fi

    # based on https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation
    case $os_name in
        CentOS)
            $SUDO_CMD yum install yum-utils
            $SUDO_CMD yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
            $SUDO_CMD yum clean all
            $SUDO_CMD yum -y install nvidia-driver-latest-dkms
            $SUDO_CMD yum -y install cuda-driver
            $SUDO_CMD yum install kernel-devel-$(uname -r) kernel-headers-$(uname -r)
            $SUDO_CMD dkms status | awk -F: '/added/ { print $1 }' | xargs -n1 $SUDO_CMD dkms install
            $SUDO_CMD modprobe nvidia
            ;;
        ubuntu)
            case $os_version in
                20.04)
                    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
                ;;
                22.04)
                    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
                ;;
                *)
                    echo "Skipping automatic CUDA installation, not supported for Ubuntu ($os_version)."
                    return
                ;;
            esac
            $SUDO_CMD dpkg -i cuda-keyring_1.1-1_all.deb
            $SUDO_CMD apt-get update
            $SUDO_CMD apt-get -y install cuda-drivers
            ;;
        RedHatEnterprise*|Kylin|Fedora|SLES|openSUSE*|Microsoft|Debian)
            echo "NVIDIA CUDA drivers may not be installed, you can install them from: https://developer.nvidia.com/cuda-downloads"
            ;;
        *)
            echo "Unsupported or unknown distribution, skipping GPU CUDA driver install: $os_name"
            ;;
    esac
}

check_install_cuda_drivers() {
    if lspci -d '10de:' | grep 'NVIDIA' >/dev/null; then
        # NVIDIA Corporation [10de] device is available
        if command -v nvidia-smi >/dev/null 2>&1; then
            CUDA_VERSION=$(nvidia-smi | grep -o "CUDA Version: [0-9]*\.[0-9]*")
            if [ -z "$CUDA_VERSION" ]; then
                echo "Warning: NVIDIA-SMI is available, but the CUDA version cannot be detected. Installing CUDA drivers..."
                install_cuda_drivers
            else
                echo "Detected CUDA version $CUDA_VERSION"
            fi
        else
            echo "Warning: NVIDIA GPU detected but NVIDIA-SMI is not available. Installing CUDA drivers..."
            install_cuda_drivers
        fi
    else
        echo "No NVIDIA GPU detected. Skipping driver installation."
    fi
}

download_ollama() {
    $SUDO_CMD mkdir -p /usr/bin
    $SUDO_CMD curl -fsSL -o /usr/bin/ollama "https://ollama.ai/download/latest/ollama-linux-$ARCH_SUFFIX"
}

configure_systemd() {
    if command -v systemctl >/dev/null 2>&1; then
        $SUDO_CMD useradd -r -s /bin/false -m -d /home/ollama ollama 2>/dev/null 

        echo "Creating systemd service file for ollama..."
        cat <<EOF | $SUDO_CMD tee /etc/systemd/system/ollama.service >/dev/null
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="HOME=/home/ollama"

[Install]
WantedBy=default.target
EOF
        echo "Reloading systemd and enabling ollama service..."
        if [ "$(systemctl is-system-running || echo 'not running')" = 'running' ]; then 
            $SUDO_CMD systemctl daemon-reload
            $SUDO_CMD systemctl enable ollama
            $SUDO_CMD systemctl restart ollama
        fi
    else
        echo "Run 'ollama serve' from the command line to start the service."
    fi
}

main() {
    check_os
    determine_architecture
    check_sudo
    download_ollama
    configure_systemd
    check_install_cuda_drivers
    echo "Installation complete. You can now run 'ollama' from the command line."
}

main
