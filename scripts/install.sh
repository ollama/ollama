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
    if [ "$(id -u)" -eq 0 ]; then
        # Running as root, no need for sudo
        SUDO_CMD=""
        return
    fi

    if command -v sudo >/dev/null 2>&1; then
        SUDO_CMD="sudo"
        echo "Downloading the ollama executable to the PATH, this will require sudo permissions."
    else
        echo "Error: sudo is not available. Please run as root or install sudo."
        exit 1
    fi
}

check_needed_commands() {
    if ! command -v curl >/dev/null 2>&1; then
        echo "Error: curl is not available. Please install curl."
        exit 1
    fi
    if ! command -v awk >/dev/null 2>&1; then
        echo "Error: awk is not available. Please install awk."
        exit 1
    fi
    if ! command -v grep >/dev/null 2>&1; then
        echo "Error: grep is not available. Please install grep."
        exit 1
    fi
    if ! command -v sed >/dev/null 2>&1; then
        echo "Error: sed is not available. Please install sed."
        exit 1
    fi
    if ! command -v tee >/dev/null 2>&1; then
        echo "Error: tee is not available. Please install tee."
        exit 1
    fi
    if ! command -v xargs >/dev/null 2>&1; then
        echo "Error: xargs is not available. Please install xargs."
        exit 1
    fi
    if ! command -v wget >/dev/null 2>&1; then
        echo "Error: wget is not available. Please install wget."
        exit 1
    fi
    if ! command -v yum >/dev/null 2>&1; then
        if ! command -v apt-get >/dev/null 2>&1; then
            echo "Error: neither yum nor apt-get are available. Please install one of them."
            exit 1
        fi
    fi
}

install_cuda_drivers() {
    local os_name os_version
    if [ -f "/etc/os-release" ]; then
        . /etc/os-release
        os_name=$(echo $ID | tr '[:upper:]' '[:lower:]') # Normalized to lowercase
        os_version=$VERSION_ID
    else
        echo "Unable to detect operating system. Skipping CUDA installation."
        return 1
    fi

    # based on https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation
    # os specific steps
    case $os_name in
        centos)
            $SUDO_CMD yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
            ;;
        ubuntu)
            if [$ARCH_SUFFIX = "arm64"]; then
                # this is possible, but we just haven't implemented it yet
                echo "Skipping automatic CUDA installation, not supported for arm64 Ubuntu ($os_version)."
                return
            fi
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
            ;;
        debian)
            if [$ARCH_SUFFIX = "arm64"]; then
                echo "Skipping automatic CUDA installation, not supported for arm64 Debian ($os_version)."
                return
            fi
            case $os_version in
                10)
                    wget https://developer.download.nvidia.com/compute/cuda/repos/debian10/x86_64/cuda-keyring_1.1-1_all.deb
                ;;
                11)
                    wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.1-1_all.deb
                ;;
                *)
                    echo "Skipping automatic CUDA installation, not supported for Debian ($os_version)."
                    return
                ;;
            esac
            # Backup the original sources.list file
            $SUDO_CMD cp /etc/apt/sources.list /etc/apt/sources.list.backup
            # Add 'contrib' to each line that ends with 'main' if 'contrib' is not already present
            $SUDO_CMD sed -i '/main$/!b;s/$/ contrib/' /etc/apt/sources.list
            ;;
    esac

    # shared installation steps
    case $os_name in
        centos)
            $SUDO_CMD yum install yum-utils
            $SUDO_CMD yum clean all
            $SUDO_CMD yum -y install nvidia-driver-latest-dkms
            $SUDO_CMD yum -y install cuda-driver
            if lsmod | grep -q nvidia; then
                # Need to reload the kernel module for the driver to take effect
                $SUDO_CMD yum install kernel-devel-$(uname -r) kernel-headers-$(uname -r)
                $SUDO_CMD dkms status | awk -F: '/added/ { print $1 }' | xargs -n1 $SUDO_CMD dkms install
                $SUDO_CMD modprobe nvidia
            fi
            ;;
        ubuntu|debian)
            $SUDO_CMD dpkg -i cuda-keyring_1.1-1_all.deb
            $SUDO_CMD apt-get update
            $SUDO_CMD apt-get -y install cuda-drivers
            nvidia-smi >/dev/null 2>&1
            if [ $? -ne 0 ]; then
                # Need to reload the kernel module for the driver to take effect
                $SUDO_CMD apt-get install linux-headers-$(uname -r)
                $SUDO_CMD dkms status | awk -F: '/added/ { print $1 }' | xargs -n1 $SUDO_CMD dkms install
                $SUDO_CMD modprobe nvidia
            fi
            ;;
        *)
            echo "NVIDIA CUDA drivers may not be installed, you can install them from: https://developer.nvidia.com/cuda-downloads"
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
        $SUDO_CMD useradd -r -s /bin/false -m -d /usr/share/ollama ollama 2>/dev/null 

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
Environment="HOME=/usr/share/ollama"

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
    check_needed_commands
    download_ollama
    configure_systemd
    check_install_cuda_drivers
    echo "Installation complete. You can now run 'ollama' from the command line."
}

main
