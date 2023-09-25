#!/bin/sh
# This script installs Ollama on Linux.
# It detects the current operating system architecture and installs the appropriate version of Ollama.

set -eu

status() { echo ">>> $*" >&2; }
error() { echo "ERROR $*"; exit 1; }
warning() { echo "WARNING: $*"; }

TEMP_DIR=$(mktemp -d)
cleanup() { rm -rf $TEMP_DIR; }
trap cleanup EXIT

required_tools() {
    local MISSING=''
    for TOOL in $*; do
        if ! command -v $TOOL >/dev/null; then
            MISSING="$MISSING $TOOL"
        fi
    done

    echo $MISSING
}

[ "$(uname -s)" = "Linux" ] || error 'This script is intended to run on Linux only.'

case "$(uname -m)" in
    x86_64) ARCH="amd64" ;;
    aarch64|arm64) ARCH="arm64" ;;
    *) error "Unsupported architecture: $ARCH" ;;
esac

SUDO=
if [ "$(id -u)" -ne 0 ]; then
    # Running as root, no need for sudo
    if ! command -v sudo >/dev/null; then
        error "Ollama install.sh requires elevated privileges. Please re-run as root."
    fi

    SUDO="sudo"
fi

MISSING_TOOLS=$(required_tools curl awk grep sed tee xargs)
if [ -n "$MISSING_TOOLS" ]; then
    error "The following tools are required but missing: $MISSING_TOOLS"
fi

status "Downloading ollama..."
$SUDO curl -fsSL -o $TEMP_DIR/ollama "https://ollama.ai/download/ollama-linux-$ARCH"

status "Installing ollama to /usr/bin..."
$SUDO install -o0 -g0 -m755 -d /usr/bin
$SUDO install -o0 -g0 -m755 $TEMP_DIR/ollama /usr/bin/ollama

install_success() { status 'Install complete. Run "ollama" from the command line.'; }
trap install_success EXIT

# Everything from this point onwards is optional.

configure_systemd() {
    if ! id ollama >/dev/null 2>&1; then
        status "Creating ollama user..."
        $SUDO useradd -r -s /bin/false -m -d /usr/share/ollama ollama
    fi

    status "Creating ollama systemd service..."
    cat <<EOF | $SUDO tee /etc/systemd/system/ollama.service >/dev/null
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
    if [ "$(systemctl is-system-running || echo 'not running')" = 'running' ]; then 
        status "Enabling and starting ollama service..."
        $SUDO systemctl daemon-reload
        $SUDO systemctl enable ollama
        $SUDO systemctl restart ollama
    fi
}

if command -v systemctl >/dev/null; then
    configure_systemd
fi

check_gpu() {
    case $1 in
        lspci) command -v lspci >/dev/null && lspci -d '10de:' | grep -q 'NVIDIA' || return 1 ;;
        lshw) command -v lshw >/dev/null && $SUDO lshw -c display -numeric | grep -q 'vendor: .* \[10DE\]' || return 1 ;;
        nvidia-smi) command -v nvidia-smi >/dev/null || return 1 ;;
    esac
}

if ! check_gpu lspci && ! check_gpu lshw; then
    warning "No NVIDIA GPU detected. Ollama will run in CPU-only mode."
    exit 0
fi

# ref: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#rhel-7-centos-7
# ref: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#rhel-8-rocky-8
# ref: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#rhel-9-rocky-9
# ref: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#fedora
install_cuda_driver_yum() {
    status 'Installing NVIDIA repository...'
    case $PACKAGE_MANAGER in
        yum)
            $SUDO $PACKAGE_MANAGER -y install yum-utils
            $SUDO $PACKAGE_MANAGER-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/$1$2/$(uname -m)/cuda-$1$2.repo
            ;;
        dnf)
            $SUDO dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/$1$2/$(uname -m)/cuda-$1$2.repo
            ;;
    esac

    case $1 in
        rhel)
            status 'Installing EPEL repository...'
            # EPEL is required for third-party dependencies such as dkms and libvdpau
            $SUDO $PACKAGE_MANAGER -y install https://dl.fedoraproject.org/pub/epel/epel-release-latest-$2.noarch.rpm || true
            ;;
    esac

    status 'Installing CUDA driver...'
    $SUDO $PACKAGE_MANAGER -y update

    if [ "$1" = 'centos' ] || [ "$1$2" = 'rhel7' ]; then
        $SUDO $PACKAGE_MANAGER -y install nvidia-driver-latest-dkms
    fi

    $SUDO $PACKAGE_MANAGER -y install cuda-drivers
}

# ref: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu
# ref: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#debian
install_cuda_driver_apt() {
    status 'Installing NVIDIA repository...'
    curl -fsSL -o $TEMP_DIR/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/$1$2/$(uname -m)/cuda-keyring_1.1-1_all.deb

    case $1 in
        debian)
            status 'Enabling contrib sources...'
            $SUDO sed 's/main/contrib/' < /etc/apt/sources.list | sudo tee /etc/apt/sources.list.d/contrib.list > /dev/null
            ;;
    esac

    status 'Installing CUDA driver...'
    $SUDO dpkg -i $TEMP_DIR/cuda-keyring.deb
    $SUDO apt-get update
    $SUDO DEBIAN_FRONTEND=noninteractive apt-get -y install cuda-drivers -q
}

if [ ! -f "/etc/os-release" ]; then
    error "Unknown distribution. Skipping CUDA installation."
fi

. /etc/os-release

OS_NAME=$ID
OS_VERSION=$VERSION_ID

PACKAGE_MANAGER=
for PACKAGE_MANAGER in dnf yum apt-get; do
    if command -v $PACKAGE_MANAGER >/dev/null; then
        break
    fi
done

if [ -z "$PACKAGE_MANAGER" ]; then
    error "Unknown package manager. Skipping CUDA installation."
fi

if ! check_gpu nvidia-smi || [ -z "$(nvidia-smi | grep -o "CUDA Version: [0-9]*\.[0-9]*")" ]; then
    case $OS_NAME in
        centos|rhel) install_cuda_driver_yum 'rhel' $OS_VERSION ;;
        rocky) install_cuda_driver_yum 'rhel' $(echo $OS_VERSION | cut -c1) ;;
        fedora) install_cuda_driver_dnf $OS_NAME $OS_VERSION ;;
        debian|ubuntu) install_cuda_driver_apt $OS_NAME $OS_VERSION ;;
    esac
fi

if ! lsmod | grep -q nvidia; then
    KERNEL_RELEASE="$(uname -r)"
    case $OS_NAME in
        centos|rhel|rocky|fedora) $SUDO $PACKAGE_MANAGER -y install kernel-devel-$KERNEL_RELEASE kernel-headers-$KERNEL_RELEASE ;;
        debian|ubuntu) $SUDO apt-get -y install linux-headers-$KERNEL_RELEASE ;;
    esac

    $SUDO dkms status | awk -F: '/added/ { print $1 }' | xargs -n1 $SUDO dkms install
    $SUDO modprobe nvidia
fi
