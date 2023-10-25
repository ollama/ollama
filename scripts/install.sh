#!/bin/sh
# This script installs Ollama on Linux.
# It detects the current operating system architecture and installs the appropriate version of Ollama as a system service.
# This script adds any necessary dependencies if available.

# Force the script to exit on error.
set -eu

# Function to display status messages to standard error.
status() { echo ">>> $*" >&2; }
# Function to display error messages and then exit the script.
error() { echo "ERROR $*"; exit 1; }
# Function to display warning messages without exiting.
warning() { echo "WARNING: $*"; }

# Create a temporary directory to download the Ollama binary.
TEMP_DIR=$(mktemp -d)
# Function to clean up temporary files when the script exits.
cleanup() { rm -rf $TEMP_DIR; }
# Set up an exit trap to ensure the cleanup function runs.
trap cleanup EXIT

# Check if a given command is available, this is used to make sure all the commands needed to install Ollama are available.
available() { command -v $1 >/dev/null; }
# Check which tools from the given list are not available.
require() {
    local MISSING=''
    for TOOL in $*; do
        if ! available $TOOL; then
            MISSING="$MISSING $TOOL"
        fi
    done

    echo $MISSING
}

# Ensure the script is running on a Linux system.
[ "$(uname -s)" = "Linux" ] || error 'This script is intended to run on Linux only.'

# Detect the system architecture and set the architecture variable to download the correct version of Ollama.
ARCH=$(uname -m)
case "$ARCH" in
    x86_64) ARCH="amd64" ;;
    aarch64|arm64) ARCH="arm64" ;;
    # Exit if the architecture is unsupported.
    *) error "Unsupported architecture: $ARCH" ;;
esac

# Check if the script is being run as root. This will prompt the user to enter their password for commands that require escalated privileges if needed.
SUDO=
if [ "$(id -u)" -ne 0 ]; then
    # Running as root, no need for sudo
    if ! available sudo; then
        error "This script requires superuser permissions. Please re-run as root."
    fi

    SUDO="sudo"
fi

# Ensure all necessary tools are available.
NEEDS=$(require curl awk grep sed tee xargs)
if [ -n "$NEEDS" ]; then
    status "ERROR: The following tools are required but missing:"
    for NEED in $NEEDS; do
        echo "  - $NEED"
    done
    exit 1
fi

# Download the appropriate version of Ollama based on the detected architecture.
status "Downloading ollama..."
curl --fail --show-error --location --progress-bar -o $TEMP_DIR/ollama "https://ollama.ai/download/ollama-linux-$ARCH"

# Determine a suitable binary directory from the PATH.
for BINDIR in /usr/local/bin /usr/bin /bin; do
    echo $PATH | grep -q $BINDIR && break || continue
done

# Install the downloaded Ollama binary to the detected BINDIR so that it can be run from the command line.
status "Installing ollama to $BINDIR..."
$SUDO install -o0 -g0 -m755 -d $BINDIR
$SUDO install -o0 -g0 -m755 $TEMP_DIR/ollama $BINDIR/ollama

# Reset the trap function to display a success message when the script exits, because the steps after this point are optional.
install_success() { status 'Install complete. Run "ollama" from the command line.'; }
trap install_success EXIT

# Everything from this point onwards is optional.

# Configure systemd if available to run Ollama as a service. This will automatically start Ollama on boot and restart it if it crashes.
configure_systemd() {
    # Create a service user for Ollama to run as. This is done to ensure Ollama runs with the least privileges possible.
    if ! id ollama >/dev/null 2>&1; then
        status "Creating ollama user..."
        $SUDO useradd -r -s /bin/false -m -d /usr/share/ollama ollama
    fi

    status "Adding current user to ollama group..."
    # Add ollama to the current user's group so the service can read their files (this is needed to create new models).
    # TODO: This can be removed when the client streams the Modelfile to the server, rather than sending the filepath.
    $SUDO usermod -a -G $(id -gn ${SUDO_USER:-$USER}) ollama

    # Define and create a systemd service for Ollama.
    status "Creating ollama systemd service..."
    cat <<EOF | $SUDO tee /etc/systemd/system/ollama.service >/dev/null
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=$BINDIR/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="HOME=/usr/share/ollama"
Environment="PATH=$PATH"

[Install]
WantedBy=default.target
EOF

    # Enable and start the Ollama service if systemd is currently running.
    SYSTEMCTL_RUNNING="$(systemctl is-system-running || true)"
    case $SYSTEMCTL_RUNNING in
        running|degraded)
            status "Enabling and starting ollama service..."
            $SUDO systemctl daemon-reload
            $SUDO systemctl enable ollama

            start_service() { $SUDO systemctl restart ollama; }
            trap start_service EXIT
            ;;
    esac
}

# This section is dedicated to the automatic installation of NVIDIA CUDA drivers 
# which are essential for Ollama to make use of the system's GPU.

# Check if systemd is available on the system and if so, configure the Ollama service.
if available systemctl; then
    configure_systemd
fi

# Check if an NVIDIA GPU is available on the system
if ! available lspci && ! available lshw; then
    warning "Unable to detect NVIDIA GPU. Install lspci or lshw to automatically detect and install NVIDIA CUDA drivers."
    exit 0
fi

# Check if the NVIDIA GPU libraries are already installed.
check_gpu() {
    case $1 in
        lspci) available lspci && lspci -d '10de:' | grep -q 'NVIDIA' || return 1 ;;
        lshw) available lshw && $SUDO lshw -c display -numeric | grep -q 'vendor: .* \[10DE\]' || return 1 ;;
        nvidia-smi) available nvidia-smi || return 1 ;;
    esac
}

# Check if NVIDIA GPU libraries are already installed, if so we are done.
if check_gpu nvidia-smi; then
    status "NVIDIA GPU installed."
    exit 0
fi

# If no GPU was detected we are done.
if ! check_gpu lspci && ! check_gpu lshw; then
    warning "No NVIDIA GPU detected. Ollama will run in CPU-only mode."
    exit 0
fi

# Function to install CUDA drivers using the yum or dnf package manager.
# ref: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#rhel-7-centos-7
# ref: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#rhel-8-rocky-8
# ref: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#rhel-9-rocky-9
# ref: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#fedora
install_cuda_driver_yum() {
    status 'Installing NVIDIA repository...'
    # Depending on the package manager (either yum or dnf), install the NVIDIA repository.
    case $PACKAGE_MANAGER in
        yum)
            $SUDO $PACKAGE_MANAGER -y install yum-utils
            $SUDO $PACKAGE_MANAGER-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/$1$2/$(uname -m)/cuda-$1$2.repo
            ;;
        dnf)
            $SUDO $PACKAGE_MANAGER config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/$1$2/$(uname -m)/cuda-$1$2.repo
            ;;
    esac

    # If the OS is RHEL, install the EPEL repository. It provides extra packages that might be needed.
    case $1 in
        rhel)
            status 'Installing EPEL repository...'
            # EPEL is required for third-party dependencies such as dkms and libvdpau
            $SUDO $PACKAGE_MANAGER -y install https://dl.fedoraproject.org/pub/epel/epel-release-latest-$2.noarch.rpm || true
            ;;
    esac

    status 'Installing CUDA driver...'

    # Install the appropriate CUDA drivers depending on the distribution.
    if [ "$1" = 'centos' ] || [ "$1$2" = 'rhel7' ]; then
        $SUDO $PACKAGE_MANAGER -y install nvidia-driver-latest-dkms
    fi

    $SUDO $PACKAGE_MANAGER -y install cuda-drivers
}

# Function to install CUDA drivers using the apt package manager.
# ref: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu
# ref: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#debian
install_cuda_driver_apt() {
    status 'Installing NVIDIA repository...'
    # Download the NVIDIA keyring package for verifying packages from NVIDIA's repository.
    curl -fsSL -o $TEMP_DIR/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/$1$2/$(uname -m)/cuda-keyring_1.1-1_all.deb

    # If the OS is Debian, enable the contrib sources for extra packages.
    case $1 in
        debian)
            status 'Enabling contrib sources...'
            $SUDO sed 's/main/contrib/' < /etc/apt/sources.list | sudo tee /etc/apt/sources.list.d/contrib.list > /dev/null
            ;;
    esac

    # Install the CUDA drivers.
    status 'Installing CUDA driver...'
    $SUDO dpkg -i $TEMP_DIR/cuda-keyring.deb
    $SUDO apt-get update

    [ -n "$SUDO" ] && SUDO_E="$SUDO -E" || SUDO_E=
    DEBIAN_FRONTEND=noninteractive $SUDO_E apt-get -y install cuda-drivers -q
}

# If there's no /etc/os-release file, we can't determine the OS distribution. 
# Hence, CUDA installation is skipped.
if [ ! -f "/etc/os-release" ]; then
    error "Unknown distribution. Skipping CUDA installation."
fi

# Source the /etc/os-release file to get OS-related variables.
. /etc/os-release

# Extract the OS name and version from the sourced file.
OS_NAME=$ID
OS_VERSION=$VERSION_ID

# Detect the package manager available on the system.
PACKAGE_MANAGER=
for PACKAGE_MANAGER in dnf yum apt-get; do
    if available $PACKAGE_MANAGER; then
        break
    fi
done

# If no known package manager is found, skip CUDA installation.
if [ -z "$PACKAGE_MANAGER" ]; then
    error "Unknown package manager. Skipping CUDA installation."
fi

# Check an NVIDIA GPU is present and if the CUDA library is already installed. 
# Depending on the detected OS, use the appropriate function to install CUDA.
if ! check_gpu nvidia-smi || [ -z "$(nvidia-smi | grep -o "CUDA Version: [0-9]*\.[0-9]*")" ]; then
    case $OS_NAME in
        centos|rhel) install_cuda_driver_yum 'rhel' $OS_VERSION ;;
        rocky) install_cuda_driver_yum 'rhel' $(echo $OS_VERSION | cut -c1) ;;
        fedora) install_cuda_driver_yum $OS_NAME $OS_VERSION ;;
        amzn) install_cuda_driver_yum 'fedora' '35' ;;
        debian) install_cuda_driver_apt $OS_NAME $OS_VERSION ;;
        ubuntu) install_cuda_driver_apt $OS_NAME $(echo $OS_VERSION | sed 's/\.//') ;;
        *) exit ;;
    esac
fi

# If the NVIDIA kernel module isn't loaded, do the necessary steps to load it. This enables the GPU to be used by Ollama without restarting the system.
if ! lsmod | grep -q nvidia; then
    KERNEL_RELEASE="$(uname -r)"
    # Install the necessary kernel headers and developer files.
    case $OS_NAME in
        centos|rhel|rocky|amzn) $SUDO $PACKAGE_MANAGER -y install kernel-devel-$KERNEL_RELEASE kernel-headers-$KERNEL_RELEASE ;;
        fedora) $SUDO $PACKAGE_MANAGER -y install kernel-devel-$KERNEL_RELEASE ;;
        debian|ubuntu) $SUDO apt-get -y install linux-headers-$KERNEL_RELEASE ;;
        *) exit ;;
    esac

    # Use dkms to install the NVIDIA CUDA module for the kernel.
    NVIDIA_CUDA_VERSION=$($SUDO dkms status | awk -F: '/added/ { print $1 }')
    if [ -n "$NVIDIA_CUDA_VERSION" ]; then
        $SUDO dkms install $NVIDIA_CUDA_VERSION
    fi

    # If the nouveau (an open-source NVIDIA driver) module is loaded, recommend a reboot to complete installation.
    if lsmod | grep -q nouveau; then
        status 'Reboot to complete NVIDIA CUDA driver install.'
        exit 0
    fi

    # Load the NVIDIA module.
    $SUDO modprobe nvidia
fi

status "NVIDIA CUDA drivers installed."
