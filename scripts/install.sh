#!/bin/sh
# This script installs Ollama on Linux and macOS.
# It detects the current operating system architecture and installs the appropriate version of Ollama.

# Wrap script in main function so that a truncated partial download doesn't end
# up executing half a script.
main() {

set -eu

red="$( (/usr/bin/tput bold || :; /usr/bin/tput setaf 1 || :) 2>&-)"
plain="$( (/usr/bin/tput sgr0 || :) 2>&-)"

status() { echo ">>> $*" >&2; }
error() { echo "${red}ERROR:${plain} $*"; exit 1; }
warning() { echo "${red}WARNING:${plain} $*"; }

TEMP_DIR=$(mktemp -d)
cleanup() { rm -rf $TEMP_DIR; }
trap cleanup EXIT

available() { command -v $1 >/dev/null; }
require() {
    local MISSING=''
    for TOOL in $*; do
        if ! available $TOOL; then
            MISSING="$MISSING $TOOL"
        fi
    done

    echo $MISSING
}

OS="$(uname -s)"
ARCH=$(uname -m)
case "$ARCH" in
    x86_64) ARCH="amd64" ;;
    aarch64|arm64) ARCH="arm64" ;;
    *) error "Unsupported architecture: $ARCH" ;;
esac

if [ -n "${OLLAMA_VERSION:-}" ]; then
    DOWNLOAD_PARAM="?version=$OLLAMA_VERSION"
else
    case "${OLLAMA_CHANNEL:-stable}" in
        stable) DOWNLOAD_PARAM="" ;;
        preview) DOWNLOAD_PARAM="?channel=preview" ;;
        *) error "Unsupported release channel: $OLLAMA_CHANNEL" ;;
    esac
fi

###########################################
# macOS
###########################################

if [ "$OS" = "Darwin" ]; then
    NEEDS=$(require curl unzip)
    if [ -n "$NEEDS" ]; then
        status "ERROR: The following tools are required but missing:"
        for NEED in $NEEDS; do
            echo "  - $NEED"
        done
        exit 1
    fi

    DOWNLOAD_URL="https://ollama.com/download/Ollama-darwin.zip${DOWNLOAD_PARAM}"

    if pgrep -x Ollama >/dev/null 2>&1; then
        status "Stopping running Ollama instance..."
        pkill -x Ollama 2>/dev/null || true
        sleep 2
    fi

    if [ -d "/Applications/Ollama.app" ]; then
        status "Removing existing Ollama installation..."
        rm -rf "/Applications/Ollama.app"
    fi

    status "Downloading Ollama for macOS..."
    curl --fail --show-error --location --progress-bar \
        -o "$TEMP_DIR/Ollama-darwin.zip" "$DOWNLOAD_URL"

    status "Installing Ollama to /Applications..."
    unzip -q "$TEMP_DIR/Ollama-darwin.zip" -d "$TEMP_DIR"
    mv "$TEMP_DIR/Ollama.app" "/Applications/"

    if [ ! -L "/usr/local/bin/ollama" ] || [ "$(readlink "/usr/local/bin/ollama")" != "/Applications/Ollama.app/Contents/Resources/ollama" ]; then
        status "Adding 'ollama' command to PATH (may require password)..."
        mkdir -p "/usr/local/bin" 2>/dev/null || sudo mkdir -p "/usr/local/bin"
        ln -sf "/Applications/Ollama.app/Contents/Resources/ollama" "/usr/local/bin/ollama" 2>/dev/null || \
            sudo ln -sf "/Applications/Ollama.app/Contents/Resources/ollama" "/usr/local/bin/ollama"
    fi

    if [ -z "${OLLAMA_NO_START:-}" ]; then
        status "Starting Ollama..."
        open -a Ollama --args hidden
    fi

    status "Install complete. You can now run 'ollama'."
    exit 0
fi

###########################################
# Linux
###########################################

[ "$OS" = "Linux" ] || error 'This script is intended to run on Linux and macOS only.'

IS_WSL2=false

KERN=$(uname -r)
case "$KERN" in
    *icrosoft*WSL2 | *icrosoft*wsl2) IS_WSL2=true;;
    *icrosoft) error "Microsoft WSL1 is not currently supported. Please use WSL2 with 'wsl --set-version <distro> 2'" ;;
    *) ;;
esac

SUDO=
if [ "$(id -u)" -ne 0 ]; then
    # Running as root, no need for sudo
    if ! available sudo; then
        error "This script requires superuser permissions. Please re-run as root."
    fi

    SUDO="sudo"
fi

NEEDS=$(require curl awk grep sed tee xargs)
if [ -n "$NEEDS" ]; then
    status "ERROR: The following tools are required but missing:"
    for NEED in $NEEDS; do
        echo "  - $NEED"
    done
    exit 1
fi

# Function to download and extract with fallback from zst to tgz
download_and_extract() {
    local url_base="$1"
    local dest_dir="$2"
    local filename="$3"

    # Check if .tar.zst is available
    if curl --fail --silent --head --location "${url_base}/${filename}.tar.zst${DOWNLOAD_PARAM}" >/dev/null 2>&1; then
        # zst file exists - check if we have zstd tool
        if ! available zstd; then
            error "This version requires zstd for extraction. Please install zstd and try again:
  - Debian/Ubuntu: sudo apt-get install zstd
  - RHEL/CentOS/Fedora: sudo dnf install zstd
  - Arch: sudo pacman -S zstd"
        fi

        status "Downloading ${filename}.tar.zst"
        curl --fail --show-error --location --progress-bar \
            "${url_base}/${filename}.tar.zst${DOWNLOAD_PARAM}" | \
            zstd -d | $SUDO tar -xf - -C "${dest_dir}"
        return 0
    fi

    # Fall back to .tgz for older versions
    status "Downloading ${filename}.tgz"
    curl --fail --show-error --location --progress-bar \
        "${url_base}/${filename}.tgz${DOWNLOAD_PARAM}" | \
        $SUDO tar -xzf - -C "${dest_dir}"
}

for BINDIR in /usr/local/bin /usr/bin /bin; do
    echo $PATH | grep -q $BINDIR && break || continue
done
OLLAMA_INSTALL_DIR=$(dirname ${BINDIR})

if [ -d "$OLLAMA_INSTALL_DIR/lib/ollama" ] ; then
    status "Cleaning up old version at $OLLAMA_INSTALL_DIR/lib/ollama"
    $SUDO rm -rf "$OLLAMA_INSTALL_DIR/lib/ollama"
fi
status "Installing ollama to $OLLAMA_INSTALL_DIR"
$SUDO install -o0 -g0 -m755 -d $BINDIR
$SUDO install -o0 -g0 -m755 -d "$OLLAMA_INSTALL_DIR/lib/ollama"
download_and_extract "https://ollama.com/download" "$OLLAMA_INSTALL_DIR" "ollama-linux-${ARCH}"

if [ "$OLLAMA_INSTALL_DIR/bin/ollama" != "$BINDIR/ollama" ] ; then
    status "Making ollama accessible in the PATH in $BINDIR"
    $SUDO ln -sf "$OLLAMA_INSTALL_DIR/ollama" "$BINDIR/ollama"
fi

# Check for NVIDIA JetPack systems with additional downloads
if [ -f /etc/nv_tegra_release ] ; then
    if grep R36 /etc/nv_tegra_release > /dev/null ; then
        download_and_extract "https://ollama.com/download" "$OLLAMA_INSTALL_DIR" "ollama-linux-${ARCH}-jetpack6"
    else
        warning "Unsupported JetPack version detected.  GPU may not be supported"
    fi
fi

install_success() {
    status 'The Ollama API is now available at 127.0.0.1:11434.'
    status 'Install complete. Run "ollama" from the command line.'
}
trap install_success EXIT

# Everything from this point onwards is optional.

configure_systemd() {
    if ! id ollama >/dev/null 2>&1; then
        status "Creating ollama user..."
        $SUDO useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama
    fi
    if getent group render >/dev/null 2>&1; then
        status "Adding ollama user to render group..."
        $SUDO usermod -a -G render ollama
    fi
    if getent group video >/dev/null 2>&1; then
        status "Adding ollama user to video group..."
        $SUDO usermod -a -G video ollama
    fi

    status "Adding current user to ollama group..."
    $SUDO usermod -a -G ollama $(whoami)

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
Environment="PATH=$PATH"

[Install]
WantedBy=default.target
EOF
    SYSTEMCTL_RUNNING="$(systemctl is-system-running || true)"
    case $SYSTEMCTL_RUNNING in
        running|degraded)
            status "Enabling and starting ollama service..."
            $SUDO systemctl daemon-reload
            $SUDO systemctl enable ollama

            start_service() { $SUDO systemctl restart ollama; }
            trap start_service EXIT
            ;;
        *)
            warning "systemd is not running"
            if [ "$IS_WSL2" = true ]; then
                warning "see https://learn.microsoft.com/en-us/windows/wsl/systemd#how-to-enable-systemd to enable it"
            fi
            ;;
    esac
}

if available systemctl; then
    configure_systemd
fi

# WSL2 only supports GPUs via nvidia passthrough
# so check for nvidia-smi to determine if GPU is available
if [ "$IS_WSL2" = true ]; then
    if available nvidia-smi && [ -n "$(nvidia-smi | grep -o "CUDA Version: [0-9]*\.[0-9]*")" ]; then
        status "Nvidia GPU detected."
    fi
    install_success
    exit 0
fi

# Don't attempt to install drivers on Jetson systems
if [ -f /etc/nv_tegra_release ] ; then
    status "NVIDIA JetPack ready."
    install_success
    exit 0
fi

# Install GPU dependencies on Linux
if ! available lspci && ! available lshw; then
    warning "Unable to detect NVIDIA/AMD GPU. Install lspci or lshw to automatically detect and install GPU dependencies."
    exit 0
fi

check_gpu() {
    # Look for devices based on vendor ID for NVIDIA and AMD
    case $1 in
        lspci)
            case $2 in
                nvidia) available lspci && lspci -d '10de:' | grep -q 'NVIDIA' || return 1 ;;
                amdgpu) available lspci && lspci -d '1002:' | grep -q 'AMD' || return 1 ;;
            esac ;;
        lshw)
            case $2 in
                nvidia) available lshw && $SUDO lshw -c display -numeric -disable network | grep -q 'vendor: .* \[10DE\]' || return 1 ;;
                amdgpu) available lshw && $SUDO lshw -c display -numeric -disable network | grep -q 'vendor: .* \[1002\]' || return 1 ;;
            esac ;;
        nvidia-smi) available nvidia-smi || return 1 ;;
    esac
}

if check_gpu nvidia-smi; then
    status "NVIDIA GPU installed."
    exit 0
fi

if ! check_gpu lspci nvidia && ! check_gpu lshw nvidia && ! check_gpu lspci amdgpu && ! check_gpu lshw amdgpu; then
    install_success
    warning "No NVIDIA/AMD GPU detected. Ollama will run in CPU-only mode."
    exit 0
fi

if check_gpu lspci amdgpu || check_gpu lshw amdgpu; then
    download_and_extract "https://ollama.com/download" "$OLLAMA_INSTALL_DIR" "ollama-linux-${ARCH}-rocm"

    install_success
    status "AMD GPU ready."
    exit 0
fi

# A generic NVIDIA driver package may not support every GPU in the system.
warning "NVIDIA GPU detected. Install a driver that supports all of your NVIDIA GPUs using your distribution's driver tools. See https://docs.ollama.com/gpu#nvidia"
install_success
}

main
