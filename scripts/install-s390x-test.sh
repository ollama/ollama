#!/bin/sh
# install.sh - Ollama installer with s390x support (build from source)

set -eu

red="$( (/usr/bin/tput bold || :; /usr/bin/tput setaf 1 || :) 2>&-)"
plain="$( (/usr/bin/tput sgr0 || :) 2>&-)"
status() { echo ">>> $*" >&2; }
error() { echo "${red}ERROR:${plain} $*"; exit 1; }
warning() { echo "${red}WARNING:${plain} $*"; }

TEMP_DIR=$(mktemp -d)
cleanup() { rm -rf "$TEMP_DIR"; }
trap cleanup EXIT

available() { command -v "$1" >/dev/null 2>&1; }

require() {
    local MISSING=''
    for TOOL in "$@"; do
        if ! available "$TOOL"; then
            MISSING="$MISSING $TOOL"
        fi
    done
    echo "$MISSING"
}

OS="$(uname -s)"
ARCH=$(uname -m)
case "$ARCH" in
    x86_64) ARCH="amd64" ;;
    aarch64|arm64) ARCH="arm64" ;;
    s390x)
        ARCH="s390x"
        status "Detected IBM Z (s390x) architecture"
        ;;
    *) error "Unsupported architecture: $ARCH" ;;
esac

VER_PARAM="${OLLAMA_VERSION:+?version=$OLLAMA_VERSION}"

###########################################
# macOS
###########################################
if [ "$OS" = "Darwin" ]; then
    NEEDS=$(require curl unzip)
    if [ -n "$NEEDS" ]; then
        status "ERROR: The following tools are required but missing:"
        for NEED in $NEEDS; do echo " - $NEED"; done
        exit 1
    fi

    DOWNLOAD_URL="https://ollama.com/download/Ollama-darwin.zip${VER_PARAM}"

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
        status "Adding 'ollama' command to PATH..."
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
    *icrosoft*WSL2 | *icrosoft*wsl2) IS_WSL2=true ;;
    *icrosoft) error "Microsoft WSL1 is not currently supported." ;;
    *) ;;
esac

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    if ! available sudo; then
        error "This script requires superuser permissions."
    fi
    SUDO="sudo"
fi

NEEDS=$(require curl awk grep sed tee xargs)
if [ -n "$NEEDS" ]; then
    status "The following tools are required but missing: $NEEDS"
    
    # Try to install curl if it's missing
    if echo "$NEEDS" | grep -q curl; then
        status "Attempting to install curl..."
        
        if available dnf; then
            dnf install -y curl
        elif available apt-get; then
            apt-get update -qq
            apt-get install -y curl
        elif available yum; then
            yum install -y curl
        elif available zypper; then
            zypper install -y curl
        elif available pacman; then
            pacman -S --noconfirm curl
        else
            error "Could not install curl automatically. Please install it manually."
        fi
        
        # Verify curl was installed
        if ! available curl; then
            error "Failed to install curl. Please install it manually."
        fi
        
        status "curl installed successfully"
        
        # Re-check for remaining missing tools
        NEEDS=$(require awk grep sed tee xargs)
    fi
    
    # If there are still missing tools, report and exit
    if [ -n "$NEEDS" ]; then
        status "ERROR: The following tools are still required but missing:"
        for NEED in $NEEDS; do echo " - $NEED"; done
        exit 1
    fi
fi

# ============================================
# s390x: Build from Source
# ============================================
if [ "$ARCH" = "s390x" ]; then
    status "Building Ollama from source for s390x..."

    if available apt-get; then
        $SUDO apt-get update -qq
        $SUDO apt-get install -y -qq build-essential cmake git wget ca-certificates curl
    else
        error "apt-get is required on s390x"
    fi

    # Install Go if missing
    if ! available go; then
        status "Installing Go 1.26.4..."
        wget -q "https://go.dev/dl/go1.26.4.linux-s390x.tar.gz" -O "$TEMP_DIR/go.tar.gz"
        $SUDO rm -rf /usr/local/go
        $SUDO tar -C /usr/local -xzf "$TEMP_DIR/go.tar.gz"
        export PATH="$PATH:/usr/local/go/bin"
    fi

    # Use mounted source if available, else clone
    if [ -d "/workspace/ollama-s390x" ]; then
        SOURCE_DIR="/workspace/ollama-s390x"
        status "Using existing source at $SOURCE_DIR"
    else
        SOURCE_DIR="$TEMP_DIR/ollama-s390x"
        status "Cloning ollama-s390x..."
        git clone --depth 1 https://github.com/Brice12347/ollama-s390x.git "$SOURCE_DIR"
    fi

    cd "$SOURCE_DIR"
    status "Building Ollama..."
    rm -rf build
    cmake -B build .
    cmake --build build --parallel "$(nproc)"

    status "Installing binary..."
    $SUDO install -m 755 ollama /usr/local/bin/ollama

    # Create ollama user
    if ! id ollama >/dev/null 2>&1; then
        $SUDO useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama 2>/dev/null || true
    fi

    status "Ollama installed successfully from source (s390x)!"
    ollama --version || true
    exit 0
fi

# ============================================
# Original amd64 / arm64 installation
# ============================================

download_and_extract() {
    local url_base="$1"
    local dest_dir="$2"
    local filename="$3"

    if curl --fail --silent --head --location "${url_base}/${filename}.tar.zst${VER_PARAM}" >/dev/null 2>&1; then
        if ! available zstd; then
            error "zstd is required for this version."
        fi
        status "Downloading ${filename}.tar.zst"
        curl --fail --show-error --location --progress-bar \
            "${url_base}/${filename}.tar.zst${VER_PARAM}" | \
            zstd -d | $SUDO tar -xf - -C "${dest_dir}"
        return 0
    fi

    status "Downloading ${filename}.tgz"
    curl --fail --show-error --location --progress-bar \
        "${url_base}/${filename}.tgz${VER_PARAM}" | \
        $SUDO tar -xzf - -C "${dest_dir}"
}

for BINDIR in /usr/local/bin /usr/bin /bin; do
    echo "$PATH" | grep -q "$BINDIR" && break || continue
done

OLLAMA_INSTALL_DIR=$(dirname "$BINDIR")

if [ -d "$OLLAMA_INSTALL_DIR/lib/ollama" ]; then
    status "Cleaning up old version..."
    $SUDO rm -rf "$OLLAMA_INSTALL_DIR/lib/ollama"
fi

status "Installing ollama to $OLLAMA_INSTALL_DIR"
$SUDO install -o0 -g0 -m755 -d "$BINDIR"
$SUDO install -o0 -g0 -m755 -d "$OLLAMA_INSTALL_DIR/lib/ollama"

download_and_extract "https://ollama.com/download" "$OLLAMA_INSTALL_DIR" "ollama-linux-${ARCH}"

if [ "$OLLAMA_INSTALL_DIR/bin/ollama" != "$BINDIR/ollama" ]; then
    $SUDO ln -sf "$OLLAMA_INSTALL_DIR/ollama" "$BINDIR/ollama"
fi

install_success() {
    status 'The Ollama API is now available at 127.0.0.1:11434.'
    status 'Install complete. Run "ollama" from the command line.'
}

trap install_success EXIT

configure_systemd() {
    if ! id ollama >/dev/null 2>&1; then
        $SUDO useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama 2>/dev/null || true
    fi
    if getent group render >/dev/null 2>&1; then $SUDO usermod -a -G render ollama; fi
    if getent group video >/dev/null 2>&1; then $SUDO usermod -a -G video ollama; fi
    $SUDO usermod -a -G ollama "$(whoami)"

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

    if systemctl is-system-running >/dev/null 2>&1; then
        $SUDO systemctl daemon-reload
        $SUDO systemctl enable ollama
        $SUDO systemctl restart ollama
    fi
}

if available systemctl; then
    configure_systemd
fi

# Skip GPU logic for s390x
if [ "$ARCH" = "s390x" ]; then
    status "IBM Z (s390x) - CPU only mode"
    install_success
    exit 0
fi


install_success
}

main
