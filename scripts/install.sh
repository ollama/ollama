#!/bin/sh
# This script installs Ollama on Linux.
# It detects the current operating system architecture and installs the appropriate version of Ollama.

NVIDIA_REPOS_URL='https://developer.download.nvidia.com/compute/cuda/repos'

available() { command -v "$1" >/dev/null; }

red="$({
    /usr/bin/tput bold || :
    /usr/bin/tput setaf 1 || :
} 2>&-)"
plain="$({ /usr/bin/tput sgr0 || :; } 2>&-)"
yellow="$({
    /usr/bin/tput bold || :
    /usr/bin/tput setaf 11 || :
} 2>&-)"

status() { printf '>>> %s \n' "$*" >&2; }
error() {
    printf '%sERROR:%s %s\n' "$red" "$*" "$plain" >&2
    exit 1
}
warning() { printf '%sWARNING:%s %s\n' "$yellow" "$*" "${plain}" >&2; }

[ "$(uname -s)" = "Linux" ] || error 'This script is intended to run on Linux only.'

arch=$(uname -m)
case "$arch" in
x86_64) arch="amd64" ;;
aarch64 | arm64) arch="arm64" ;;
*) error "Unsupported architecture: $arch" ;;
esac

case "$(uname -r)" in
*icrosoft*WSL2 | *icrosoft*wsl2) is_wsl2() { :; } ;;
*icrosoft) error "Microsoft WSL1 is not currently supported. Please use WSL2 with 'wsl --set-version <distro> 2'" ;;
*) is_wsl2() { false; } ;;
esac

require() {
    rc=0
    for tool; do
        if ! available "$tool"; then
            rc=1
            printf %s\\n "$tool"
        fi
    done
    return $rc
}

ver_param="${OLLAMA_VERSION:+version=$OLLAMA_VERSION}"

if [ "$(id -u)" -ne 0 ]; then
    if ! available sudo; then
        error 'This script requires superuser permissions. Please re-run as root.'
    fi
else
    if ! available sudo; then
        # Dummy sudo if not available
        sudo() {
            while [ "${1#-}" != "$1" ]; do shift; done
            "$@"
        }
    fi
fi

if ! needs=$(require curl awk grep sed tee xargs); then
    error "$(printf 'The following tools are required but missing:\n%s\n' "$needs")"
fi

for bin_dir in /usr/local/bin /usr/bin /bin; do
    case :$PATH: in
    *":$bin_dir:"* | *":$bin_dir/:"*) break ;;
    esac
    false
done || error 'Cannot determine installation directory'
ollama_install_dir=${bin_dir%/*}

lib_ollama=$ollama_install_dir/lib/ollama

if [ -d "$lib_ollama" ]; then
    status "Cleaning up old version at $lib_ollama"
    sudo rm -rf -- "$lib_ollama"
fi

status "Installing ollama to $ollama_install_dir"
sudo install -o0 -g0 -m755 -d "$bin_dir" "$lib_ollama"
status "Downloading Linux ${arch} bundle"

url_arch=$(url_encode "$arch")

curl --fail --show-error --location --progress-bar \
    --get --data-urlencode "$ver_param" \
    "https://ollama.com/download/ollama-linux-${url_arch}.tgz" |
    sudo tar -xzf - -C "$ollama_install_dir"

if [ "$ollama_install_dir/bin/ollama" != "$bin_dir/ollama" ]; then
    status "Making ollama accessible in the PATH in $bin_dir"
    sudo ln -sf "$ollama_install_dir/ollama" "$bin_dir/ollama"
fi

# Check for NVIDIA JetPack systems with additional downloads
if [ -f /etc/nv_tegra_release ]; then
    if grep R36 /etc/nv_tegra_release >/dev/null; then
        status "Downloading JetPack 6 components"
        curl --fail --show-error --location --progress-bar \
            --get --data-urlencode "$ver_param" \
            "https://ollama.com/download/ollama-linux-${url_arch}-jetpack6.tgz" |
            sudo tar -xzf - -C "$ollama_install_dir"
    elif grep R35 /etc/nv_tegra_release >/dev/null; then
        status "Downloading JetPack 5 components"
        curl --fail --show-error --location --progress-bar \
            --get --data-urlencode "$ver_param" \
            "https://ollama.com/download/ollama-linux-${url_arch}-jetpack5.tgz" |
            sudo tar -xzf - -C "$ollama_install_dir"
    else
        warning "Unsupported JetPack version detected.  GPU may not be supported"
    fi
fi

install_success() {
    status 'The Ollama API is now available at 127.0.0.1:11434.'
    status 'Install complete. Run "ollama" from the command line.'
}
trap install_success EXIT INT TERM

# Everything from this point onwards is optional.

configure_systemd() {
    if ! id ollama >/dev/null 2>&1; then
        status "Creating ollama user..."
        sudo useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama
    fi
    if getent group render >/dev/null 2>&1; then
        status "Adding ollama user to render group..."
        sudo usermod -a -G render ollama
    fi
    if getent group video >/dev/null 2>&1; then
        status "Adding ollama user to video group..."
        sudo usermod -a -G video ollama
    fi

    if [ "$(id -u)" != 0 ]; then
        status "Adding current user to ollama group..."
        sudo usermod -a -G ollama "$(id -un)"
    fi

    status "Creating ollama systemd service..."
    sudo tee /etc/systemd/system/ollama.service >/dev/null <<EOF
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=$bin_dir/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=$PATH"

[Install]
WantedBy=default.target
EOF
    systemctl_running="$(systemctl is-system-running || true)"
    case "$systemctl_running" in
    running | degraded)
        status "Enabling and starting ollama service..."
        sudo systemctl daemon-reload
        sudo systemctl enable ollama

        start_service() {
            sudo systemctl restart ollama
        }
        trap start_service EXIT INT TERM
        ;;
    *)
        warning "systemd is not running"
        if is_wsl2; then
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
if is_wsl2; then
    if available nvidia-smi && nvidia-smi | grep -q "CUDA Version: [0-9]*\.[0-9]*"; then
        status "Nvidia GPU detected."
    fi
    install_success
    exit 0
fi

# Don't attempt to install drivers on Jetson systems
if [ -f /etc/nv_tegra_release ]; then
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
    available "$1" || return 1
    # Look for devices based on vendor ID for NVIDIA and AMD
    case "$1" in
    lspci)
        case "$2" in
        nvidia) lspci -d '10de:' | grep -q 'NVIDIA' || return 1 ;;
        amdgpu) lspci -d '1002:' | grep -q 'AMD' || return 1 ;;
        esac
        ;;
    lshw)
        case "$2" in
        nvidia) sudo lshw -c display -numeric -disable network | grep -q 'vendor: .* \[10DE\]' || return 1 ;;
        amdgpu) sudo lshw -c display -numeric -disable network | grep -q 'vendor: .* \[1002\]' || return 1 ;;
        esac
        ;;
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
    status "Downloading Linux ROCm ${arch} bundle"
    curl --fail --show-error --location --progress-bar \
        --get --data-urlencode "$ver_param" \
        "https://ollama.com/download/ollama-linux-${url_arch}-rocm.tgz" |
        sudo tar -xzf - -C "$ollama_install_dir"

    install_success
    status "AMD GPU ready."
    exit 0
fi

cuda_repo_err_message="NVIDIA GPU detected, but your OS and Architecture are not supported by NVIDIA.  Please install the CUDA driver manually https://docs.nvidia.com/cuda/cuda-installation-guide-linux/"
# ref: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#rhel-7-centos-7
# ref: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#rhel-8-rocky-8
# ref: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#rhel-9-rocky-9
# ref: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#fedora
install_cuda_driver_yum() {
    u_os_name=$(url_encode "$1")
    u_os_version=$(url_encode "$2")
    u_cuda_arch=$(url_encode "$(uname -m | sed -e 's/aarch64/sbsa/')")

    status 'Installing NVIDIA repository...'

    cuda_repo_url="${NVIDIA_REPOS_URL}/${u_os_name}${u_os_version}/${u_cuda_arch}/cuda-${u_os_name}${u_os_version.repo}"

    case "$package_manager" in yum)
        sudo "$package_manager" -y install yum-utils
        ;;
    esac
    case "$package_manager" in yum | dnf)
        curl -I --silent --fail --location "$cuda_repo_url" >/dev/null || error "$cuda_repo_err_message"
        sudo "$package_manager" config-manager --add-repo "$cuda_repo_url"
        ;;
    esac

    case "$1" in
    rhel)
        status 'Installing EPEL repository...'
        # EPEL is required for third-party dependencies such as dkms and libvdpau
        epel_url="https://dl.fedoraproject.org/pub/epel/epel-release-latest-$u_os_version.noarch.rpm"
        sudo "$package_manager" -y install "$epel_url" || true
        ;;
    esac

    status 'Installing CUDA driver...'

    if [ "$1" = 'centos' ] || [ "$1$2" = 'rhel7' ]; then
        sudo "$package_manager" -y install nvidia-driver-latest-dkms
    fi

    sudo "$package_manager" -y install cuda-drivers
}

url_encode() {
    awk 'BEGIN {
        for (i=0; i<125; i++) m[sprintf("%c", i)] = i
        for (i=1; i<=length(ARGV[1]); i++) {
            c = substr(ARGV[1], i, 1)
            if (c ~ /[[:alnum:]_.!~*\47()-]/) printf "%s", c
            else printf "%%%02X", m[c]
        }
    }' "$1"
}

# ref: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu
# ref: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#debian
install_cuda_driver_apt() {
    u_os_name=$(url_encode "$1")
    u_os_version=$(url_encode "$2")
    u_cuda_arch=$(url_encode "$(uname -m | sed -e 's/aarch64/sbsa/')")
    (
        status 'Installing NVIDIA repository...'

        tmp_dir=
        trap 'rm -rf -- "$tmp_dir"' EXIT INT TERM
        tmp_dir=$(mktemp -d) || error 'Cannot create temp dir'
        cuda_keyring_url="${NVIDIA_REPOS_URL}/${u_os_name}${u_os_version}/${u_cuda_arch}/cuda-keyring_1.1-1_all.deb"

        if ! curl -fsSL -I --fail --location "$cuda_keyring_url" >/dev/null; then
            error "$cuda_repo_err_message"
        fi
        curl -fsSL -o "$tmp_dir/cuda-keyring.deb" "$cuda_keyring_url"

        case "$1" in
        debian)
            status 'Enabling contrib sources...'
            sudo sed 's/main/contrib/' /etc/apt/sources.list |
                sudo tee /etc/apt/sources.list.d/contrib.list >/dev/null
            if [ -f "/etc/apt/sources.list.d/debian.sources" ]; then
                sudo sed 's/main/contrib/' /etc/apt/sources.list.d/debian.sources |
                    sudo tee /etc/apt/sources.list.d/contrib.sources >/dev/null
            fi
            ;;
        esac

        status 'Installing CUDA driver...'
        sudo dpkg -i "$tmp_dir/cuda-keyring.deb"
        sudo apt-get update

        DEBIAN_FRONTEND=noninteractive sudo -e apt-get -y install cuda-drivers -q
    ) || exit
}

if [ ! -f "/etc/os-release" ]; then
    error "Unknown distribution. Skipping CUDA installation."
fi

. /etc/os-release || error 'Cannot determine OS release'

u_os_name=$ID
u_os_version=$VERSION_ID

for package_manager in apt-get dnf yum; do
    available "$package_manager" && break
    package_manager=
done

if [ -z "$package_manager" ]; then
    error "Unknown package manager. Skipping CUDA installation."
fi

if ! check_gpu nvidia-smi || { nvidia-smi | grep -q 'CUDA Version: [0-9]*\.[0-9]*'; }; then
    case "$u_os_name" in
    centos | rhel) install_cuda_driver_yum 'rhel' "${u_os_version%%.*}" ;;
    rocky) install_cuda_driver_yum 'rhel' "${u_os_version%"${u_os_version#?}"}" ;;
    fedora)
        if [ "${u_os_version%%.*}" -lt '39' ]; then
            install_cuda_driver_yum "$u_os_name" "$u_os_version"
        else
            install_cuda_driver_yum "$u_os_name" '39'
        fi
        ;;
    amzn) install_cuda_driver_yum 'fedora' '37' ;;
    debian) install_cuda_driver_apt "$u_os_name" "$u_os_version" ;;
    ubuntu) install_cuda_driver_apt "$u_os_name" "${u_os_version%%.*}${u_os_version#*.}" ;;
    *) exit ;;
    esac
fi

if ! { grep -q nvidia /proc/modules && grep -q nvidia_uvm /proc/modules; }; then
    kernel_release="$(uname -r)"
    case "$u_os_name" in
    rocky) sudo "$package_manager" -y install kernel-devel kernel-headers ;;
    centos | rhel | amzn) sudo "$package_manager" -y install "kernel-devel-$kernel_release" "kernel-headers-$kernel_release" ;;
    fedora) sudo "$package_manager" -y install "kernel-devel-$kernel_release" ;;
    debian | ubuntu) sudo apt-get -y install "linux-headers-$kernel_release" ;;
    *) exit ;;
    esac

    nvidia_cuda_version=$(sudo dkms status | awk -F: '/added/ { print $1 }')
    if [ -n "$nvidia_cuda_version" ]; then
        sudo dkms install "$nvidia_cuda_version"
    fi

    if grep -q nouveau /proc/modules; then
        status 'Reboot to complete NVIDIA CUDA driver install.'
        exit 0
    fi

    sudo modprobe nvidia nvidia_uvm
fi

# make sure the NVIDIA modules are loaded on boot with nvidia-persistenced
if available nvidia-persistenced; then
    sudo touch /etc/modules-load.d/nvidia.conf
    for module in nvidia nvidia-uvm; do
        if ! grep -qxF "$module" /etc/modules-load.d/nvidia.conf; then
            printf %s\\n "$module" | sudo tee -a /etc/modules-load.d/nvidia.conf >/dev/null
        fi
    done
fi

status "NVIDIA GPU ready."
install_success
