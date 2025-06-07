#!/bin/sh
# This script installs Ollama on Linux.
# It detects the current operating system architecture and installs the appropriate version of Ollama.

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

[ "$(uname -s)" = "Linux" ] || error 'This script is intended to run on Linux only.'

ARCH=$(uname -m)
case "$ARCH" in
    x86_64) ARCH="amd64" ;;
    aarch64|arm64) ARCH="arm64" ;;
    *) error "Unsupported architecture: $ARCH" ;;
esac

IS_WSL2=false

KERN=$(uname -r)
case "$KERN" in
    *icrosoft*WSL2 | *icrosoft*wsl2) IS_WSL2=true;;
    *icrosoft) error "Microsoft WSL1 is not currently supported. Please use WSL2 with 'wsl --set-version <distro> 2'" ;;
    *) ;;
esac

VER_PARAM="${OLLAMA_VERSION:+?version=$OLLAMA_VERSION}"

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
status "Downloading Linux ${ARCH} bundle"
curl --fail --show-error --location --progress-bar \
    "https://ollama.com/download/ollama-linux-${ARCH}.tgz${VER_PARAM}" | \
    $SUDO tar -xzf - -C "$OLLAMA_INSTALL_DIR"

if [ "$OLLAMA_INSTALL_DIR/bin/ollama" != "$BINDIR/ollama" ] ; then
    status "Making ollama accessible in the PATH in $BINDIR"
    $SUDO ln -sf "$OLLAMA_INSTALL_DIR/ollama" "$BINDIR/ollama"
fi

# Check for NVIDIA JetPack systems with additional downloads
if [ -f /etc/nv_tegra_release ] ; then
    if grep R36 /etc/nv_tegra_release > /dev/null ; then
        status "Downloading JetPack 6 components"
        curl --fail --show-error --location --progress-bar \
            "https://ollama.com/download/ollama-linux-${ARCH}-jetpack6.tgz${VER_PARAM}" | \
            $SUDO tar -xzf - -C "$OLLAMA_INSTALL_DIR"
    elif grep R35 /etc/nv_tegra_release > /dev/null ; then
        status "Downloading JetPack 5 components"
        curl --fail --show-error --location --progress-bar \
            "https://ollama.com/download/ollama-linux-${ARCH}-jetpack5.tgz${VER_PARAM}" | \
            $SUDO tar -xzf - -C "$OLLAMA_INSTALL_DIR"
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
                ascend) available lspci && lspci -d '19e5:' | grep -q 'Processing accelerators: Huawei' || return 1 ;;
            esac ;;
        lshw)
            case $2 in
                nvidia) available lshw && $SUDO lshw -c display -numeric -disable network | grep -q 'vendor: .* \[10DE\]' || return 1 ;;
                amdgpu) available lshw && $SUDO lshw -c display -numeric -disable network | grep -q 'vendor: .* \[1002\]' || return 1 ;;
                ascend) available lshw && $SUDO lshw -c display -numeric -disable network | grep -q 'vendor: .* \[19E5\]' || return 1 ;;
            esac ;;
        nvidia-smi) available nvidia-smi || return 1 ;;
        npu-smi) available npu-smi || return 1 ;;
    esac
}

if check_gpu nvidia-smi; then
    status "NVIDIA GPU installed."
    exit 0
fi

if check_gpu npu-smi; then
    status "ASCEND GPU installed."
    exit 0
fi

if ! check_gpu lspci nvidia && ! check_gpu lshw nvidia && ! check_gpu lspci amdgpu && ! check_gpu lshw amdgpu && ! check_gpu lspci ascend && ! check_gpu lshw ascend; then
    install_success
    warning "No NVIDIA/AMD/Ascend GPU detected. Ollama will run in CPU-only mode."
    exit 0
fi

if check_gpu lspci amdgpu || check_gpu lshw amdgpu; then
    status "Downloading Linux ROCm ${ARCH} bundle"
    curl --fail --show-error --location --progress-bar \
        "https://ollama.com/download/ollama-linux-${ARCH}-rocm.tgz${VER_PARAM}" | \
        $SUDO tar -xzf - -C "$OLLAMA_INSTALL_DIR"

    install_success
    status "AMD GPU ready."
    exit 0
fi

# Ascend
install_ascend_driver_yum() {
    status 'Installing ASCNED driver version: $ASCEND_DRIVER_VERSION ,firmware version: $ASCEND_FIRMWARE_VERSION...'
    $SUDO $PACKAGE_MANAGER -y install gcc gcc-c++ make cmake unzip zlib-devel libffi-devel openssl-devel pciutils net-tools sqlite-devel lapack-devel gcc-gfortran python3-devel
    $SUDO groupadd -g HwHiAiUser
    $SUDO useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    $SUDO usermod -aG HwHiAiUser $USER

    # driver version, mabey get from it
    # npu-smi info
    # +------------------------------------------------------------------------------------------------+
    # | npu-smi 24.1.rc1.b060            Version: 24.1.rc1.b060                                        |
    wget '--header=Referer: https://www.hiascend.com/' "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Ascend HDK/Ascend HDK $ASCEND_DRIVER_VERSION/Ascend-hdk-$1-npu-driver_$(echo "$ASCEND_DRIVER_VERSION" | tr '[:upper:]' '[:lower:]')_linux-$(uname -m).run"
    $SUDO sh Ascend-hdk-$1-npu-driver_$(echo "$ASCEND_DRIVER_VERSION" | tr '[:upper:]' '[:lower:]')_linux-$(uname -m).run --full --install-for-all
    rm -rf ./Ascend-hdk-$1-npu-driver_$(echo "$ASCEND_DRIVER_VERSION" | tr '[:upper:]' '[:lower:]')_linux-$(uname -m).run

    wget '--header=Referer: https://www.hiascend.com/' "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Ascend HDK/Ascend HDK $ASCEND_DRIVER_VERSION/Ascend-hdk-$1-npu-firmware_$ASCEND_FIRMWARE_VERSION.231.run"
    $SUDO sh Ascend-hdk-$1-npu-firmware_$ASCEND_FIRMWARE_VERSION.231.run --full
    rm -rf ./Ascend-hdk-$1-npu-firmware_$ASCEND_FIRMWARE_VERSION.231.run
}

install_ascend_driver_apt() {
    status 'Installing ASCNED driver version: $ASCEND_DRIVER_VERSION ,firmware version: $ASCEND_FIRMWARE_VERSION...'
    apt-get -y install gcc g++ make cmake zlib1g zlib1g-dev openssl libsqlite3-dev libssl-dev libffi-dev unzip pciutils net-tools libblas-dev gfortran libblas3 python3-dev
    groupadd -g HwHiAiUser
    useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
    usermod -aG HwHiAiUser $USER

    # driver version,mabey get from it
    # npu-smi info
    # +------------------------------------------------------------------------------------------------+
    # | npu-smi 24.1.rc1.b060            Version: 24.1.rc1.b060                                        |
    wget '--header=Referer: https://www.hiascend.com/' "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Ascend HDK/Ascend HDK $ASCEND_DRIVER_VERSION/Ascend-hdk-$1-npu-driver_$(echo "$ASCEND_DRIVER_VERSION" | tr '[:upper:]' '[:lower:]')_linux-$(uname -m).run"
    sh Ascend-hdk-$1-npu-driver_$(echo "$ASCEND_DRIVER_VERSION" | tr '[:upper:]' '[:lower:]')_linux-$(uname -m).run --full --install-for-all
    rm -rf ./Ascend-hdk-$1-npu-driver_$(echo "$ASCEND_DRIVER_VERSION" | tr '[:upper:]' '[:lower:]')_linux-$(uname -m).run

    wget '--header=Referer: https://www.hiascend.com/' "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Ascend HDK/Ascend HDK $ASCEND_DRIVER_VERSION/Ascend-hdk-$1-npu-firmware_$ASCEND_FIRMWARE_VERSION.231.run"
    $SUDO sh Ascend-hdk-$1-npu-firmware_$ASCEND_FIRMWARE_VERSION.231.run --full
    rm -rf ./Ascend-hdk-$1-npu-firmware_$ASCEND_FIRMWARE_VERSION.231.run
}

install_ascend_cann() {
    status 'Installing ASCNED CANN version: $ASCEND_CANN_VERSION...'
    echo "ASCEND_CANN_VERSION=$ASCEND_CANN_VERSION, 1st paramenter=$1"
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple attrs numpy==1.24.0 decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py wheel typing_extensions
    wget '--header=Referer: https://www.hiascend.com/' "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN $ASCEND_CANN_VERSION/Ascend-cann-toolkit_${ASCEND_CANN_VERSION}_linux-$(uname -m).run"
    bash Ascend-cann-toolkit_${ASCEND_CANN_VERSION}_linux-$(uname -m).run --install
    rm -rf ./Ascend-cann-toolkit_${ASCEND_CANN_VERSION}_linux-$(uname -m).run

    wget '--header=Referer: https://www.hiascend.com/' "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN $ASCEND_CANN_VERSION/Ascend-cann-kernels-$1_${ASCEND_CANN_VERSION}_linux-$(uname -m).run"
    bash Ascend-cann-kernels-$1_${ASCEND_CANN_VERSION}_linux-$(uname -m).run --install
    rm -rf ./Ascend-cann-kernels-$1_${ASCEND_CANN_VERSION}_linux.run
}

install_ggml_cann() {
    status 'Installing ggml-cann: soc type=$1'
    local package_name
    case $1 in
        910b) package_name="ollama-linux-${ARCH}-cann-atlas-a2" ;;
        310p) package_name="ollama-linux-${ARCH}-cann-300i-duo" ;;
        *) exit ;;
    esac
    status "Downloading ${package_name} components"
    curl --fail --show-error --location --progress-bar \
        "https://ollama.com/download/${package_name}.tgz${VER_PARAM}" | \
        $SUDO tar -xzf - -C "$OLLAMA_INSTALL_DIR"
}

# use env val: ASCEND_DRIVER_VERSION ASCEND_FIRMWARE_VERSIO and ASCEND_CANN_VERSION to get version 
# ref:https://ascend.github.io/docs/sources/ascend/quick_install.html
if check_gpu lspci ascend || check_gpu lshw ascend; then
    if [ -z "$ASCEND_DRIVER_VERSION" ]; then
        ASCEND_DRIVER_VERSION="25.0.RC1.1"
    fi
    if [ -z "$ASCEND_FIRMWARE_VERSION" ]; then
        ASCEND_FIRMWARE_VERSION="7.7.0.1"
    fi
    if [ -z "$ASCEND_CANN_VERSION" ]; then
        ASCEND_CANN_VERSION="8.1.RC1"
    fi
    echo "after set ASCEND_DRIVER_VERSION=${ASCEND_DRIVER_VERSION}";

    type=910b
    if available npu-smi; then
        type=$(npu-smi info -m | grep 'Ascend' | awk '{print $5}' | head -n 1 | tr '[:upper:]' '[:lower:]'|sed 's/[0-9]$//')
    else
        if ! available lspci; then
            case $OS_NAME in
                openeuler) yum install pciutils ;;
                ubuntu) apt install pciutils ;;
                *) exit ;;
            esac
        fi
        if lspci -n -D | grep -q d802; then
            echo "Ascend device is atlas 800 A2, soc type ${type}"
        elif lspci -n -D | grep -q d500; then
            type=310p
            echo "Ascend device is atlas 300, soc type ${type}"
        else
            echo "Auto-detect ascend device id failed, please first install pciutils."
        fi
    fi

    if ! available npu-smi; then
        case $OS_NAME in
            openeuler) install_ascend_driver_yum $type ;;
            ubuntu) install_ascend_driver_apt $type ;;
            *) exit ;;
        esac
    else
        echo "Ascend driver has been installed. Please confirm that the version is [23.0.X -> 25.0.X]. If it is not, please uninstall it manually and then reinstall ollama."
    fi

    install_ascend_cann $type
    echo "source ~/Ascend/ascend-toolkit/set_env.sh" >> ~/.bashrc
    source ~/.bashrc

    install_ggml_cann $type
fi

# NVIDIA
CUDA_REPO_ERR_MSG="NVIDIA GPU detected, but your OS and Architecture are not supported by NVIDIA.  Please install the CUDA driver manually https://docs.nvidia.com/cuda/cuda-installation-guide-linux/"
# ref: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#rhel-7-centos-7
# ref: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#rhel-8-rocky-8
# ref: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#rhel-9-rocky-9
# ref: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#fedora
install_cuda_driver_yum() {
    status 'Installing NVIDIA repository...'
    
    case $PACKAGE_MANAGER in
        yum)
            $SUDO $PACKAGE_MANAGER -y install yum-utils
            if curl -I --silent --fail --location "https://developer.download.nvidia.com/compute/cuda/repos/$1$2/$(uname -m | sed -e 's/aarch64/sbsa/')/cuda-$1$2.repo" >/dev/null ; then
                $SUDO $PACKAGE_MANAGER-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/$1$2/$(uname -m | sed -e 's/aarch64/sbsa/')/cuda-$1$2.repo
            else
                error $CUDA_REPO_ERR_MSG
            fi
            ;;
        dnf)
            if curl -I --silent --fail --location "https://developer.download.nvidia.com/compute/cuda/repos/$1$2/$(uname -m | sed -e 's/aarch64/sbsa/')/cuda-$1$2.repo" >/dev/null ; then
                $SUDO $PACKAGE_MANAGER config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/$1$2/$(uname -m | sed -e 's/aarch64/sbsa/')/cuda-$1$2.repo
            else
                error $CUDA_REPO_ERR_MSG
            fi
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

    if [ "$1" = 'centos' ] || [ "$1$2" = 'rhel7' ]; then
        $SUDO $PACKAGE_MANAGER -y install nvidia-driver-latest-dkms
    fi

    $SUDO $PACKAGE_MANAGER -y install cuda-drivers
}

# ref: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu
# ref: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#debian
install_cuda_driver_apt() {
    status 'Installing NVIDIA repository...'
    if curl -I --silent --fail --location "https://developer.download.nvidia.com/compute/cuda/repos/$1$2/$(uname -m | sed -e 's/aarch64/sbsa/')/cuda-keyring_1.1-1_all.deb" >/dev/null ; then
        curl -fsSL -o $TEMP_DIR/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/$1$2/$(uname -m | sed -e 's/aarch64/sbsa/')/cuda-keyring_1.1-1_all.deb
    else
        error $CUDA_REPO_ERR_MSG
    fi

    case $1 in
        debian)
            status 'Enabling contrib sources...'
            $SUDO sed 's/main/contrib/' < /etc/apt/sources.list | $SUDO tee /etc/apt/sources.list.d/contrib.list > /dev/null
            if [ -f "/etc/apt/sources.list.d/debian.sources" ]; then
                $SUDO sed 's/main/contrib/' < /etc/apt/sources.list.d/debian.sources | $SUDO tee /etc/apt/sources.list.d/contrib.sources > /dev/null
            fi
            ;;
    esac

    status 'Installing CUDA driver...'
    $SUDO dpkg -i $TEMP_DIR/cuda-keyring.deb
    $SUDO apt-get update

    [ -n "$SUDO" ] && SUDO_E="$SUDO -E" || SUDO_E=
    DEBIAN_FRONTEND=noninteractive $SUDO_E apt-get -y install cuda-drivers -q
}

if [ ! -f "/etc/os-release" ]; then
    error "Unknown distribution. Skipping CUDA installation."
fi

. /etc/os-release

OS_NAME=$ID
OS_VERSION=$VERSION_ID

PACKAGE_MANAGER=
for PACKAGE_MANAGER in dnf yum apt-get; do
    if available $PACKAGE_MANAGER; then
        break
    fi
done

if [ -z "$PACKAGE_MANAGER" ]; then
    error "Unknown package manager. Skipping CUDA installation."
fi

if ! check_gpu nvidia-smi || [ -z "$(nvidia-smi | grep -o "CUDA Version: [0-9]*\.[0-9]*")" ]; then
    case $OS_NAME in
        centos|rhel) install_cuda_driver_yum 'rhel' $(echo $OS_VERSION | cut -d '.' -f 1) ;;
        rocky) install_cuda_driver_yum 'rhel' $(echo $OS_VERSION | cut -c1) ;;
        fedora) [ $OS_VERSION -lt '39' ] && install_cuda_driver_yum $OS_NAME $OS_VERSION || install_cuda_driver_yum $OS_NAME '39';;
        amzn) install_cuda_driver_yum 'fedora' '37' ;;
        debian) install_cuda_driver_apt $OS_NAME $OS_VERSION ;;
        ubuntu) install_cuda_driver_apt $OS_NAME $(echo $OS_VERSION | sed 's/\.//') ;;
        *) exit ;;
    esac
fi

if ! lsmod | grep -q nvidia || ! lsmod | grep -q nvidia_uvm; then
    KERNEL_RELEASE="$(uname -r)"
    case $OS_NAME in
        rocky) $SUDO $PACKAGE_MANAGER -y install kernel-devel kernel-headers ;;
        centos|rhel|amzn) $SUDO $PACKAGE_MANAGER -y install kernel-devel-$KERNEL_RELEASE kernel-headers-$KERNEL_RELEASE ;;
        fedora) $SUDO $PACKAGE_MANAGER -y install kernel-devel-$KERNEL_RELEASE ;;
        debian|ubuntu) $SUDO apt-get -y install linux-headers-$KERNEL_RELEASE ;;
        *) exit ;;
    esac

    NVIDIA_CUDA_VERSION=$($SUDO dkms status | awk -F: '/added/ { print $1 }')
    if [ -n "$NVIDIA_CUDA_VERSION" ]; then
        $SUDO dkms install $NVIDIA_CUDA_VERSION
    fi

    if lsmod | grep -q nouveau; then
        status 'Reboot to complete NVIDIA CUDA driver install.'
        exit 0
    fi

    $SUDO modprobe nvidia
    $SUDO modprobe nvidia_uvm
fi

# make sure the NVIDIA modules are loaded on boot with nvidia-persistenced
if available nvidia-persistenced; then
    $SUDO touch /etc/modules-load.d/nvidia.conf
    MODULES="nvidia nvidia-uvm"
    for MODULE in $MODULES; do
        if ! grep -qxF "$MODULE" /etc/modules-load.d/nvidia.conf; then
            echo "$MODULE" | $SUDO tee -a /etc/modules-load.d/nvidia.conf > /dev/null
        fi
    done
fi

status "GPU ready."
install_success
