#!/usr/bin/env bash

################################################################################
# bootstrap_dev_env.sh
#
# Purpose: Automate the setup of an s390x development environment for building
#          ollama from source with various build configurations.
#
# Usage: ./scripts/bootstrap_dev_env.sh [OPTIONS]
#
# Options:
#   -h, --help              Show this help message
#   -d, --dir DIR           Installation directory (default: $HOME/ollama-s390x)
#   -b, --build VARIANT     Build variant to compile (default: cpu)
#                           Options: cpu, cpu-no-vxe, cpu-debug, cpu-static, zdnn
#   --skip-deps             Skip dependency installation
#   --skip-clone            Skip repository cloning
#   --skip-build            Skip building ollama
#
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

################################################################################
# Configuration Variables
################################################################################

# Repository configuration
REPO_URL="${OLLAMA_REPO_URL:-https://github.com/Brice12347/ollama-s390x.git}"
REPO_BRANCH="${OLLAMA_REPO_BRANCH:-main}"

# Installation directory
INSTALL_DIR="${OLLAMA_INSTALL_DIR:-$HOME/ollama-s390x}"

# Build configuration
BUILD_VARIANT="${OLLAMA_BUILD_VARIANT:-cpu}"
SKIP_DEPS=false
SKIP_CLONE=false
SKIP_BUILD=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

################################################################################
# Helper Functions
################################################################################

# Print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Show help message
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Automate the setup of an s390x development environment for building ollama.

Options:
  -h, --help              Show this help message
  -d, --dir DIR           Installation directory (default: $HOME/ollama-s390x)
  -b, --build VARIANT     Build variant to compile (default: cpu)
                          Options: cpu, cpu-no-vxe, cpu-debug, cpu-static, zdnn
  --skip-deps             Skip dependency installation
  --skip-clone            Skip repository cloning
  --skip-build            Skip building ollama

Build Variants:
  cpu                     Standard CPU build (default)
  cpu-no-vxe              CPU build without Vector Extensions
  cpu-debug               CPU build with debug symbols
  cpu-static              Static CPU build
  zdnn                    Build with IBM zDNN acceleration

Environment Variables:
  OLLAMA_REPO_URL         Repository URL (default: https://github.com/Brice12347/ollama-s390x.git)
  OLLAMA_REPO_BRANCH      Repository branch (default: main)
  OLLAMA_INSTALL_DIR      Installation directory (default: $HOME/ollama-s390x)
  OLLAMA_BUILD_VARIANT    Build variant (default: cpu)

Examples:
  # Basic setup with default CPU build
  $0

  # Setup with custom directory and zDNN build
  $0 -d /opt/ollama -b zdnn

  # Only install dependencies
  $0 --skip-clone --skip-build

EOF
}

# Detect package manager
detect_package_manager() {
    if command -v apt-get &> /dev/null; then
        echo "apt"
    elif command -v dnf &> /dev/null; then
        echo "dnf"
    elif command -v yum &> /dev/null; then
        echo "yum"
    else
        print_error "No supported package manager found (apt, dnf, or yum)"
        exit 1
    fi
}

# Check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

################################################################################
# Dependency Installation Functions
################################################################################

# Install dependencies on Ubuntu/Debian
install_deps_apt() {
    print_info "Installing dependencies using apt..."
    
    sudo apt-get update
    
    # Core build tools
    sudo apt-get install -y \
        build-essential \
        git \
        curl \
        wget \
        ca-certificates
    
    # CMake (check version and install from Kitware if needed)
    if command_exists cmake; then
        CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
        print_info "Found CMake version: $CMAKE_VERSION"
    fi
    
    # Install CMake from apt (may need manual upgrade for older distros)
    sudo apt-get install -y cmake
    
    # Ninja build system
    sudo apt-get install -y ninja-build
    
    # Compiler (GCC)
    sudo apt-get install -y gcc g++
    
    # Python3
    sudo apt-get install -y python3 python3-pip
    
    # OpenBLAS and LAPACK libraries
    sudo apt-get install -y \
        libopenblas-dev \
        liblapack-dev \
        liblapacke-dev
    
    # Install Go
    install_go
    
    print_success "Dependencies installed successfully (apt)"
}

# Install dependencies on RHEL/Fedora/CentOS
install_deps_dnf() {
    local PKG_MGR=$1
    print_info "Installing dependencies using $PKG_MGR..."
    
    # Core build tools
    sudo $PKG_MGR install -y \
        gcc \
        gcc-c++ \
        make \
        git \
        curl \
        wget \
        ca-certificates
    
    # CMake
    sudo $PKG_MGR install -y cmake
    
    # Ninja build system
    sudo $PKG_MGR install -y ninja-build
    
    # Python3
    sudo $PKG_MGR install -y python3 python3-pip
    
    # OpenBLAS and LAPACK libraries
    sudo $PKG_MGR install -y \
        openblas-devel \
        lapack-devel
    
    # Install Go
    install_go
    
    print_success "Dependencies installed successfully ($PKG_MGR)"
}

# Install Go programming language
install_go() {
    if command_exists go; then
        GO_VERSION=$(go version | awk '{print $3}')
        print_info "Go is already installed: $GO_VERSION"
        return 0
    fi
    
    print_info "Installing Go..."
    
    # Determine architecture
    ARCH=$(uname -m)
    case $ARCH in
        s390x)
            GO_ARCH="s390x"
            ;;
        x86_64)
            GO_ARCH="amd64"
            ;;
        aarch64)
            GO_ARCH="arm64"
            ;;
        *)
            print_error "Unsupported architecture: $ARCH"
            exit 1
            ;;
    esac
    
    # Get latest Go version
    GO_VERSION=$(curl -s https://go.dev/VERSION?m=text | head -n1)
    GO_TARBALL="${GO_VERSION}.linux-${GO_ARCH}.tar.gz"
    GO_URL="https://go.dev/dl/${GO_TARBALL}"
    
    print_info "Downloading Go ${GO_VERSION} for ${GO_ARCH}..."
    
    # Download and install Go
    cd /tmp
    wget -q "$GO_URL"
    sudo rm -rf /usr/local/go
    sudo tar -C /usr/local -xzf "$GO_TARBALL"
    rm "$GO_TARBALL"
    
    # Add Go to PATH if not already present
    if ! grep -q "/usr/local/go/bin" ~/.bashrc; then
        echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
        echo 'export PATH=$PATH:$HOME/go/bin' >> ~/.bashrc
    fi
    
    export PATH=$PATH:/usr/local/go/bin
    export PATH=$PATH:$HOME/go/bin
    
    print_success "Go installed successfully: $(go version)"
}

# Main dependency installation function
install_dependencies() {
    print_info "Starting dependency installation..."
    
    PKG_MGR=$(detect_package_manager)
    print_info "Detected package manager: $PKG_MGR"
    
    case $PKG_MGR in
        apt)
            install_deps_apt
            ;;
        dnf|yum)
            install_deps_dnf "$PKG_MGR"
            ;;
        *)
            print_error "Unsupported package manager: $PKG_MGR"
            exit 1
            ;;
    esac
    
    # Verify critical dependencies
    print_info "Verifying installed dependencies..."
    
    local MISSING_DEPS=()
    
    for cmd in cmake ninja git go gcc python3; do
        if ! command_exists "$cmd"; then
            MISSING_DEPS+=("$cmd")
        fi
    done
    
    if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
        print_error "Missing dependencies: ${MISSING_DEPS[*]}"
        exit 1
    fi
    
    print_success "All dependencies verified"
}

################################################################################
# Repository Management Functions
################################################################################

# Clone the ollama-s390x repository
clone_repository() {
    print_info "Cloning ollama-s390x repository..."
    
    if [ -d "$INSTALL_DIR" ]; then
        print_warning "Directory $INSTALL_DIR already exists"
        read -p "Do you want to remove it and re-clone? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$INSTALL_DIR"
        else
            print_info "Skipping clone, using existing directory"
            return 0
        fi
    fi
    
    # Create parent directory if needed
    mkdir -p "$(dirname "$INSTALL_DIR")"
    
    # Clone repository
    git clone --branch "$REPO_BRANCH" "$REPO_URL" "$INSTALL_DIR"
    
    print_success "Repository cloned to $INSTALL_DIR"
}

################################################################################
# Configuration Functions
################################################################################

# Create .env.s390x configuration file
create_env_config() {
    print_info "Creating .env.s390x configuration file..."
    
    local ENV_FILE="$INSTALL_DIR/.env.s390x"
    
    cat > "$ENV_FILE" << 'EOF'
# s390x-specific environment configuration for ollama
# This file contains build settings optimized for IBM Z architecture

# Architecture
export GOARCH=s390x
export CGO_ENABLED=1

# Disable GPU backends (not available on s390x)
export OLLAMA_SKIP_CUDA=1
export OLLAMA_SKIP_ROCM=1
export OLLAMA_SKIP_VULKAN=1
export OLLAMA_SKIP_METAL=1

# Disable MLX backend (macOS-specific)
export OLLAMA_MLX_BACKENDS=""

# CPU-only build (default for s390x)
export OLLAMA_CPU_TARGET="s390x"

# OpenBLAS configuration
export USE_OPENBLAS=1

# Build parallelism
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)

# s390x architecture flags
export CFLAGS="-march=native -mtune=native"
export CXXFLAGS="-march=native -mtune=native"

# Optional: Enable Vector Extensions if available
# Uncomment the following line if your system supports Vector Extensions
# export OLLAMA_S390X_VXE=1

# Optional: Enable zDNN acceleration
# Uncomment and configure if using IBM zDNN
# export OLLAMA_ZDNN=1
# export ZDNN_INSTALL_DIR=/path/to/zdnn

# Debug settings (uncomment for debug builds)
# export CMAKE_BUILD_TYPE=Debug
# export OLLAMA_DEBUG=1

EOF
    
    print_success "Configuration file created: $ENV_FILE"
    print_info "To use this configuration, run: source $ENV_FILE"
}

################################################################################
# Build Functions
################################################################################

# Build ollama with specified variant
build_ollama() {
    local VARIANT=$1
    
    print_info "Building ollama with variant: $VARIANT"
    
    cd "$INSTALL_DIR"
    
    # Source environment configuration
    if [ -f ".env.s390x" ]; then
        source .env.s390x
    fi
    
    # Clean previous build
    if [ -d "build" ]; then
        print_info "Cleaning previous build..."
        rm -rf build
    fi
    
    # Set build-specific flags
    local CMAKE_FLAGS=""
    
    case $VARIANT in
        cpu)
            print_info "Building standard CPU variant..."
            CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Release"
            ;;
        cpu-no-vxe)
            print_info "Building CPU variant without Vector Extensions..."
            CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Release -DOLLAMA_S390X_VXE=OFF"
            export OLLAMA_S390X_VXE=0
            ;;
        cpu-debug)
            print_info "Building CPU variant with debug symbols..."
            CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Debug"
            export OLLAMA_DEBUG=1
            ;;
        cpu-static)
            print_info "Building static CPU variant..."
            CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF"
            ;;
        zdnn)
            print_info "Building with IBM zDNN acceleration..."
            CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Release -DOLLAMA_ZDNN=ON"
            export OLLAMA_ZDNN=1
            
            if [ -z "${ZDNN_INSTALL_DIR:-}" ]; then
                print_warning "ZDNN_INSTALL_DIR not set, using default /usr/local"
                export ZDNN_INSTALL_DIR=/usr/local
            fi
            CMAKE_FLAGS="$CMAKE_FLAGS -DZDNN_INSTALL_DIR=$ZDNN_INSTALL_DIR"
            ;;
        *)
            print_error "Unknown build variant: $VARIANT"
            print_info "Valid variants: cpu, cpu-no-vxe, cpu-debug, cpu-static, zdnn"
            exit 1
            ;;
    esac
    
    # Configure with CMake
    print_info "Configuring build with CMake..."
    cmake -B build $CMAKE_FLAGS .
    
    # Build
    print_info "Building ollama (this may take a while)..."
    cmake --build build --parallel $(nproc)
    
    print_success "Build completed successfully!"
    print_info "Binary location: $INSTALL_DIR/build/bin/ollama"
}

# Display build matrix information
show_build_matrix() {
    cat << EOF

${BLUE}═══════════════════════════════════════════════════════════════${NC}
${GREEN}Available Build Variants for s390x${NC}
${BLUE}═══════════════════════════════════════════════════════════════${NC}

${YELLOW}cpu${NC}           Standard CPU build (default)
                - Optimized for s390x architecture
                - Uses OpenBLAS for linear algebra
                - Release build with optimizations

${YELLOW}cpu-no-vxe${NC}    CPU build without Vector Extensions
                - For older s390x systems without VXE support
                - Portable across all s390x generations

${YELLOW}cpu-debug${NC}     CPU build with debug symbols
                - Includes debugging information
                - Useful for development and troubleshooting

${YELLOW}cpu-static${NC}    Static CPU build
                - Statically linked binary
                - Portable across different Linux distributions

${YELLOW}zdnn${NC}          Build with IBM zDNN acceleration
                - Requires IBM zDNN library installed
                - Hardware-accelerated neural network operations
                - Set ZDNN_INSTALL_DIR environment variable

${BLUE}═══════════════════════════════════════════════════════════════${NC}

EOF
}

################################################################################
# Main Script Logic
################################################################################

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -d|--dir)
                INSTALL_DIR="$2"
                shift 2
                ;;
            -b|--build)
                BUILD_VARIANT="$2"
                shift 2
                ;;
            --skip-deps)
                SKIP_DEPS=true
                shift
                ;;
            --skip-clone)
                SKIP_CLONE=true
                shift
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Main function
main() {
    print_info "Starting ollama s390x development environment setup..."
    print_info "Installation directory: $INSTALL_DIR"
    print_info "Build variant: $BUILD_VARIANT"
    echo
    
    # Install dependencies
    if [ "$SKIP_DEPS" = false ]; then
        install_dependencies
        echo
    else
        print_warning "Skipping dependency installation"
    fi
    
    # Clone repository
    if [ "$SKIP_CLONE" = false ]; then
        clone_repository
        echo
    else
        print_warning "Skipping repository clone"
    fi
    
    # Create environment configuration
    create_env_config
    echo
    
    # Show build matrix
    show_build_matrix
    
    # Build ollama
    if [ "$SKIP_BUILD" = false ]; then
        build_ollama "$BUILD_VARIANT"
        echo
    else
        print_warning "Skipping build"
    fi
    
    # Final instructions
    print_success "Setup complete!"
    echo
    print_info "Next steps:"
    echo "  1. Source the environment configuration:"
    echo "     source $INSTALL_DIR/.env.s390x"
    echo
    echo "  2. Run ollama:"
    echo "     $INSTALL_DIR/build/bin/ollama serve"
    echo
    echo "  3. To rebuild with a different variant:"
    echo "     $0 -d $INSTALL_DIR -b <variant> --skip-deps --skip-clone"
    echo
    print_info "For more information, see: $INSTALL_DIR/README.md"
}

# Entry point
parse_arguments "$@"
main

# Made with Bob
