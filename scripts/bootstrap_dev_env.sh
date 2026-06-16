#!/usr/bin/env bash

################################################################################
# bootstrap_dev_env.sh
#
# Purpose: Automate the setup of an s390x development environment for building
#          ollama from source within the z-spyre-runtimes container.
#
# Usage: ./scripts/bootstrap_dev_env.sh [OPTIONS]
#
# Options:
#   -h, --help              Show this help message
#   -b, --build VARIANT     Build variant to compile (default: cpu)
#                           Options: cpu, cpu-no-vxe, cpu-debug, cpu-static, zdnn
#   --skip-container        Skip z-spyre-runtimes container setup
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

# z-spyre-runtimes configuration
ZSPYRE_REPO_URL="git@github.ibm.com:zosdev/z-spyre-runtimes.git"
ZSPYRE_DIR="$HOME/z-spyre-runtimes"

# ollama-s390x configuration
OLLAMA_REPO_URL="git@github.com:Brice12347/ollama-s390x.git"
OLLAMA_DIR="$HOME/ollama-s390x"

# Build configuration
BUILD_VARIANT="${OLLAMA_BUILD_VARIANT:-cpu}"
SKIP_CONTAINER=false
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

Automate the setup of an s390x development environment for building ollama
within the z-spyre-runtimes container.

Options:
  -h, --help              Show this help message
  -b, --build VARIANT     Build variant to compile (default: cpu)
                          Options: cpu, cpu-no-vxe, cpu-debug, cpu-static, zdnn
  --skip-container        Skip z-spyre-runtimes container setup
  --skip-deps             Skip dependency installation
  --skip-clone            Skip repository cloning
  --skip-build            Skip building ollama

Build Variants:
  cpu                     Standard CPU build (default)
  cpu-no-vxe              CPU build without Vector Extensions
  cpu-debug               CPU build with debug symbols
  cpu-static              Static CPU build
  zdnn                    Build with IBM zDNN acceleration

Examples:
  # Full setup (container + build)
  $0

  # Setup with custom build variant
  $0 -b zdnn

  # Skip container setup (if already running)
  $0 --skip-container

  # Only install dependencies
  $0 --skip-container --skip-clone --skip-build

EOF
}

# Check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

################################################################################
# Container Setup Functions
################################################################################

# Setup z-spyre-runtimes container
setup_container() {
    print_info "Setting up z-spyre-runtimes container environment..."
    
    # Clone z-spyre-runtimes repository
    if [ -d "$ZSPYRE_DIR" ]; then
        print_warning "Directory $ZSPYRE_DIR already exists, using existing clone"
    else
        print_info "Cloning z-spyre-runtimes repository..."
        git clone "$ZSPYRE_REPO_URL" "$ZSPYRE_DIR"
        print_success "Repository cloned to $ZSPYRE_DIR"
    fi
    
    # Navigate to runtime-container directory
    cd "$ZSPYRE_DIR/runtime-container"
    
    # Prompt for IBM Cloud API key
    print_info "IBM Cloud API key is required for container registry access"
    read -p "Enter your IBM Cloud API key: " -s ICR_TOKEN
    echo
    
    if [ -z "$ICR_TOKEN" ]; then
        print_error "IBM Cloud API key cannot be empty"
        exit 1
    fi
    
    export ICR_TOKEN
    print_success "IBM Cloud API key set"
    
    # Run make commands
    print_info "Logging into IBM Container Registry..."
    make login
    
    print_info "Pulling container image..."
    make pull
    
    print_info "Starting container..."
    make run
    
    print_success "Container setup complete"
    print_warning "Note: The following steps should be run inside the container"
}

################################################################################
# Dependency Installation Functions
################################################################################

# Install Go programming language
install_go() {
    if command_exists go; then
        GO_VERSION=$(go version | awk '{print $3}')
        print_info "Go is already installed: $GO_VERSION"
        return 0
    fi
    
    print_info "Installing Go to user directory..."
    
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
    
    # Download and install Go to user's home directory
    GO_INSTALL_DIR="$HOME/.local/go"
    cd /tmp
    wget -q "$GO_URL"
    rm -rf "$GO_INSTALL_DIR"
    mkdir -p "$HOME/.local"
    tar -C "$HOME/.local" -xzf "$GO_TARBALL"
    rm "$GO_TARBALL"
    
    # Add Go to PATH if not already present
    if ! grep -q "$GO_INSTALL_DIR/bin" ~/.bashrc; then
        echo "export PATH=\$PATH:$GO_INSTALL_DIR/bin" >> ~/.bashrc
        echo 'export PATH=$PATH:$HOME/go/bin' >> ~/.bashrc
    fi
    
    export PATH=$PATH:$GO_INSTALL_DIR/bin
    export PATH=$PATH:$HOME/go/bin
    
    print_success "Go installed successfully: $(go version)"
}

# Install dependencies
install_dependencies() {
    print_info "Checking dependencies..."
    
    # Note: System packages must be installed by system administrator
    print_warning "PREREQUISITE: The following system packages must be installed:"
    print_warning "  - build-essential (or equivalent build tools)"
    print_warning "  - git, curl, wget, ca-certificates"
    print_warning "  - cmake (version 3.24 or higher recommended)"
    print_warning "  - ninja-build"
    print_warning "  - gcc, g++"
    print_warning "  - python3, python3-pip"
    print_warning "  - libopenblas-dev, liblapack-dev, liblapacke-dev"
    print_warning ""
    print_warning "On Debian/Ubuntu systems, install with:"
    print_warning "  sudo apt update && sudo apt install -y build-essential git curl wget ca-certificates cmake ninja-build gcc g++ python3 python3-pip libopenblas-dev liblapack-dev liblapacke-dev"
    print_warning ""
    print_warning "On RHEL/Fedora systems, install with:"
    print_warning "  sudo dnf install -y gcc gcc-c++ git curl wget ca-certificates cmake ninja-build python3 python3-pip openblas-devel lapack-devel"
    print_warning ""
    
    # Install Go (user-local, no sudo required)
    install_go
    
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
        print_error "Please install the required system packages listed above and run this script again."
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
    
    if [ -d "$OLLAMA_DIR" ]; then
        print_warning "Directory $OLLAMA_DIR already exists"
        print_info "Removing existing directory and re-cloning..."
        rm -rf "$OLLAMA_DIR"
    fi
    
    # Create parent directory if needed
    mkdir -p "$(dirname "$OLLAMA_DIR")"
    
    # Clone repository
    git clone "$OLLAMA_REPO_URL" "$OLLAMA_DIR"
    
    print_success "Repository cloned to $OLLAMA_DIR"
}

################################################################################
# Configuration Functions
################################################################################

# Create .env.s390x configuration file
create_env_config() {
    print_info "Creating .env.s390x configuration file..."
    
    local ENV_FILE="$OLLAMA_DIR/.env.s390x"
    
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
    
    cd "$OLLAMA_DIR"
    
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
    cmake --build build --parallel 8
    
    print_success "Build completed successfully!"
    print_info "Binary location: $OLLAMA_DIR/ollama"
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
            -b|--build)
                BUILD_VARIANT="$2"
                shift 2
                ;;
            --skip-container)
                SKIP_CONTAINER=true
                shift
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
    print_info "Build variant: $BUILD_VARIANT"
    echo
    
    # Setup container
    if [ "$SKIP_CONTAINER" = false ]; then
        setup_container
        echo
        print_warning "Container is now running. Execute the following steps inside the container:"
        print_info "Run this script again with --skip-container flag inside the container"
        exit 0
    else
        print_info "Skipping container setup (assuming already in container)"
    fi
    
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
    echo "     source $OLLAMA_DIR/.env.s390x"
    echo
    echo "  2. Run ollama:"
    echo "     $OLLAMA_DIR/ollama serve"
    echo
    echo "  3. To rebuild with a different variant:"
    echo "     $0 -b <variant> --skip-container --skip-deps --skip-clone"
    echo
}

# Entry point
parse_arguments "$@"
main

# Made with Bob
