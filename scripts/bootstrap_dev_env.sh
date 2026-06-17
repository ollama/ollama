#!/usr/bin/env bash

################################################################################
# bootstrap_dev_env.sh
#
# Purpose: Simple script to clone and build ollama from source
#
# Usage: ./scripts/bootstrap_dev_env.sh
#
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

################################################################################
# Colors for output
################################################################################

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

################################################################################
# Helper Functions
################################################################################

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

################################################################################
# Main Setup
################################################################################

main() {
    print_info "Starting ollama setup..."
    
    # Get current working directory and set absolute path for repo
    WORK_DIR=$(pwd)
    REPO_DIR="$WORK_DIR/ollama-s390x"
    
    # Clone the repository
    print_info "Cloning ollama-s390x repository..."
    if [ -d "$REPO_DIR" ]; then
        print_warning "Directory $REPO_DIR already exists, removing it..."
        rm -rf "$REPO_DIR"
    fi
    
    git clone https://github.com/Brice12347/ollama-s390x.git "$REPO_DIR"
    print_success "Repository cloned to $REPO_DIR"
    
    # Install dependencies
    print_info "Installing dependencies..."
    dnf install -y golang cmake ninja-build
    print_success "Dependencies installed"
    
    # Change to repo directory using absolute path
    cd "$REPO_DIR"
    print_info "Changed to $REPO_DIR"
    
    # # Configure with CMake
    # print_info "Configuring build with CMake..."
    # cmake -B build .
    # print_success "CMake configuration complete"
    
    # # Build
    # print_info "Building ollama (this may take a while)..."
    # cmake --build build --parallel 8
    # print_success "Build complete"
    
    # # Start ollama server
    # print_info "Starting ollama server..."
    # ./ollama serve
}

# Run main function
main

# Made with Bob
