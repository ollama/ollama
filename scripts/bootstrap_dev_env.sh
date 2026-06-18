#!/usr/bin/env bash

################################################################################
# bootstrap_dev_env.sh
#
# Purpose: Bootstrap Ollama development environment with containerized setup
#          using Podman for s390x architecture
#
# Usage: ./scripts/bootstrap_dev_env.sh
#
# Requirements:
#   - podman installed
#   - Network connectivity for pulling base images
#
################################################################################

set -euo pipefail
shopt -s nullglob

################################################################################
# Configuration
################################################################################

LOG_DIR="$HOME/.ollama-bootstrap/logs"
RESULTS_DIR="$HOME/.ollama-bootstrap/results"
LOG_FILE="$LOG_DIR/run-$(date +%Y%m%d-%H%M%S).log"
CONTAINER_NAME="ollama"
IMAGE_NAME="ollama:local"

################################################################################
# Colors for output
################################################################################

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

################################################################################
# Logging Infrastructure
################################################################################

setup_logging() {
    # Create logging directories
    mkdir -p "$(dirname "$LOG_FILE")" "$RESULTS_DIR"
    
    # Redirect all output to log file while keeping terminal output
    exec > >(tee -a "$LOG_FILE")
    exec 2>&1
    
    log_info "Logging initialized: $LOG_FILE"
    log_info "Results directory: $RESULTS_DIR"
}

log_info() {
    echo -e "${BLUE}[INFO $(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS $(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING $(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR $(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" >&2
}

################################################################################
# Container Management
################################################################################

stop_existing_container() {
    log_info "Checking for existing ollama container..."
    
    if podman ps -a --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        log_warning "Found existing ollama container, stopping and removing..."
        podman stop "$CONTAINER_NAME" 2>/dev/null || true
        podman rm "$CONTAINER_NAME" 2>/dev/null || true
        log_success "Existing container removed"
    else
        log_info "No existing container found"
    fi
}

create_dockerfile() {
    log_info "Creating Dockerfile inline..."
    
    cat << 'EOF' > ollama.Dockerfile
# Ollama Development Container
FROM quay.io/podman/stable

# Install build dependencies
RUN dnf install -y \
    git \
    golang \
    cmake \
    ninja-build \
    gcc \
    gcc-c++ \
    make \
    curl \
    ca-certificates \
    && dnf clean all

# Set Go environment
ENV GOPATH=/go
ENV PATH=$PATH:/usr/local/go/bin:$GOPATH/bin

# Clone ollama-s390x repository
RUN git clone https://github.com/Brice12347/ollama-s390x.git /workspace/ollama-s390x

# Set working directory
WORKDIR /workspace/ollama-s390x

# Expose ollama port
EXPOSE 11434

# Default command
CMD ["/bin/bash"]
EOF

    log_success "Dockerfile created: ollama.Dockerfile"
}

build_container_image() {
    log_info "Building container image (this may take several minutes)..."
    
    if podman build -t "$IMAGE_NAME" -f ollama.Dockerfile .; then
        log_success "Container image built successfully: $IMAGE_NAME"
    else
        log_error "Failed to build container image"
        return 1
    fi
}

run_container() {
    log_info "Starting ollama container..."
    
    podman run -d -it \
        --name "$CONTAINER_NAME" \
        --cap-add=sys_admin \
        --cap-add mknod \
        --security-opt seccomp=unconfined \
        --security-opt label=disable \
        -p 11434:11434 \
        "$IMAGE_NAME" \
        bash
    
    if [ $? -eq 0 ]; then
        log_success "Container started successfully: $CONTAINER_NAME"
    else
        log_error "Failed to start container"
        return 1
    fi
}

cleanup_dockerfile() {
    log_info "Cleaning up temporary Dockerfile..."
    
    if [ -f ollama.Dockerfile ]; then
        rm -f ollama.Dockerfile
        log_success "Temporary Dockerfile removed"
    fi
}

verify_container() {
    log_info "Verifying container status..."
    
    if podman ps --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        log_success "Container is running"
        
        # Show container info
        log_info "Container details:"
        podman ps --filter "name=${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    else
        log_error "Container is not running"
        return 1
    fi
}

################################################################################
# Final Instructions
################################################################################

print_final_instructions() {
    cat << EOF

${GREEN}*****************************************************************
* Ollama Development Environment Ready!                         *
*                                                               *
* Access the container:                                         *
* $ podman exec -it ollama bash                                 *
*                                                               *
* Inside the container, you can:                                *
* - Navigate to: /workspace/ollama-s390x                        *
* - Build ollama: cmake -B build . && cmake --build build       *
* - Run ollama: ./ollama serve                                  *
*                                                               *
* Container Management:                                         *
* - View logs: podman logs ollama                               *
* - Stop: podman stop ollama                                    *
* - Start: podman start ollama                                  *
* - Remove: podman rm ollama                                    *
*                                                               *
* Log file: ${LOG_FILE}
* Results: ${RESULTS_DIR}
*****************************************************************${NC}

EOF
}

################################################################################
# Cleanup
################################################################################

cleanup_on_error() {
    log_error "Setup failed. Cleaning up..."
    
    # Stop and remove container
    podman stop "$CONTAINER_NAME" 2>/dev/null || true
    podman rm "$CONTAINER_NAME" 2>/dev/null || true
    
    # Remove temporary Dockerfile
    rm -f ollama.Dockerfile
    
    log_info "Cleanup complete"
}

################################################################################
# Main Execution
################################################################################

main() {
    log_info "=========================================="
    log_info "Ollama Development Environment Bootstrap"
    log_info "=========================================="
    
    # Set up error handling
    trap cleanup_on_error ERR
    
    # Initialize logging
    setup_logging
    
    # Stop and remove any existing container
    stop_existing_container
    
    # Create Dockerfile inline
    create_dockerfile
    
    # Build container image
    build_container_image
    
    # Run container
    run_container
    
    # Clean up temporary Dockerfile
    cleanup_dockerfile
    
    # Verify container is running
    verify_container
    
    # Save results
    log_info "Saving results..."
    cat > "$RESULTS_DIR/bootstrap-$(date +%Y%m%d-%H%M%S).txt" << EOF
Bootstrap completed successfully
Timestamp: $(date)
Log file: $LOG_FILE
Container: $CONTAINER_NAME
Image: $IMAGE_NAME
EOF
    
    # Print final instructions
    print_final_instructions
    
    log_success "Bootstrap complete!"
}

# Run main function
main

# Made with Bob
