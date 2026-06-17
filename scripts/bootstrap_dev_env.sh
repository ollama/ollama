#!/usr/bin/env bash

################################################################################
# bootstrap_dev_env.sh
#
# Purpose: Bootstrap Ollama development environment with containerized setup
#          using Podman/Docker Compose for s390x architecture
#
# Usage: ./scripts/bootstrap_dev_env.sh
#
# Requirements:
#   - podman and podman-compose installed
#   - Root or sudo access for logging directory creation
#   - Network connectivity for pulling base images
#
################################################################################

set -euo pipefail
shopt -s nullglob

################################################################################
# Configuration
################################################################################

LOG_FILE="/var/log/ollama-bootstrap/run-$(date +%Y%m%d-%H%M%S).log"
RESULTS_DIR="/var/log/ollama-bootstrap/results"
TIMEOUT_SECONDS=180

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
# User Input
################################################################################

prompt_username() {
    log_info "Prompting for username..."
    echo -n "Enter your username: " > /dev/tty
    read -r USERNAME < /dev/tty
    
    if [ -z "$USERNAME" ]; then
        log_error "USERNAME cannot be empty"
        exit 1
    fi
    
    log_success "Username captured: $USERNAME"
    log_info "Notebooks will be mounted at: /Wonder/$USERNAME/notebooks"
}

################################################################################
# Docker Configuration Files
################################################################################

create_ollama_dockerfile() {
    log_info "Creating Ollama Dockerfile..."
    
    cat > Dockerfile.ollama << 'EOF'
# Ollama Development Container - Debian-based
FROM debian:bookworm-slim

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    golang \
    cmake \
    ninja-build \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set Go environment
ENV GOPATH=/go
ENV PATH=$PATH:/usr/local/go/bin:$GOPATH/bin

# Create workspace
WORKDIR /workspace

# Clone ollama repository
RUN git clone https://github.com/Brice12347/ollama-s390x.git /workspace/ollama

# Build ollama
WORKDIR /workspace/ollama
RUN cmake -B build . && \
    cmake --build build --parallel 8

# Expose ollama port
EXPOSE 11434

# Start ollama server
CMD ["./ollama", "serve"]
EOF

    log_success "Ollama Dockerfile created"
}

create_jupyter_dockerfile() {
    log_info "Creating Jupyter Dockerfile..."
    
    cat > Dockerfile.jupyter << 'EOF'
# Jupyter Development Container - Python 3.12
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install JupyterLab and requests
RUN pip install --no-cache-dir \
    jupyterlab \
    requests

# Create jovyan user for Jupyter
RUN useradd -m -s /bin/bash jovyan

# Create work directory
RUN mkdir -p /home/jovyan/work && \
    chown -R jovyan:jovyan /home/jovyan

# Switch to jovyan user
USER jovyan
WORKDIR /home/jovyan

# Expose Jupyter port
EXPOSE 8888

# Start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
EOF

    log_success "Jupyter Dockerfile created"
}

create_docker_compose() {
    log_info "Creating docker-compose.yml..."
    
    cat > docker-compose.yml << EOF
version: '3.8'

services:
  ollama:
    container_name: ollama
    build:
      context: .
      dockerfile: Dockerfile.ollama
    ports:
      - "11434:11434"
    networks:
      - ollama-network
    restart: unless-stopped

  jupyter:
    container_name: jupyter
    build:
      context: .
      dockerfile: Dockerfile.jupyter
    ports:
      - "8877:8888"
    volumes:
      - /Wonder/$USERNAME/notebooks:/home/jovyan/work:Z
    networks:
      - ollama-network
    restart: unless-stopped
    depends_on:
      - ollama

networks:
  ollama-network:
    driver: bridge
EOF

    log_success "docker-compose.yml created"
}

################################################################################
# Container Management
################################################################################

build_and_start_containers() {
    log_info "Building and starting containers with podman-compose..."
    
    # Check if podman-compose is available
    if ! command -v podman-compose &> /dev/null; then
        log_error "podman-compose not found. Please install it first."
        exit 1
    fi
    
    # Build containers
    log_info "Building container images (this may take several minutes)..."
    podman-compose build --no-cache
    
    # Start containers
    log_info "Starting containers..."
    podman-compose up -d
    
    log_success "Containers started successfully"
}

wait_for_containers() {
    log_info "Waiting for containers to be ready (timeout: ${TIMEOUT_SECONDS}s)..."
    
    local elapsed=0
    local interval=5
    
    while [ $elapsed -lt $TIMEOUT_SECONDS ]; do
        if podman ps | grep -q "ollama" && podman ps | grep -q "jupyter"; then
            log_success "Containers are running"
            return 0
        fi
        
        sleep $interval
        elapsed=$((elapsed + interval))
        log_info "Waiting... (${elapsed}s/${TIMEOUT_SECONDS}s)"
    done
    
    log_error "Timeout waiting for containers to start"
    return 1
}

verify_containers() {
    log_info "Verifying container status..."
    
    # Show Jupyter logs
    log_info "Jupyter container logs (last 10 lines):"
    podman logs jupyter --tail 10
    
    # Test Jupyter endpoint
    log_info "Testing Jupyter endpoint..."
    if curl -I http://127.0.0.1:8888 2>/dev/null | head -n 1; then
        log_success "Jupyter is responding"
    else
        log_warning "Jupyter endpoint test failed (may need more time to start)"
    fi
    
    # Test Ollama endpoint
    log_info "Testing Ollama endpoint..."
    if curl -I http://127.0.0.1:11434 2>/dev/null | head -n 1; then
        log_success "Ollama is responding"
    else
        log_warning "Ollama endpoint test failed (may need more time to start)"
    fi
}

################################################################################
# Final Instructions
################################################################################

print_final_instructions() {
    local hostname=$(hostname -f 2>/dev/null || hostname)
    
    cat << EOF

${GREEN}*****************************************************************
* Ollama Development Environment Ready!                         *
*                                                               *
* Access Ollama container:                                      *
* $ podman exec -it ollama bash                                 *
*                                                               *
* Access Jupyter from your laptop:                              *
* $ ssh -L 8877:127.0.0.1:8877 ${USERNAME}@${hostname} -p 22
*                                                               *
* Then open: http://localhost:8877                              *
*                                                               *
* Container Management:                                         *
* - View logs: podman logs <container-name>                     *
* - Stop all: podman-compose down                               *
* - Restart: podman-compose restart                             *
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
    
    # Stop and remove containers
    if command -v podman-compose &> /dev/null; then
        podman-compose down 2>/dev/null || true
    fi
    
    # Remove generated files
    rm -f Dockerfile.ollama Dockerfile.jupyter docker-compose.yml
    
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
    
    # Get username
    prompt_username
    
    # Create notebooks directory
    log_info "Creating notebooks directory..."
    mkdir -p "/Wonder/$USERNAME/notebooks"
    log_success "Notebooks directory created: /Wonder/$USERNAME/notebooks"
    
    # Create Docker configuration files
    create_ollama_dockerfile
    create_jupyter_dockerfile
    create_docker_compose
    
    # Build and start containers
    build_and_start_containers
    
    # Wait for containers to be ready
    wait_for_containers
    
    # Verify containers are working
    verify_containers
    
    # Save results
    log_info "Saving results..."
    cat > "$RESULTS_DIR/bootstrap-$(date +%Y%m%d-%H%M%S).txt" << EOF
Bootstrap completed successfully
Username: $USERNAME
Timestamp: $(date)
Log file: $LOG_FILE
Containers: ollama, jupyter
EOF
    
    # Print final instructions
    print_final_instructions
    
    log_success "Bootstrap complete!"
}

# Run main function
main

# Made with Bob
