#!/usr/bin/env bash
################################################################################
# bootstrap_dev_env.sh
#
# Purpose: Bootstrap a professional Ollama Development Workspace
#          using Podman Compose. Designed for productive daily development,
#          especially for architecture-specific work (e.g. s390x).
#
# Features:
#   - Persistent source code volume (edit on host, build inside container)
#   - Separate dev container (ollama-dev) + runtime container (ollama)
#   - JupyterLab with Ollama Python client pre-installed
#   - Persistent model storage
#   - Clean project structure
#   - Support for custom forks via REPO_URL
#
# Usage:
#   ./bootstrap_dev_env.sh
#   REPO_URL=https://github.com/yourfork/ollama-s390x.git ./bootstrap_dev_env.sh
#
################################################################################
set -euo pipefail
shopt -s nullglob

################################################################################
# Configuration
################################################################################
LOG_DIR="$HOME/.ollama-dev/logs"
RESULTS_DIR="$HOME/.ollama-dev/results"
LOG_FILE="$LOG_DIR/run-$(date +%Y%m%d-%H%M%S).log"
COMPOSE_FILE="compose.yml"

# Repository configuration
REPO_URL="${REPO_URL:-https://github.com/Brice12347/ollama-s390x.git}"
REPO_DIR="ollama-src"
WORKSPACE_DIR="ollama-s390x"   # Directory name inside the container

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

################################################################################
# Logging
################################################################################
setup_logging() {
    mkdir -p "$(dirname "$LOG_FILE")" "$RESULTS_DIR"
    exec > >(tee -a "$LOG_FILE")
    exec 2>&1
    log_info "Logging initialized: $LOG_FILE"
}

log_info()    { echo -e "${BLUE}[INFO $(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS $(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING $(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR $(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" >&2; }

################################################################################
# Workspace Setup
################################################################################
setup_workspace() {
    log_info "Setting up workspace structure..."

    mkdir -p notebooks
    mkdir -p ollama-models
    mkdir -p "$REPO_DIR"

    # Clone repository on first run if directory is empty
    if [ ! -d "$REPO_DIR/.git" ]; then
        log_info "Cloning repository into ./$REPO_DIR ..."
        if git clone "$REPO_URL" "$REPO_DIR"; then
            log_success "Repository cloned successfully into ./$REPO_DIR"
        else
            log_error "Failed to clone repository. You can clone it manually later."
        fi
    else
        log_info "Repository already exists in ./$REPO_DIR (skipping clone)"
    fi
}

################################################################################
# Compose File Generation
################################################################################
create_compose_file() {
    log_info "Generating ${COMPOSE_FILE}..."

    ARCH=$(uname -m)

    if [[ "$ARCH" == "s390x" ]]; then
        log_warning "Detected s390x architecture. The official ollama/ollama image is not available for s390x."
        log_info "Generating compose file without the official 'ollama' service (only ollama-dev for building)."

        cat > "${COMPOSE_FILE}" << 'COMPOSE_EOF'
version: "3.8"

services:
  # JupyterLab with Ollama Python client
  jupyter:
    container_name: jupyter
    image: docker.io/library/python:3.12-slim
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/home/jovyan/work:Z
    working_dir: /home/jovyan/work
    networks:
      - ollama-net
    command:
      - sh
      - -c
      - |
        apt-get update -qq &&
        apt-get install -y -qq build-essential python3-dev curl &&
        pip install --quiet --no-cache-dir jupyterlab requests ollama &&
        jupyter lab \
          --ip=0.0.0.0 \
          --port=8888 \
          --no-browser \
          --ServerApp.token='' \
          --ServerApp.password='' \
          --ServerApp.allow_root=True
    restart: unless-stopped

  # Development container for building Ollama from source (main service on s390x)
  ollama-dev:
    container_name: ollama-dev
    image: docker.io/library/debian:bookworm
    networks:
      - ollama-net
    volumes:
      - ./ollama-src:/workspace/ollama-s390x:Z
      - ./ollama-models:/root/.ollama:Z
    working_dir: /workspace/ollama-s390x
    command:
      - sh
      - -c
      - |
        apt-get update && apt-get install -y \
          curl \
          git \
          vim \
          htop \
          ca-certificates \
          wget \
          tar && \
        wget https://go.dev/dl/go1.22.5.linux-s390x.tar.gz -O /tmp/go.tar.gz && \
        rm -rf /usr/local/go && \
        tar -C /usr/local -xzf /tmp/go.tar.gz && \
        rm /tmp/go.tar.gz && \
        export PATH=$PATH:/usr/local/go/bin && \
        echo 'export PATH=$PATH:/usr/local/go/bin' >> /root/.bashrc && \
        go version && \
        sleep infinity
    restart: unless-stopped

networks:
  ollama-net:
    driver: bridge
COMPOSE_EOF

    else
        # Non-s390x: include official ollama runtime service
        cat > "${COMPOSE_FILE}" << 'COMPOSE_EOF'
version: "3.8"

services:
  # JupyterLab with Ollama Python client
  jupyter:
    container_name: jupyter
    image: docker.io/library/python:3.12-slim
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/home/jovyan/work:Z
    working_dir: /home/jovyan/work
    networks:
      - ollama-net
    command:
      - sh
      - -c
      - |
        apt-get update -qq &&
        apt-get install -y -qq build-essential python3-dev curl &&
        pip install --quiet --no-cache-dir jupyterlab requests ollama &&
        jupyter lab \
          --ip=0.0.0.0 \
          --port=8888 \
          --no-browser \
          --ServerApp.token='' \
          --ServerApp.password='' \
          --ServerApp.allow_root=True
    restart: unless-stopped

  # Development container for building Ollama from source
  ollama-dev:
    container_name: ollama-dev
    image: docker.io/library/debian:bookworm
    networks:
      - ollama-net
    volumes:
      - ./ollama-src:/workspace/ollama-s390x:Z
      - ./ollama-models:/root/.ollama:Z
    working_dir: /workspace/ollama-s390x
    command: ["sleep", "infinity"]
    command:
      - sh
      - -c
      - |
        apt-get update && apt-get install -y \
          curl \
          git \
          vim \
          htop \
          ca-certificates \
          wget \
          tar && \
        wget https://go.dev/dl/go1.22.5.linux-amd64.tar.gz -O /tmp/go.tar.gz && \
        rm -rf /usr/local/go && \
        tar -C /usr/local -xzf /tmp/go.tar.gz && \
        rm /tmp/go.tar.gz && \
        export PATH=$PATH:/usr/local/go/bin && \
        echo 'export PATH=$PATH:/usr/local/go/bin' >> /root/.bashrc && \
        go version && \
    restart: unless-stopped

  # Official Ollama runtime (for testing models) - not available on s390x
  ollama:
    container_name: ollama
    image: docker.io/ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ./ollama-models:/root/.ollama:Z
    networks:
      - ollama-net
    restart: unless-stopped

networks:
  ollama-net:
    driver: bridge
COMPOSE_EOF
    fi

    log_success "${COMPOSE_FILE} generated successfully (architecture: $ARCH)"
}

start_stack() {
    log_info "Starting Ollama Development Workspace..."

    # Try native podman compose first, fall back gracefully
    if podman compose up -d --remove-orphans 2>&1; then
        log_success "Development stack started successfully!"
    else
        log_warning "podman compose had issues. Trying with podman-compose provider..."
        if command -v podman-compose >/dev/null 2>&1; then
            podman-compose up -d --remove-orphans
        else
            log_error "Failed to start the stack. Please check podman compose installation."
            return 1
        fi
    fi
}

################################################################################
# Final Instructions
################################################################################
print_instructions() {
    ARCH=$(uname -m)

    cat << EOF
${GREEN}
╔══════════════════════════════════════════════════════════════════════════════╗
║           Ollama Development Workspace Ready                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
${NC}

${CYAN}Workspace Structure:${NC}
  ./notebooks/           → Jupyter notebooks
  ./ollama-src/          → Your source code (mounted into ollama-dev)
  ./ollama-models/       → Persistent model storage

${CYAN}Access JupyterLab:${NC}
  http://localhost:8888

${CYAN}Enter Development Container (for building):${NC}
  podman compose exec ollama-dev bash
  cd /workspace/ollama-s390x
  cmake -B build . && cmake --build build --parallel \$(nproc)

EOF

    if [[ "$ARCH" != "s390x" ]]; then
        cat << EOF
${CYAN}Use Official Ollama (for testing models):${NC}
  podman compose exec ollama ollama list
  podman compose exec ollama ollama run llama3.2

EOF
    fi

    cat << EOF
${CYAN}Useful Commands:${NC}
  podman compose logs -f          # Follow all logs
  podman compose ps               # Container status
  podman compose restart          # Restart services
  podman compose down             # Stop everything

${YELLOW}Note:${NC} This workspace defaults to: ${REPO_URL}
You can override it with: REPO_URL=... ./bootstrap_dev_env.sh

Log file: ${LOG_FILE}
EOF
}

################################################################################
# Dependency Installation
################################################################################
install_curl() {
    if ! command -v curl >/dev/null 2>&1; then
        log_info "curl not found. Installing via dnf..."
        if command -v dnf >/dev/null 2>&1; then
            dnf install -y curl
            log_success "curl installed successfully"
        else
            log_error "dnf is not available. Please install curl manually."
            exit 1
        fi
    else
        log_info "curl is already installed"
    fi
}

install_podman_compose() {
    if ! command -v podman-compose >/dev/null 2>&1; then
        log_info "podman-compose not found. Installing via pip3..."
        if command -v pip3 >/dev/null 2>&1; then
            pip3 install podman-compose
            log_success "podman-compose installed successfully"
        else
            log_error "pip3 is not installed. Please install Python 3 and pip3 first."
            exit 1
        fi
    else
        log_info "podman-compose is already installed"
    fi
}

################################################################################
# Main
################################################################################
main() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}   Ollama Development Workspace Bootstrap${NC}"
    echo -e "${BLUE}========================================${NC}"

    if ! command -v podman >/dev/null 2>&1; then
        log_error "podman is not installed. Please install it first."
        exit 1
    fi

    setup_logging
    install_curl
    install_podman_compose
    setup_workspace
    create_compose_file
    start_stack
    print_instructions

    log_success "Bootstrap complete. Happy developing!"
}

main
