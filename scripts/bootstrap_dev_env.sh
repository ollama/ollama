#!/usr/bin/env bash

################################################################################
# bootstrap_dev_env.sh
#
# Purpose: Automate the setup of containerized s390x development environments
#          for building and serving ollama and for running JupyterLab.
#
# Usage: ./scripts/bootstrap_dev_env.sh [OPTIONS]
#
# Options:
#   -h, --help              Show this help message
#   -b, --build VARIANT     Build variant to compile (default: cpu)
#                           Options: cpu, cpu-no-vxe, cpu-debug, cpu-static, zdnn
#   --skip-container        Skip ollama container setup
#   --skip-deps             Skip dependency installation inside ollama container
#   --skip-clone            Skip repository cloning inside ollama container
#   --skip-build            Skip building ollama inside ollama container
#
################################################################################

set -e
set -u
set -o pipefail

################################################################################
# Configuration Variables
################################################################################

OLLAMA_REPO_URL="git@github.com:Brice12347/ollama-s390x.git"
OLLAMA_REPO_DIR="/workspace/ollama-s390x"
OLLAMA_CONTAINER_NAME="ollama-dev"
OLLAMA_IMAGE_TAG="ollama-s390x-dev:latest"
OLLAMA_DOCKERFILE_PATH="/tmp/ollama-dev.Dockerfile"

JUPYTER_CONTAINER_NAME="jupyter"
JUPYTER_IMAGE_TAG="ollama-s390x-jupyter:latest"
JUPYTER_DOCKERFILE_PATH="/tmp/jupyter-dev.Dockerfile"

BUILD_VARIANT="${OLLAMA_BUILD_VARIANT:-cpu}"
SKIP_CONTAINER=false
SKIP_DEPS=false
SKIP_CLONE=false
SKIP_BUILD=false

CONTAINER_ENGINE=""
USERNAME_INPUT=""
NOTEBOOKS_DIR=""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Automate the setup of containerized development environments for building and
serving ollama and for running JupyterLab.

Options:
  -h, --help              Show this help message
  -b, --build VARIANT     Build variant to compile (default: cpu)
                          Options: cpu, cpu-no-vxe, cpu-debug, cpu-static, zdnn
  --skip-container        Skip ollama container setup
  --skip-deps             Skip dependency installation inside ollama container
  --skip-clone            Skip repository cloning inside ollama container
  --skip-build            Skip building ollama inside ollama container

Build Variants:
  cpu                     Standard CPU build (default)
  cpu-no-vxe              CPU build without Vector Extensions
  cpu-debug               CPU build with debug symbols
  cpu-static              Static CPU build
  zdnn                    Build with IBM zDNN acceleration

Examples:
  # Full setup for ollama and jupyter containers
  $0

  # Setup with custom build variant
  $0 -b zdnn

  # Recreate jupyter only while skipping ollama setup steps
  $0 --skip-container --skip-deps --skip-clone --skip-build

EOF
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

cleanup_file() {
    local file_path=$1
    if [ -f "$file_path" ]; then
        rm -f "$file_path"
    fi
}

################################################################################
# Validation and Argument Parsing
################################################################################

validate_build_variant() {
    case "$BUILD_VARIANT" in
        cpu|cpu-no-vxe|cpu-debug|cpu-static|zdnn)
            ;;
        *)
            print_error "Unknown build variant: $BUILD_VARIANT"
            print_info "Valid variants: cpu, cpu-no-vxe, cpu-debug, cpu-static, zdnn"
            exit 1
            ;;
    esac
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -b|--build)
                if [[ $# -lt 2 ]]; then
                    print_error "Missing value for $1"
                    exit 1
                fi
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

################################################################################
# Container Engine Functions
################################################################################

detect_container_engine() {
    if command_exists podman; then
        CONTAINER_ENGINE="podman"
    elif command_exists docker; then
        CONTAINER_ENGINE="docker"
    else
        print_error "Neither podman nor docker is installed"
        exit 1
    fi

    print_success "Using container engine: $CONTAINER_ENGINE"
}

cleanup_existing_artifacts() {
    print_info "Cleaning up existing containers and temporary Dockerfiles..."

    "$CONTAINER_ENGINE" rm -f "$OLLAMA_CONTAINER_NAME" >/dev/null 2>&1 || true
    "$CONTAINER_ENGINE" rm -f "$JUPYTER_CONTAINER_NAME" >/dev/null 2>&1 || true
    "$CONTAINER_ENGINE" rmi -f "$OLLAMA_IMAGE_TAG" >/dev/null 2>&1 || true
    "$CONTAINER_ENGINE" rmi -f "$JUPYTER_IMAGE_TAG" >/dev/null 2>&1 || true

    cleanup_file "$OLLAMA_DOCKERFILE_PATH"
    cleanup_file "$JUPYTER_DOCKERFILE_PATH"

    print_success "Cleanup complete"
}

################################################################################
# Ollama Container Functions
################################################################################

create_ollama_dockerfile() {
    print_info "Creating ollama Dockerfile at $OLLAMA_DOCKERFILE_PATH"

    cat > "$OLLAMA_DOCKERFILE_PATH" << EOF
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y git openssh-client ca-certificates curl golang-go cmake ninja-build && \
    rm -rf /var/lib/apt/lists/*

CMD ["bash", "-lc", "sleep infinity"]
EOF

    print_success "Ollama Dockerfile created"
}

build_ollama_image() {
    print_info "Building ollama image: $OLLAMA_IMAGE_TAG"
    "$CONTAINER_ENGINE" build -t "$OLLAMA_IMAGE_TAG" -f "$OLLAMA_DOCKERFILE_PATH" .
    print_success "Ollama image built"
}

run_ollama_container() {
    print_info "Starting ollama container: $OLLAMA_CONTAINER_NAME"
    "$CONTAINER_ENGINE" run -d \
        --name "$OLLAMA_CONTAINER_NAME" \
        --network host \
        "$OLLAMA_IMAGE_TAG"
    print_success "Ollama container started"
}

run_ollama_clone() {
    if [ "$SKIP_CLONE" = true ]; then
        print_warning "Skipping repository clone inside ollama container"
        return
    fi

    print_info "Cloning ollama repository inside ollama container"
    "$CONTAINER_ENGINE" exec "$OLLAMA_CONTAINER_NAME" bash -lc "
        rm -rf '$OLLAMA_REPO_DIR' &&
        git clone '$OLLAMA_REPO_URL' '$OLLAMA_REPO_DIR'
    "
    print_success "Repository cloned inside ollama container"
}

run_ollama_dependency_step() {
    if [ "$SKIP_DEPS" = true ]; then
        print_warning "Skipping dependency installation inside ollama container"
        return
    fi

    print_info "Installing ollama dependencies inside container"
    "$CONTAINER_ENGINE" exec "$OLLAMA_CONTAINER_NAME" bash -lc "
        apt-get update &&
        apt-get install -y golang-go cmake ninja-build
    "
    print_success "Dependency installation completed inside ollama container"
}

run_ollama_build() {
    if [ "$SKIP_BUILD" = true ]; then
        print_warning "Skipping ollama build inside container"
        return
    fi

    print_info "Configuring and building ollama inside container"
    "$CONTAINER_ENGINE" exec "$OLLAMA_CONTAINER_NAME" bash -lc "
        cd '$OLLAMA_REPO_DIR' &&
        cmake -B build . &&
        cmake --build build --parallel 8
    "
    print_success "Ollama build completed inside container"
}

start_ollama_service() {
    print_info "Starting ollama service inside container"
    "$CONTAINER_ENGINE" exec -d "$OLLAMA_CONTAINER_NAME" bash -lc "
        cd '$OLLAMA_REPO_DIR' &&
        ./ollama serve
    "
    print_success "Ollama service started inside container"
}

setup_ollama_container() {
    if [ "$SKIP_CONTAINER" = true ]; then
        print_warning "Skipping ollama container setup"
        return
    fi

    create_ollama_dockerfile
    build_ollama_image
    run_ollama_container
    run_ollama_clone
    run_ollama_dependency_step
    run_ollama_build
    start_ollama_service
}

################################################################################
# Jupyter Container Functions
################################################################################

prompt_for_username() {
    local prompt_message=$1

    read -r -p "$prompt_message" USERNAME_INPUT

    if [ -z "$USERNAME_INPUT" ]; then
        print_error "USERNAME cannot be empty"
        exit 1
    fi

    NOTEBOOKS_DIR="/Wonder/$USERNAME_INPUT/notebooks"
}

create_jupyter_dockerfile() {
    print_info "Creating jupyter Dockerfile at $JUPYTER_DOCKERFILE_PATH"

    cat > "$JUPYTER_DOCKERFILE_PATH" << EOF
FROM docker.io/library/python:3.12-slim

WORKDIR /home/jovyan/work

RUN apt-get update -qq && \
    apt-get install -y -qq build-essential python3-dev && \
    rm -rf /var/lib/apt/lists/*

CMD ["sh", "-c", "sleep infinity"]
EOF

    print_success "Jupyter Dockerfile created"
}

build_jupyter_image() {
    print_info "Building jupyter image: $JUPYTER_IMAGE_TAG"
    "$CONTAINER_ENGINE" build -t "$JUPYTER_IMAGE_TAG" -f "$JUPYTER_DOCKERFILE_PATH" .
    print_success "Jupyter image built"
}

run_jupyter_container() {
    print_info "Starting jupyter container: $JUPYTER_CONTAINER_NAME"
    "$CONTAINER_ENGINE" run -d \
        --name "$JUPYTER_CONTAINER_NAME" \
        --network host \
        -v "$NOTEBOOKS_DIR:/home/jovyan/work:Z" \
        -w /home/jovyan/work \
        "$JUPYTER_IMAGE_TAG" \
        sh -c "
            pip install --quiet --no-cache-dir jupyterlab requests &&
            jupyter lab \
              --ip=127.0.0.1 \
              --port=8888 \
              --no-browser \
              --ServerApp.token='' \
              --ServerApp.password='' \
              --ServerApp.allow_root=True
        "
    print_success "Jupyter container started"
}

show_jupyter_logs() {
    print_info "Showing recent jupyter logs"
    "$CONTAINER_ENGINE" logs "$JUPYTER_CONTAINER_NAME" --tail 10
    print_success "Displayed jupyter logs"
}

wait_for_jupyter() {
    print_info "Waiting 3 minutes for JupyterLab to initialize"
    sleep 180
    print_success "Wait period completed"
}

check_jupyter_endpoint() {
    print_info "Checking JupyterLab endpoint"
    curl -I http://127.0.0.1:8888
    print_success "JupyterLab endpoint responded"
}

setup_jupyter_container() {
    prompt_for_username "Enter USERNAME for Jupyter notebook mount path: "
    create_jupyter_dockerfile
    build_jupyter_image
    run_jupyter_container
    show_jupyter_logs
    wait_for_jupyter
    check_jupyter_endpoint
}

################################################################################
# Output Functions
################################################################################

show_build_matrix() {
    cat << EOF

${BLUE}═══════════════════════════════════════════════════════════════${NC}
${GREEN}Available Build Variants for s390x${NC}
${BLUE}═══════════════════════════════════════════════════════════════${NC}

${YELLOW}cpu${NC}           Standard CPU build (default)
                - Release build inside the ollama container

${YELLOW}cpu-no-vxe${NC}    Reserved compatibility variant
                - Accepted for CLI compatibility
                - Current workflow still runs: cmake -B build .

${YELLOW}cpu-debug${NC}     Reserved debug variant
                - Accepted for CLI compatibility
                - Current workflow still runs: cmake -B build .

${YELLOW}cpu-static${NC}    Reserved static variant
                - Accepted for CLI compatibility
                - Current workflow still runs: cmake -B build .

${YELLOW}zdnn${NC}          Reserved zDNN variant
                - Accepted for CLI compatibility
                - Current workflow still runs: cmake -B build .

${BLUE}═══════════════════════════════════════════════════════════════${NC}

EOF
}

show_final_instructions() {
    cat << EOF

${GREEN}Setup complete!${NC}

${BLUE}Ollama container:${NC}
  Name: $OLLAMA_CONTAINER_NAME
  Image: $OLLAMA_IMAGE_TAG
  Service: running inside the container via ./ollama serve

${BLUE}Jupyter container:${NC}
  Name: $JUPYTER_CONTAINER_NAME
  Image: $JUPYTER_IMAGE_TAG
  Notebook mount: $NOTEBOOKS_DIR
  URL: http://127.0.0.1:8888

${BLUE}Next steps:${NC}
  1. In a separate CLI, run:
     ssh -L 8888:127.0.0.1:8888 ${USERNAME_INPUT}@b39-triframe1.pok.stglabs.ibm.com -p 22

  2. Open JupyterLab in your browser:
     http://127.0.0.1:8888

  3. Inspect running containers if needed:
     $CONTAINER_ENGINE ps

  4. View service logs if needed:
     $CONTAINER_ENGINE logs $OLLAMA_CONTAINER_NAME
     $CONTAINER_ENGINE logs $JUPYTER_CONTAINER_NAME

EOF
}

################################################################################
# Main Script Logic
################################################################################

main() {
    print_info "Starting containerized ollama and jupyter environment setup..."
    print_info "Build variant: $BUILD_VARIANT"
    echo

    validate_build_variant
    detect_container_engine
    cleanup_existing_artifacts
    echo

    show_build_matrix

    setup_ollama_container
    echo

    setup_jupyter_container
    echo

    show_final_instructions
}

parse_arguments "$@"
main

# Made with Bob
