#!/bin/bash
# ============================================================================
# OLLAMA LOCAL DEVELOPMENT STOP SCRIPT
# ============================================================================
# This script stops and removes the local development stack
# Usage: ./scripts/local-stop.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Get project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
COMPOSE_FILE="${PROJECT_ROOT}/docker/docker-compose.local.yml"

compose_cmd() {
    if command -v docker-compose &> /dev/null; then
        echo docker-compose
        return
    fi

    if docker compose version &> /dev/null; then
        echo docker compose
        return
    fi

    echo "Docker Compose is not installed"
    exit 1
}

cd "$PROJECT_ROOT"

DOCKER_COMPOSE="$(compose_cmd)"

echo ""
log_info "Stopping local development stack..."
echo ""

# Check if stopping with cleanup
if [ "$1" == "--cleanup" ] || [ "$1" == "-c" ]; then
    log_warning "Removing all containers, networks, and volumes (DATA WILL BE DELETED!)"
    read -p "Are you sure? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Removing containers, networks, and volumes..."
        $DOCKER_COMPOSE -f "$COMPOSE_FILE" down -v
        log_success "All services stopped and volumes removed"
    else
        log_info "Cleanup cancelled"
        exit 0
    fi
else
    log_info "Stopping containers (data will be preserved)..."
    $DOCKER_COMPOSE -f "$COMPOSE_FILE" stop
    log_success "All services stopped"
fi

echo ""
log_success "Local development stack is stopped"
echo ""
