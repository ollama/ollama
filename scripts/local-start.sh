#!/bin/bash
# ============================================================================
# OLLAMA LOCAL DEVELOPMENT SETUP SCRIPT
# ============================================================================
# This script initializes and starts the complete local development stack
# Usage: ./scripts/local-start.sh

set -e  # Exit on first error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get directory where script is located
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

    log_error "Docker Compose is not installed"
    exit 1
}

# Functions
log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        echo "  Install from: https://docs.docker.com/get-docker/"
        exit 1
    fi
    log_success "Docker is installed: $(docker --version)"

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed"
        echo "  Install from: https://docs.docker.com/compose/install/"
        exit 1
    fi
    log_success "Docker Compose is installed"

    # Check Docker daemon
    if ! docker ps &> /dev/null; then
        log_error "Docker daemon is not running"
        echo "  Start Docker Desktop or daemon"
        exit 1
    fi
    log_success "Docker daemon is running"
}

# Setup environment
setup_environment() {
    log_info "Setting up environment..."

    if [ ! -f "$PROJECT_ROOT/.env.local" ]; then
        log_warning ".env.local not found, creating from template..."
        if [ -f "$PROJECT_ROOT/.env.example" ]; then
            cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env.local"
            log_success "Created .env.local from .env.example"
        else
            log_warning ".env.example not found; skipping .env.local creation"
        fi
    else
        log_success ".env.local exists"
    fi
}

# Start services
start_services() {
    log_info "Starting all services..."
    log_warning "This may take 30-60 seconds on first run..."

    cd "$PROJECT_ROOT"

    DOCKER_COMPOSE="$(compose_cmd)"

    # Build images
    log_info "Building Docker images..."
    $DOCKER_COMPOSE -f "$COMPOSE_FILE" build --no-cache

    # Start services
    log_info "Starting containers..."
    $DOCKER_COMPOSE -f "$COMPOSE_FILE" up -d

    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."

    local max_attempts=60
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        if $DOCKER_COMPOSE -f "$COMPOSE_FILE" ps | grep -q "healthy"; then
            log_success "Services are healthy"
            break
        fi

        attempt=$((attempt + 1))
        echo -ne "\r  Attempt $attempt/$max_attempts..."
        sleep 1
    done

    if [ $attempt -eq $max_attempts ]; then
        log_warning "Services took longer than expected"
        log_info "Checking service status..."
        $DOCKER_COMPOSE -f "$COMPOSE_FILE" ps
    fi
}

# Verify services
verify_services() {
    log_info "Verifying services..."

    # Check API
    if curl -s http://127.0.0.1:8000/health &> /dev/null; then
        log_success "API is healthy"
    else
        log_warning "API is not responding yet"
    fi

    # Check Database
    if command -v psql &> /dev/null; then
        if psql -h 127.0.0.1 -U ollama -d ollama -c "SELECT 1" &> /dev/null; then
            log_success "PostgreSQL is accessible"
        else
            log_warning "PostgreSQL connection failed"
        fi
    fi

    # Check Redis
    if command -v redis-cli &> /dev/null; then
        if redis-cli -h 127.0.0.1 ping &> /dev/null; then
            log_success "Redis is accessible"
        else
            log_warning "Redis connection failed"
        fi
    fi

    # Check Ollama
    if curl -s http://127.0.0.1:11434/api/tags &> /dev/null; then
        log_success "Ollama is accessible"
    else
        log_warning "Ollama is not responding yet"
    fi

    # Check Qdrant
    if curl -s http://127.0.0.1:6333/health &> /dev/null; then
        log_success "Qdrant is accessible"
    else
        log_warning "Qdrant is not responding yet"
    fi
}

# Display status
display_status() {
    echo ""
    log_success "Local development stack is running!"
    echo ""
    echo "Service URLs:"
    echo "  API Server:       http://127.0.0.1:8000"
    echo "  API Docs:         http://127.0.0.1:8000/docs"
    echo "  PostgreSQL:       postgresql://ollama:ollama-dev-password@127.0.0.1:5432/ollama"
    echo "  Redis:            redis://127.0.0.1:6379"
    echo "  Ollama:           http://127.0.0.1:11434"
    echo "  Qdrant:           http://127.0.0.1:6333"
    echo "  Prometheus:       http://127.0.0.1:9090"
    echo "  Grafana:          http://127.0.0.1:3000 (admin/admin)"
    echo ""
    echo "Next steps:"
    echo "  1. Pull a model:  docker exec ollama-engine-local ollama pull llama2"
    echo "  2. Run tests:     pytest tests/ -v"
    echo "  3. Check health:  curl http://127.0.0.1:8000/health"
    echo ""
    echo "View logs:"
    echo "  $(compose_cmd) -f ${COMPOSE_FILE} logs -f"
    echo ""
}

# Main
main() {
    echo ""
    log_info "Ollama Local Development Setup"
    echo ""

    check_prerequisites
    setup_environment
    start_services
    verify_services
    display_status
}

# Run main
main
