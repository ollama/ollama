#!/bin/bash
# =============================================================================
# LOCAL DEVELOPMENT AUTOMATION
# Complete local environment management with Docker Compose automation
# =============================================================================

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENVIRONMENT="${1:-local}"
ACTION="${2:-start}"
COMPOSE_FILE="${PROJECT_ROOT}/docker/docker-compose.local.yml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

compose_cmd() {
    if command -v docker-compose &>/dev/null; then
        echo docker-compose
        return
    fi

    if docker compose version &>/dev/null; then
        echo docker compose
        return
    fi

    error "Docker Compose is not installed"
    exit 1
}

# Logging
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $*" | tee -a "$LOG_DIR/local-dev-${TIMESTAMP}.log"
}

success() {
    echo -e "${GREEN}[✓]${NC} $*"
}

error() {
    echo -e "${RED}[✗]${NC} $*"
}

warning() {
    echo -e "${YELLOW}[!]${NC} $*"
}

# =============================================================================
# LOCAL ENVIRONMENT SETUP
# =============================================================================

setup_environment() {
    log "🔧 Setting up local development environment..."

    # Create .env files if they don't exist
    if [[ ! -f "$PROJECT_ROOT/.env.local" ]]; then
        log "Creating .env.local..."
        LOCAL_IP="$(hostname -I | awk '{print $1}')"
        cat > "$PROJECT_ROOT/.env.local" << EOF
# Local Development Environment
ENVIRONMENT=local
DEBUG=true
LOG_LEVEL=debug

# FastAPI Configuration
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
PUBLIC_API_URL=http://${LOCAL_IP}:8000

# Database
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/ollama
POSTGRES_DB=ollama
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# Redis
REDIS_URL=redis://redis:6379/0
REDIS_PASSWORD=redis-dev-password

# Ollama
OLLAMA_BASE_URL=http://ollama:11434

# Qdrant (Vector DB)
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=qdrant-dev-key

# Monitoring
PROMETHEUS_URL=http://prometheus:9090
GRAFANA_URL=http://grafana:3000
GRAFANA_ADMIN_PASSWORD=admin

# Development features
RELOAD_ON_CHANGE=true
AUTO_MIGRATE=true
SEED_DATABASE=true

# Testing
TEST_DATABASE_URL=postgresql://postgres:postgres@postgres:5432/ollama_test
EOF
        success "Created .env.local"
    fi

    # Create docker-compose override for local development
    if [[ ! -f "$PROJECT_ROOT/docker-compose.override.yml" ]]; then
        log "Creating docker-compose.override.yml..."
        cat > "$PROJECT_ROOT/docker-compose.override.yml" << 'EOF'
version: '3.9'

services:
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile
    volumes:
      - ./ollama:/app/ollama
      - ./tests:/app/tests
    environment:
      - PYTHONUNBUFFERED=1
      - RELOAD_ON_CHANGE=true
    ports:
      - "127.0.0.1:8000:8000"
    command: python -m uvicorn ollama.main:app --reload --host 0.0.0.0

  postgres:
    environment:
      - POSTGRES_INITDB_ARGS=-c log_statement=all
    volumes:
      - postgres_local_data:/var/lib/postgresql/data

  redis:
    command: redis-server --appendonly yes --loglevel debug

volumes:
  postgres_local_data:
EOF
        success "Created docker-compose.override.yml"
    fi

    success "Local environment configured"
}

# =============================================================================
# DOCKER COMPOSE OPERATIONS
# =============================================================================

validate_docker() {
    log "✓ Validating Docker setup..."

    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi

    compose_cmd >/dev/null

    if ! docker ps &>/dev/null; then
        error "Docker daemon is not running"
        exit 1
    fi

    success "Docker setup validated"
}

compose_up() {
    log "🚀 Starting Docker containers..."

    cd "$PROJECT_ROOT"

    DOCKER_COMPOSE="$(compose_cmd)"

    $DOCKER_COMPOSE -f "$COMPOSE_FILE" up -d \
        --remove-orphans \
        --build

    log "Waiting for services to be ready..."
    sleep 10

    success "Docker containers started"
}

compose_down() {
    log "🛑 Stopping Docker containers..."

    cd "$PROJECT_ROOT"

    DOCKER_COMPOSE="$(compose_cmd)"
    $DOCKER_COMPOSE -f "$COMPOSE_FILE" down --remove-orphans

    success "Docker containers stopped"
}

compose_logs() {
    log "📋 Showing Docker logs..."

    cd "$PROJECT_ROOT"

    DOCKER_COMPOSE="$(compose_cmd)"
    $DOCKER_COMPOSE -f "$COMPOSE_FILE" logs -f --tail=50
}

compose_ps() {
    log "📦 Docker container status..."

    cd "$PROJECT_ROOT"

    DOCKER_COMPOSE="$(compose_cmd)"
    $DOCKER_COMPOSE -f "$COMPOSE_FILE" ps
}

compose_clean() {
    log "🧹 Cleaning Docker volumes and networks..."

    cd "$PROJECT_ROOT"

    DOCKER_COMPOSE="$(compose_cmd)"
    $DOCKER_COMPOSE -f "$COMPOSE_FILE" down -v --remove-orphans

    warning "All local volumes have been removed"
    success "Docker cleanup completed"
}

# =============================================================================
# DATABASE SETUP
# =============================================================================

migrate_database() {
    log "📊 Running database migrations..."

    cd "$PROJECT_ROOT"

    DOCKER_COMPOSE="$(compose_cmd)"
    $DOCKER_COMPOSE -f "$COMPOSE_FILE" exec -T api \
        alembic upgrade head

    success "Database migrations completed"
}

seed_database() {
    log "🌱 Seeding database with test data..."

    cd "$PROJECT_ROOT"

    DOCKER_COMPOSE="$(compose_cmd)"
    $DOCKER_COMPOSE -f "$COMPOSE_FILE" exec -T api \
        python -m scripts.seed_database

    success "Database seeded"
}

reset_database() {
    log "🔄 Resetting database..."

    cd "$PROJECT_ROOT"

    DOCKER_COMPOSE="$(compose_cmd)"
    $DOCKER_COMPOSE -f "$COMPOSE_FILE" exec -T postgres \
        psql -U postgres -d ollama -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"

    migrate_database
    seed_database

    success "Database reset and reseeded"
}

# =============================================================================
# DEVELOPMENT OPERATIONS
# =============================================================================

build_services() {
    log "🔨 Building services..."

    cd "$PROJECT_ROOT"

    DOCKER_COMPOSE="$(compose_cmd)"
    $DOCKER_COMPOSE -f "$COMPOSE_FILE" build --no-cache

    success "Services built"
}

run_tests() {
    log "🧪 Running tests in container..."

    cd "$PROJECT_ROOT"

    DOCKER_COMPOSE="$(compose_cmd)"
    $DOCKER_COMPOSE -f "$COMPOSE_FILE" exec -T api \
        pytest tests/ -v --cov=ollama --cov-report=html

    success "Tests completed"
}

run_type_checks() {
    log "📝 Running type checks..."

    cd "$PROJECT_ROOT"

    DOCKER_COMPOSE="$(compose_cmd)"
    $DOCKER_COMPOSE -f "$COMPOSE_FILE" exec -T api \
        mypy ollama/ --strict

    success "Type checks completed"
}

run_linting() {
    log "🔨 Running linting..."

    cd "$PROJECT_ROOT"

    DOCKER_COMPOSE="$(compose_cmd)"
    $DOCKER_COMPOSE -f "$COMPOSE_FILE" exec -T api \
        ruff check ollama/ tests/

    success "Linting completed"
}

run_all_checks() {
    log "✓ Running all quality checks..."

    run_linting
    run_type_checks
    run_tests

    success "All quality checks completed"
}

# =============================================================================
# HEALTH CHECKS
# =============================================================================

health_check_services() {
    log "🏥 Checking service health..."

    cd "$PROJECT_ROOT"

    # Check API
    log "Checking API..."
    if curl -sf http://localhost:8000/api/v1/health >/dev/null; then
        success "API is healthy"
    else
        error "API is not responding"
        return 1
    fi

    # Check PostgreSQL
    log "Checking PostgreSQL..."
    DOCKER_COMPOSE="$(compose_cmd)"
    if $DOCKER_COMPOSE -f "$COMPOSE_FILE" exec -T postgres \
        psql -U postgres -c "SELECT 1" >/dev/null 2>&1; then
        success "PostgreSQL is healthy"
    else
        error "PostgreSQL is not responding"
        return 1
    fi

    # Check Redis
    log "Checking Redis..."
    DOCKER_COMPOSE="$(compose_cmd)"
    if $DOCKER_COMPOSE -f "$COMPOSE_FILE" exec -T redis \
        redis-cli ping >/dev/null 2>&1; then
        success "Redis is healthy"
    else
        error "Redis is not responding"
        return 1
    fi

    # Check Ollama
    log "Checking Ollama..."
    if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
        success "Ollama is healthy"
    else
        error "Ollama is not responding"
        return 1
    fi

    success "All services are healthy"
}

# =============================================================================
# MONITORING & DEBUGGING
# =============================================================================

show_metrics() {
    log "📊 Prometheus metrics available at: http://localhost:9090"
    log "📈 Grafana dashboards available at: http://localhost:3000"
    log "🐘 pgAdmin available at: http://localhost:5050"
    log "🔍 Jaeger tracing available at: http://localhost:16686"
}

open_shell() {
    local service="${1:api}"

    log "Opening shell to $service container..."

    cd "$PROJECT_ROOT"

    DOCKER_COMPOSE="$(compose_cmd)"
    $DOCKER_COMPOSE -f "$COMPOSE_FILE" exec "$service" bash || \
    $DOCKER_COMPOSE -f "$COMPOSE_FILE" exec "$service" sh
}

show_ports() {
    log "📡 Available services:"
    cat << EOF

    API:          http://localhost:8000
    Health:       http://localhost:8000/api/v1/health
    Prometheus:   http://localhost:9090
    Grafana:      http://localhost:3000 (admin/admin)
    pgAdmin:      http://localhost:5050
    Jaeger:       http://localhost:16686
    Redis:        localhost:6379
    PostgreSQL:   localhost:5432
    Ollama:       http://localhost:11434
    Qdrant:       http://localhost:6333

EOF
}

# =============================================================================
# FULL CYCLE MANAGEMENT
# =============================================================================

full_setup() {
    log "🚀 FULL LOCAL SETUP"

    validate_docker
    setup_environment
    compose_down
    compose_clean
    compose_up
    migrate_database
    seed_database
    sleep 5
    health_check_services
    run_all_checks

    success "✅ Full setup completed!"
    show_ports
}

full_teardown() {
    log "💣 FULL LOCAL TEARDOWN"

    compose_down
    compose_clean

    success "✅ Full teardown completed!"
}

full_reset() {
    log "🔄 FULL LOCAL RESET"

    compose_down
    reset_database
    compose_up
    health_check_services

    success "✅ Full reset completed!"
}

# =============================================================================
# MAIN
# =============================================================================

usage() {
    cat << EOF
Usage: $0 <action> [options]

Actions:
  start           Start local development environment
  stop            Stop containers
  restart         Restart containers
  logs            Show container logs
  ps              Show container status
  clean           Clean volumes and networks

  setup           Full local setup (clean + start + migrate + seed + test)
  reset           Reset database and restart
  teardown        Full teardown (stop + clean)

  test            Run tests
  lint            Run linting
  type-check      Run type checking
  all-checks      Run all quality checks

  health          Check service health
  shell [service] Open shell to container (default: api)
  ports           Show available service ports
  logs            Show service logs
  metrics         Show monitoring endpoints

  migrate         Run database migrations
  seed            Seed database with test data
  reset-db        Reset and reseed database

  build           Build services
  up              Start containers
  down            Stop containers

Examples:
  $0 start              # Start development environment
  $0 test               # Run tests
  $0 shell postgres     # Connect to PostgreSQL container
  $0 setup              # Full setup from scratch
  $0 reset              # Reset database

EOF
    exit 0
}

main() {
    if [[ "$ENVIRONMENT" == "--help" ]] || [[ "$ENVIRONMENT" == "-h" ]]; then
        usage
    fi

    # Handle single-word actions (shift if first arg looks like an action)
    if [[ -z "$ACTION" ]] || [[ "$ACTION" == "--help" ]]; then
        ACTION="$ENVIRONMENT"
    fi

    log "🐳 LOCAL DEVELOPMENT AUTOMATION"
    log "Action: $ACTION"

    case "$ACTION" in
        # Container management
        up)
            validate_docker
            compose_up
            ;;
        start)
            validate_docker
            setup_environment
            compose_up
            health_check_services
            show_ports
            ;;
        down|stop)
            compose_down
            ;;
        restart)
            compose_down
            compose_up
            ;;
        logs)
            compose_logs
            ;;
        ps)
            compose_ps
            ;;
        clean)
            compose_clean
            ;;
        build)
            validate_docker
            build_services
            ;;

        # Full management
        setup)
            full_setup
            ;;
        reset)
            full_reset
            ;;
        teardown)
            full_teardown
            ;;

        # Database
        migrate)
            migrate_database
            ;;
        seed)
            seed_database
            ;;
        reset-db)
            reset_database
            ;;

        # Testing & QA
        test)
            run_tests
            ;;
        lint)
            run_linting
            ;;
        type-check)
            run_type_checks
            ;;
        all-checks)
            run_all_checks
            ;;

        # Monitoring
        health)
            health_check_services
            ;;
        shell)
            open_shell "${2:api}"
            ;;
        ports)
            show_ports
            ;;
        metrics)
            show_metrics
            ;;

        *)
            error "Unknown action: $ACTION"
            usage
            exit 1
            ;;
    esac
}

main "$@"
