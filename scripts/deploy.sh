#!/bin/bash
# ============================================================================
# One-command deployment for Ollama Elite stack
# ============================================================================
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

COMPOSE_FILE="docker/docker-compose.elite.yml"

compose_cmd() {
    if command -v docker-compose &>/dev/null; then
        echo docker-compose
        return
    fi

    if docker compose version &>/dev/null; then
        echo docker compose
        return
    fi

    fail "Docker Compose not installed"
}

step() { echo -e "\n➡️  $1"; }
success() { echo -e "✅ $1"; }
fail() { echo -e "❌ $1"; exit 1; }

# 1) Pre-flight checks
step "Running pre-flight checks"
if ! command -v docker &>/dev/null; then fail "Docker not installed"; fi
if ! command -v nvidia-smi &>/dev/null; then echo "⚠️  GPU not detected (nvidia-smi missing)"; fi
if ! test -f .env.production; then fail ".env.production missing"; fi

COMPOSE_BIN="$(compose_cmd)"

# Required dirs
mkdir -p /mnt/data/ollama/models \
         /mnt/data/ollama/postgres \
         /mnt/data/ollama/qdrant \
         /mnt/backups/postgres \
         /mnt/backups/qdrant \
         /var/lib/ollama/logs

# 2) Pull images
step "Pulling images"
$COMPOSE_BIN -f "$COMPOSE_FILE" pull

# 3) Start stack
step "Starting stack"
$COMPOSE_BIN -f "$COMPOSE_FILE" up -d

# 4) Show status
step "Services status"
$COMPOSE_BIN -f "$COMPOSE_FILE" ps

# 5) Tail key logs (short)
step "Recent logs (ollama-api)"
$COMPOSE_BIN -f "$COMPOSE_FILE" logs --tail=50 ollama-api

success "Deployment complete. Stack is running on the target server (nginx 80/443, API 8000 internal)."
