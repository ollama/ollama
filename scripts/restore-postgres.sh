#!/bin/bash
# ============================================================================
# PostgreSQL Restore Script
# Restores database from backup file
# Usage: ./restore-postgres.sh /path/to/backup.sql.gz
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="${PROJECT_ROOT}/docker/docker-compose.elite.yml"

compose_cmd() {
    if command -v docker-compose &>/dev/null; then
        echo docker-compose
        return
    fi

    if docker compose version &>/dev/null; then
        echo docker compose
        return
    fi

    log_error "Docker Compose not installed"
    exit 1
}

# Configuration
POSTGRES_HOST="${POSTGRES_HOST:-postgres}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_USER="${POSTGRES_USER:-ollama}"
POSTGRES_DB="${POSTGRES_DB:-ollama}"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# Check arguments
# ============================================================================
if [ $# -eq 0 ]; then
    log_error "Usage: $0 <backup_file.sql.gz>"
    log_info "Available backups:"
    ls -lh /mnt/backups/postgres/*.sql.gz 2>/dev/null || echo "  No backups found"
    exit 1
fi

BACKUP_FILE="$1"

# ============================================================================
# Verify backup file
# ============================================================================
if [ ! -f "$BACKUP_FILE" ]; then
    log_error "Backup file not found: ${BACKUP_FILE}"
    exit 1
fi

log_info "Backup file: ${BACKUP_FILE}"
log_info "Size: $(du -h "$BACKUP_FILE" | cut -f1)"

# Test if valid gzip
if ! gzip -t "$BACKUP_FILE" 2>/dev/null; then
    log_error "Invalid gzip file"
    exit 1
fi

# ============================================================================
# Confirm restoration
# ============================================================================
log_warn "⚠️  WARNING: This will DROP and RECREATE the database!"
log_warn "Database: ${POSTGRES_DB}@${POSTGRES_HOST}"
read -p "Are you sure you want to continue? (type 'yes' to confirm): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    log_info "Restore cancelled"
    exit 0
fi

# ============================================================================
# Stop dependent services
# ============================================================================
log_info "Stopping dependent services..."
DOCKER_COMPOSE="$(compose_cmd)"
$DOCKER_COMPOSE -f "$COMPOSE_FILE" stop ollama-api grafana

# ============================================================================
# Restore database
# ============================================================================
log_info "Starting restore at $(date)"

# Drop and recreate database
docker exec ollama-postgres psql -U "$POSTGRES_USER" -d postgres <<EOF
DROP DATABASE IF EXISTS ${POSTGRES_DB};
CREATE DATABASE ${POSTGRES_DB};
GRANT ALL PRIVILEGES ON DATABASE ${POSTGRES_DB} TO ${POSTGRES_USER};
EOF

# Restore from backup
log_info "Restoring data..."
gunzip -c "$BACKUP_FILE" | docker exec -i ollama-postgres pg_restore \
    -U "$POSTGRES_USER" \
    -d "$POSTGRES_DB" \
    --no-owner \
    --no-acl \
    --clean \
    --if-exists \
    -v 2>&1 | tee /tmp/restore.log

# ============================================================================
# Verify restoration
# ============================================================================
log_info "Verifying restoration..."

TABLE_COUNT=$(docker exec ollama-postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c \
    "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';")

log_info "Tables restored: ${TABLE_COUNT}"

# ============================================================================
# Restart services
# ============================================================================
log_info "Restarting services..."
$DOCKER_COMPOSE -f "$COMPOSE_FILE" start ollama-api grafana

log_info "Restore completed successfully at $(date)"
log_info "Restore log: /tmp/restore.log"

exit 0
