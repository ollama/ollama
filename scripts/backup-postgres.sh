#!/bin/bash
# ============================================================================
# PostgreSQL Backup Script
# Creates compressed SQL dumps and prepares for GCS sync
# Run via cron: 0 2 * * * /opt/ollama/scripts/backup-postgres.sh
# ============================================================================

set -e

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/mnt/backups/postgres}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATE=$(date +%Y%m%d)

# Database config
POSTGRES_HOST="${POSTGRES_HOST:-postgres}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_USER="${POSTGRES_USER:-ollama}"
POSTGRES_DB="${POSTGRES_DB:-ollama}"

# Backup filename
BACKUP_FILE="${BACKUP_DIR}/ollama_${TIMESTAMP}.sql.gz"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# Create backup directory
# ============================================================================
mkdir -p "$BACKUP_DIR"

# ============================================================================
# Create backup
# ============================================================================
log_info "Starting PostgreSQL backup at $(date)"
log_info "Database: ${POSTGRES_DB}@${POSTGRES_HOST}:${POSTGRES_PORT}"
log_info "Backup file: ${BACKUP_FILE}"

# Run pg_dump from inside the container
docker exec ollama-postgres pg_dump \
    -U "$POSTGRES_USER" \
    -d "$POSTGRES_DB" \
    -Fc \
    --no-owner \
    --no-acl \
    --clean \
    --if-exists \
    | gzip > "$BACKUP_FILE"

# ============================================================================
# Verify backup
# ============================================================================
if [ -f "$BACKUP_FILE" ]; then
    BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
    log_info "Backup created successfully: ${BACKUP_SIZE}"
else
    log_error "Backup failed - file not created"
    exit 1
fi

# Test if backup is valid gzip
if ! gzip -t "$BACKUP_FILE" 2>/dev/null; then
    log_error "Backup verification failed - corrupt gzip file"
    exit 1
fi

# ============================================================================
# Backup Grafana database (if exists)
# ============================================================================
GRAFANA_BACKUP="${BACKUP_DIR}/grafana_${TIMESTAMP}.sql.gz"

if docker exec ollama-postgres psql -U "$POSTGRES_USER" -lqt | cut -d \| -f 1 | grep -qw grafana; then
    log_info "Backing up Grafana database..."
    docker exec ollama-postgres pg_dump \
        -U "$POSTGRES_USER" \
        -d grafana \
        -Fc \
        --no-owner \
        --no-acl \
        | gzip > "$GRAFANA_BACKUP"
    log_info "Grafana backup created: $(du -h $GRAFANA_BACKUP | cut -f1)"
fi

# ============================================================================
# Create metadata file
# ============================================================================
cat > "${BACKUP_DIR}/backup_${TIMESTAMP}.meta" <<EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "database": "${POSTGRES_DB}",
  "host": "${POSTGRES_HOST}",
  "backup_file": "$(basename $BACKUP_FILE)",
  "backup_size": "$(stat -f%z "$BACKUP_FILE" 2>/dev/null || stat -c%s "$BACKUP_FILE")",
  "postgres_version": "$(docker exec ollama-postgres psql -U $POSTGRES_USER -t -c 'SELECT version();' | head -n 1 | xargs)"
}
EOF

# ============================================================================
# Clean old local backups
# ============================================================================
log_info "Cleaning backups older than ${RETENTION_DAYS} days..."
find "$BACKUP_DIR" -name "ollama_*.sql.gz" -mtime +${RETENTION_DAYS} -delete
find "$BACKUP_DIR" -name "grafana_*.sql.gz" -mtime +${RETENTION_DAYS} -delete
find "$BACKUP_DIR" -name "backup_*.meta" -mtime +${RETENTION_DAYS} -delete

log_info "PostgreSQL backup completed successfully at $(date)"

exit 0
