#!/bin/bash
# ============================================================================
# GCS Sync Script - Backup all Ollama data to Google Cloud Storage
# Runs every hour via gcs-sync container
# ============================================================================

set -e

# Configuration
GCS_BUCKET="${GCS_BUCKET:-gs://elevatediq-ollama-backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATE=$(date +%Y%m%d)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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
# Sync Models (Large Files - Incremental)
# ============================================================================
sync_models() {
    log_info "Syncing models to GCS..."
    
    if [ -d "/data/models" ] && [ "$(ls -A /data/models)" ]; then
        gsutil -m rsync -r -d /data/models "${GCS_BUCKET}/models/"
        log_info "Models synced successfully"
    else
        log_warn "No models found to sync"
    fi
}

# ============================================================================
# Sync PostgreSQL Backups
# ============================================================================
sync_postgres() {
    log_info "Syncing PostgreSQL backups to GCS..."
    
    if [ -d "/data/postgres" ] && [ "$(ls -A /data/postgres)" ]; then
        gsutil -m rsync -r /data/postgres "${GCS_BUCKET}/postgres/${DATE}/"
        log_info "PostgreSQL backups synced successfully"
        
        # Clean old local backups (keep 7 days)
        find /data/postgres -name "*.sql.gz" -mtime +7 -delete 2>/dev/null || true
    else
        log_warn "No PostgreSQL backups found"
    fi
}

# ============================================================================
# Sync Redis Snapshots
# ============================================================================
sync_redis() {
    log_info "Syncing Redis snapshots to GCS..."
    
    if [ -f "/data/redis/dump.rdb" ]; then
        cp /data/redis/dump.rdb "/tmp/redis_${TIMESTAMP}.rdb"
        gsutil cp "/tmp/redis_${TIMESTAMP}.rdb" "${GCS_BUCKET}/redis/${DATE}/"
        rm "/tmp/redis_${TIMESTAMP}.rdb"
        log_info "Redis snapshot synced successfully"
    else
        log_warn "No Redis snapshot found"
    fi
    
    if [ -f "/data/redis/appendonly.aof" ]; then
        cp /data/redis/appendonly.aof "/tmp/redis_${TIMESTAMP}.aof"
        gsutil cp "/tmp/redis_${TIMESTAMP}.aof" "${GCS_BUCKET}/redis/${DATE}/"
        rm "/tmp/redis_${TIMESTAMP}.aof"
        log_info "Redis AOF synced successfully"
    fi
}

# ============================================================================
# Sync Qdrant Snapshots
# ============================================================================
sync_qdrant() {
    log_info "Syncing Qdrant snapshots to GCS..."
    
    if [ -d "/data/qdrant" ] && [ "$(ls -A /data/qdrant)" ]; then
        gsutil -m rsync -r /data/qdrant "${GCS_BUCKET}/qdrant/${DATE}/"
        log_info "Qdrant snapshots synced successfully"
        
        # Clean old local snapshots (keep 3 days)
        find /data/qdrant -name "*.snapshot" -mtime +3 -delete 2>/dev/null || true
    else
        log_warn "No Qdrant snapshots found"
    fi
}

# ============================================================================
# Cleanup Old GCS Backups (Retention Policy)
# ============================================================================
cleanup_old_backups() {
    log_info "Applying retention policies..."
    
    # Keep 30 days of PostgreSQL backups
    gsutil ls "${GCS_BUCKET}/postgres/" | head -n -30 | xargs -r gsutil -m rm -r 2>/dev/null || true
    
    # Keep 14 days of Redis backups
    gsutil ls "${GCS_BUCKET}/redis/" | head -n -14 | xargs -r gsutil -m rm -r 2>/dev/null || true
    
    # Keep 7 days of Qdrant backups
    gsutil ls "${GCS_BUCKET}/qdrant/" | head -n -7 | xargs -r gsutil -m rm -r 2>/dev/null || true
    
    log_info "Cleanup completed"
}

# ============================================================================
# Verify Backups
# ============================================================================
verify_backups() {
    log_info "Verifying backups in GCS..."
    
    local errors=0
    
    # Check models
    if ! gsutil ls "${GCS_BUCKET}/models/" &>/dev/null; then
        log_error "Models backup verification failed"
        ((errors++))
    fi
    
    # Check postgres
    if ! gsutil ls "${GCS_BUCKET}/postgres/${DATE}/" &>/dev/null; then
        log_warn "No PostgreSQL backups for today"
    fi
    
    # Check redis
    if ! gsutil ls "${GCS_BUCKET}/redis/${DATE}/" &>/dev/null; then
        log_warn "No Redis backups for today"
    fi
    
    # Check qdrant
    if ! gsutil ls "${GCS_BUCKET}/qdrant/${DATE}/" &>/dev/null; then
        log_warn "No Qdrant backups for today"
    fi
    
    if [ $errors -eq 0 ]; then
        log_info "Backup verification passed"
        return 0
    else
        log_error "Backup verification failed with $errors errors"
        return 1
    fi
}

# ============================================================================
# Main Execution
# ============================================================================
main() {
    log_info "Starting GCS backup sync at $(date)"
    log_info "Target bucket: ${GCS_BUCKET}"
    
    # Sync all components
    sync_models
    sync_postgres
    sync_redis
    sync_qdrant
    
    # Cleanup old backups
    cleanup_old_backups
    
    # Verify
    verify_backups
    
    log_info "GCS backup sync completed successfully at $(date)"
}

# Run main function
main

exit 0
