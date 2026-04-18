#!/bin/bash
# ============================================================================
# Qdrant Snapshot Backup Script
# Creates snapshots via API and prepares for GCS sync
# Run via cron: 0 */6 * * * /opt/ollama/scripts/backup-qdrant.sh
# ============================================================================

set -e

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/mnt/backups/qdrant}"
QDRANT_HOST="${QDRANT_HOST:-localhost}"
QDRANT_PORT="${QDRANT_PORT:-6333}"
COLLECTION="${QDRANT_COLLECTION:-embeddings}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

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
# Create backup directory
# ============================================================================
mkdir -p "$BACKUP_DIR"

# ============================================================================
# Create snapshot via API
# ============================================================================
log_info "Creating Qdrant snapshot at $(date)"
log_info "Collection: ${COLLECTION}"

# Create snapshot
RESPONSE=$(curl -s -X POST "http://${QDRANT_HOST}:${QDRANT_PORT}/collections/${COLLECTION}/snapshots" \
    -H "Content-Type: application/json")

# Check if successful
if echo "$RESPONSE" | grep -q '"status":"ok"'; then
    SNAPSHOT_NAME=$(echo "$RESPONSE" | grep -o '"name":"[^"]*"' | cut -d'"' -f4)
    log_info "Snapshot created: ${SNAPSHOT_NAME}"
else
    log_error "Failed to create snapshot"
    echo "$RESPONSE"
    exit 1
fi

# ============================================================================
# Wait for snapshot to complete
# ============================================================================
log_info "Waiting for snapshot to complete..."
sleep 5

# ============================================================================
# Download snapshot
# ============================================================================
SNAPSHOT_FILE="${BACKUP_DIR}/${COLLECTION}_${TIMESTAMP}.snapshot"

log_info "Downloading snapshot..."
if curl -s -f "http://${QDRANT_HOST}:${QDRANT_PORT}/collections/${COLLECTION}/snapshots/${SNAPSHOT_NAME}" \
    -o "$SNAPSHOT_FILE"; then
    SNAPSHOT_SIZE=$(du -h "$SNAPSHOT_FILE" | cut -f1)
    log_info "Snapshot downloaded: ${SNAPSHOT_SIZE}"
else
    log_error "Failed to download snapshot"
    exit 1
fi

# ============================================================================
# Create metadata
# ============================================================================
cat > "${BACKUP_DIR}/${COLLECTION}_${TIMESTAMP}.meta" <<EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "collection": "${COLLECTION}",
  "snapshot_name": "${SNAPSHOT_NAME}",
  "snapshot_file": "$(basename $SNAPSHOT_FILE)",
  "snapshot_size": "$(stat -f%z "$SNAPSHOT_FILE" 2>/dev/null || stat -c%s "$SNAPSHOT_FILE")"
}
EOF

# ============================================================================
# Clean old local snapshots (keep 3 days)
# ============================================================================
log_info "Cleaning old local snapshots..."
find "$BACKUP_DIR" -name "${COLLECTION}_*.snapshot" -mtime +3 -delete
find "$BACKUP_DIR" -name "${COLLECTION}_*.meta" -mtime +3 -delete

log_info "Qdrant backup completed successfully at $(date)"

exit 0
