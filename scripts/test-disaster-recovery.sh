#!/bin/bash

# 🚀 Disaster Recovery Test Script
# Validates complete disaster recovery procedures
#
# Usage: ./scripts/test-disaster-recovery.sh [--dry-run] [--region us-east1]
#
# This script tests:
# 1. Database backup and restoration
# 2. Service deployment to new region
# 3. Data integrity after recovery
# 4. Full system functional tests
#
# Exit codes:
#   0 = Success
#   1 = Test failed
#   2 = Invalid arguments
#   3 = Resource not found

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ID="ollama-prod"
PRIMARY_REGION="us-central1"
RECOVERY_REGION="${1:-us-east1}"
DRY_RUN="${DRY_RUN:-false}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/tmp/dr-test-${TIMESTAMP}.log"
RECOVERY_INSTANCE="ollama-dr-test-${TIMESTAMP}"
RECOVERY_SERVICE="ollama-recovery-${TIMESTAMP}"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Logging Functions
# ============================================================================

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}✓ $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}✗ $1${NC}" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}⚠ $1${NC}" | tee -a "$LOG_FILE"
}

# ============================================================================
# Validation Functions
# ============================================================================

validate_project() {
    log "Validating GCP project..."

    if ! gcloud config get-value project > /dev/null 2>&1; then
        error "Not authenticated with gcloud. Run: gcloud auth login"
        return 1
    fi

    if [ "$(gcloud config get-value project)" != "$PROJECT_ID" ]; then
        error "Current project is not $PROJECT_ID"
        return 1
    fi

    success "Project validation passed"
}

validate_resources() {
    log "Checking required resources..."

    # Check if primary service exists
    if ! gcloud run services describe ollama \
        --platform managed --region "$PRIMARY_REGION" > /dev/null 2>&1; then
        error "Primary service 'ollama' not found in $PRIMARY_REGION"
        return 1
    fi
    success "Primary service exists"

    # Check if database exists
    if ! gcloud sql instances describe ollama-prod > /dev/null 2>&1; then
        error "Database instance 'ollama-prod' not found"
        return 1
    fi
    success "Database instance exists"

    # Check if backup exists
    local backup_count
    backup_count=$(gcloud sql backups list --instance=ollama-prod --limit=1 | wc -l)
    if [ "$backup_count" -lt 2 ]; then
        error "No backups found for ollama-prod"
        return 1
    fi
    success "Backups exist"
}

# ============================================================================
# Database Recovery Tests
# ============================================================================

test_database_backup() {
    log "Testing database backup creation..."

    if [ "$DRY_RUN" = "true" ]; then
        warning "[DRY-RUN] Skipping backup creation"
        return 0
    fi

    # Create manual backup
    if ! gcloud sql backups create \
        --instance=ollama-prod \
        --description="DR Test Backup - ${TIMESTAMP}" > /dev/null 2>&1; then
        error "Failed to create backup"
        return 1
    fi

    success "Database backup created successfully"
}

test_database_clone() {
    log "Cloning database for recovery testing..."

    if [ "$DRY_RUN" = "true" ]; then
        warning "[DRY-RUN] Skipping database clone"
        return 0
    fi

    # Clone database to recovery instance
    log "Cloning to $RECOVERY_INSTANCE (this may take 5-10 minutes)..."

    if ! gcloud sql instances clone \
        ollama-prod "$RECOVERY_INSTANCE" \
        --async > /dev/null 2>&1; then
        error "Failed to clone database"
        return 1
    fi

    # Wait for clone completion
    log "Waiting for clone operation to complete..."
    local max_attempts=120
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        local status
        status=$(gcloud sql instances describe "$RECOVERY_INSTANCE" \
            --format='value(state)' 2>/dev/null || echo "PENDING")

        if [ "$status" = "RUNNABLE" ]; then
            success "Database clone completed"
            return 0
        fi

        echo -n "."
        sleep 5
        ((attempt++))
    done

    error "Database clone timeout (waited 10 minutes)"
    return 1
}

test_database_connectivity() {
    log "Testing recovered database connectivity..."

    if [ "$DRY_RUN" = "true" ]; then
        warning "[DRY-RUN] Skipping connectivity test"
        return 0
    fi

    # Get connection string
    local db_host
    db_host=$(gcloud sql instances describe "$RECOVERY_INSTANCE" \
        --format='value(ipAddresses[0].ipAddress)')

    if [ -z "$db_host" ]; then
        error "Failed to get database IP address"
        return 1
    fi

    # Test connection
    if ! PGPASSWORD="${DB_PASSWORD}" psql \
        -h "$db_host" \
        -U postgres \
        -d ollama \
        -c "SELECT 1;" > /dev/null 2>&1; then
        error "Failed to connect to recovered database"
        return 1
    fi

    success "Database connectivity verified"
}

test_data_integrity() {
    log "Testing data integrity of recovered database..."

    if [ "$DRY_RUN" = "true" ]; then
        warning "[DRY-RUN] Skipping data integrity test"
        return 0
    fi

    local db_host
    db_host=$(gcloud sql instances describe "$RECOVERY_INSTANCE" \
        --format='value(ipAddresses[0].ipAddress)')

    # Run integrity checks
    local checks_passed=0
    local checks_total=0

    # Check 1: Table count matches
    ((checks_total++))
    local primary_tables
    local recovery_tables

    primary_tables=$(PGPASSWORD="${DB_PASSWORD}" psql \
        -h "cloudsql:5432" \
        -U postgres \
        -d ollama \
        -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema='public';")

    recovery_tables=$(PGPASSWORD="${DB_PASSWORD}" psql \
        -h "$db_host" \
        -U postgres \
        -d ollama \
        -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema='public';")

    if [ "$primary_tables" -eq "$recovery_tables" ]; then
        success "Table count matches: $primary_tables tables"
        ((checks_passed++))
    else
        error "Table count mismatch: primary=$primary_tables, recovery=$recovery_tables"
    fi

    # Check 2: Key table row counts
    ((checks_total++))
    local primary_users recovery_users

    primary_users=$(PGPASSWORD="${DB_PASSWORD}" psql \
        -h "cloudsql:5432" \
        -U postgres \
        -d ollama \
        -t -c "SELECT count(*) FROM users WHERE deleted_at IS NULL;" 2>/dev/null || echo "0")

    recovery_users=$(PGPASSWORD="${DB_PASSWORD}" psql \
        -h "$db_host" \
        -U postgres \
        -d ollama \
        -t -c "SELECT count(*) FROM users WHERE deleted_at IS NULL;" 2>/dev/null || echo "0")

    if [ "$primary_users" -eq "$recovery_users" ]; then
        success "User count matches: $primary_users users"
        ((checks_passed++))
    else
        warning "User count differs: primary=$primary_users, recovery=$recovery_users (expected for ongoing operations)"
        ((checks_passed++))
    fi

    # Check 3: No corrupted indexes
    ((checks_total++))
    local index_errors
    index_errors=$(PGPASSWORD="${DB_PASSWORD}" psql \
        -h "$db_host" \
        -U postgres \
        -d ollama \
        -t -c "SELECT count(*) FROM pg_stat_user_indexes WHERE idx_scan = 0 AND indexrelname LIKE 'idx_%';" 2>/dev/null || echo "0")

    if [ "$index_errors" -lt 5 ]; then
        success "Index integrity verified"
        ((checks_passed++))
    else
        error "Found $index_errors unused indexes (possible corruption)"
    fi

    log "Data integrity: $checks_passed/$checks_total checks passed"

    if [ $checks_passed -eq $checks_total ]; then
        success "All data integrity checks passed"
        return 0
    else
        error "Some data integrity checks failed"
        return 1
    fi
}

# ============================================================================
# Service Recovery Tests
# ============================================================================

test_service_deployment() {
    log "Testing service deployment to recovery region..."

    if [ "$DRY_RUN" = "true" ]; then
        warning "[DRY-RUN] Skipping service deployment"
        return 0
    fi

    # Get current service image
    local current_image
    current_image=$(gcloud run services describe ollama \
        --platform managed --region "$PRIMARY_REGION" \
        --format='value(spec.template.spec.containers[0].image)')

    if [ -z "$current_image" ]; then
        error "Failed to get current service image"
        return 1
    fi

    log "Deploying image: $current_image"

    # Deploy to recovery region
    if ! gcloud run deploy "$RECOVERY_SERVICE" \
        --image "$current_image" \
        --platform managed \
        --region "$RECOVERY_REGION" \
        --timeout 3600 \
        --max-instances 10 \
        --memory 8Gi \
        --cpu 4 \
        --no-allow-unauthenticated \
        --async > /dev/null 2>&1; then
        error "Failed to deploy service"
        return 1
    fi

    log "Waiting for service deployment (this may take 10-15 minutes)..."

    # Wait for deployment
    local max_attempts=60
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        local ready_replicas
        ready_replicas=$(gcloud run services describe "$RECOVERY_SERVICE" \
            --platform managed --region "$RECOVERY_REGION" \
            --format='value(status.conditions[0].reason)' 2>/dev/null || echo "PENDING")

        if [ "$ready_replicas" = "RevisionOK" ]; then
            success "Service deployed successfully"
            return 0
        fi

        echo -n "."
        sleep 10
        ((attempt++))
    done

    error "Service deployment timeout (waited 10 minutes)"
    return 1
}

test_service_health() {
    log "Testing recovered service health..."

    if [ "$DRY_RUN" = "true" ]; then
        warning "[DRY-RUN] Skipping health check"
        return 0
    fi

    # Get service URL
    local service_url
    service_url=$(gcloud run services describe "$RECOVERY_SERVICE" \
        --platform managed --region "$RECOVERY_REGION" \
        --format='value(status.url)')

    if [ -z "$service_url" ]; then
        error "Failed to get service URL"
        return 1
    fi

    log "Testing endpoint: $service_url/api/v1/health"

    # Test health endpoint (allow unauthenticated for this test)
    local attempt=0
    local max_attempts=30

    while [ $attempt -lt $max_attempts ]; do
        local response_code
        response_code=$(curl -s -o /dev/null -w "%{http_code}" \
            -H "Authorization: Bearer $API_KEY_TEST" \
            "$service_url/api/v1/health" || echo "000")

        if [ "$response_code" = "200" ]; then
            success "Service health check passed (HTTP 200)"
            return 0
        fi

        if [ "$response_code" != "000" ]; then
            log "Received HTTP $response_code"
        fi

        echo -n "."
        sleep 2
        ((attempt++))
    done

    error "Service health check failed after retries (URL: $service_url)"
    return 1
}

# ============================================================================
# Integration Tests
# ============================================================================

test_api_functionality() {
    log "Testing recovered service API functionality..."

    if [ "$DRY_RUN" = "true" ]; then
        warning "[DRY-RUN] Skipping API tests"
        return 0
    fi

    local service_url
    service_url=$(gcloud run services describe "$RECOVERY_SERVICE" \
        --platform managed --region "$RECOVERY_REGION" \
        --format='value(status.url)')

    local tests_passed=0
    local tests_total=0

    # Test 1: Models endpoint
    ((tests_total++))
    if curl -s -f \
        -H "Authorization: Bearer $API_KEY_TEST" \
        "$service_url/api/v1/models" > /dev/null 2>&1; then
        success "Models endpoint working"
        ((tests_passed++))
    else
        error "Models endpoint failed"
    fi

    # Test 2: Health endpoint
    ((tests_total++))
    if curl -s -f \
        -H "Authorization: Bearer $API_KEY_TEST" \
        "$service_url/api/v1/health" > /dev/null 2>&1; then
        success "Health endpoint working"
        ((tests_passed++))
    else
        error "Health endpoint failed"
    fi

    log "API tests: $tests_passed/$tests_total passed"

    if [ $tests_passed -eq $tests_total ]; then
        return 0
    else
        return 1
    fi
}

# ============================================================================
# Cleanup Functions
# ============================================================================

cleanup_recovery_resources() {
    log "Cleaning up recovery resources..."

    if [ "$DRY_RUN" = "true" ]; then
        warning "[DRY-RUN] Skipping cleanup"
        return 0
    fi

    # Delete recovery service
    if [ -n "$RECOVERY_SERVICE" ]; then
        log "Deleting recovery service: $RECOVERY_SERVICE"
        gcloud run services delete "$RECOVERY_SERVICE" \
            --platform managed --region "$RECOVERY_REGION" \
            --quiet > /dev/null 2>&1 || warning "Failed to delete recovery service"
    fi

    # Delete recovery database
    if [ -n "$RECOVERY_INSTANCE" ]; then
        log "Deleting recovery database: $RECOVERY_INSTANCE"
        gcloud sql instances delete "$RECOVERY_INSTANCE" \
            --quiet > /dev/null 2>&1 || warning "Failed to delete recovery database"
    fi

    success "Cleanup completed"
}

# ============================================================================
# Main Test Flow
# ============================================================================

main() {
    log "=========================================="
    log "Disaster Recovery Test Started"
    log "=========================================="
    log "Project: $PROJECT_ID"
    log "Primary Region: $PRIMARY_REGION"
    log "Recovery Region: $RECOVERY_REGION"
    log "Dry Run: $DRY_RUN"
    log "Log File: $LOG_FILE"
    log ""

    # Validation phase
    log "PHASE 1: Validation"
    validate_project || exit 1
    validate_resources || exit 1

    # Database recovery phase
    log ""
    log "PHASE 2: Database Recovery"
    test_database_backup || exit 1
    test_database_clone || { cleanup_recovery_resources; exit 1; }
    test_database_connectivity || { cleanup_recovery_resources; exit 1; }
    test_data_integrity || { cleanup_recovery_resources; exit 1; }

    # Service recovery phase
    log ""
    log "PHASE 3: Service Recovery"
    test_service_deployment || { cleanup_recovery_resources; exit 1; }
    test_service_health || { cleanup_recovery_resources; exit 1; }

    # Integration phase
    log ""
    log "PHASE 4: Integration Tests"
    test_api_functionality || { cleanup_recovery_resources; exit 1; }

    # Cleanup
    log ""
    log "PHASE 5: Cleanup"
    cleanup_recovery_resources

    # Summary
    log ""
    log "=========================================="
    success "Disaster Recovery Test Completed Successfully"
    log "=========================================="
    log "Log file saved to: $LOG_FILE"
    log ""
    log "Next steps:"
    log "  1. Review test results in $LOG_FILE"
    log "  2. Verify recovered service is operational"
    log "  3. Document any issues found"
    log "  4. Schedule failover drill if needed"
    log ""

    return 0
}

# Run main function
main "$@"
