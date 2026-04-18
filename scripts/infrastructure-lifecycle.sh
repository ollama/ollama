#!/bin/bash
# =============================================================================
# COMPLETE INFRASTRUCTURE LIFECYCLE MANAGEMENT
# Full automation for deploy, teardown, and restore with disaster recovery
# =============================================================================

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENVIRONMENT="${1:-dev}"
ACTION="${2:-deploy}"
DRY_RUN="${3:-false}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_ROOT}/logs"
BACKUP_DIR="${PROJECT_ROOT}/backups"

# GCP Configuration
GCP_PROJECT_ID="${GCP_PROJECT_ID:-}"
GCP_REGION="${GCP_REGION:-us-central1}"
TF_STATE_BUCKET="${TF_STATE_BUCKET:-${GCP_PROJECT_ID}-ollama-tf-state}"

# Ensure directories exist
mkdir -p "$LOG_DIR" "$BACKUP_DIR"

# Logging setup
LOG_FILE="$LOG_DIR/infra-lifecycle-${ENVIRONMENT}-${ACTION}-${TIMESTAMP}.log"
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*"
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

# Execution wrapper with dry-run support
execute() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY-RUN] Would execute: $*"
    else
        log "Executing: $*"
        "$@"
    fi
}

# =============================================================================
# VALIDATION & PREREQUISITES
# =============================================================================

validate_prerequisites() {
    log "Validating prerequisites..."

    local missing_tools=()

    # Check required tools
    for tool in gcloud terraform docker docker-compose git python3 jq; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done

    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi

    # Validate GCP credentials
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        error "No active GCP credentials found. Please run: gcloud auth login"
        exit 1
    fi

    # Set default project
    if [[ -z "$GCP_PROJECT_ID" ]]; then
        GCP_PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
        if [[ -z "$GCP_PROJECT_ID" ]]; then
            error "GCP_PROJECT_ID not set and no default project configured"
            exit 1
        fi
    fi

    success "Prerequisites validated"
}

validate_environment() {
    log "Validating environment configuration..."

    if [[ ! -f "$PROJECT_ROOT/config/${ENVIRONMENT}.yaml" ]]; then
        error "Environment config not found: config/${ENVIRONMENT}.yaml"
        exit 1
    fi

    success "Environment ${ENVIRONMENT} validated"
}

# =============================================================================
# BACKUP & DISASTER RECOVERY
# =============================================================================

backup_database() {
    log "📦 Backing up PostgreSQL database..."

    local backup_file="$BACKUP_DIR/postgres-${ENVIRONMENT}-${TIMESTAMP}.sql.gz"

    execute bash "$SCRIPT_DIR/backup-postgres.sh" \
        --environment="$ENVIRONMENT" \
        --output="$backup_file"

    success "Database backed up to: $backup_file"
    echo "$backup_file"
}

backup_vectors() {
    log "📦 Backing up Qdrant vectors..."

    local backup_file="$BACKUP_DIR/qdrant-${ENVIRONMENT}-${TIMESTAMP}.tar.gz"

    execute bash "$SCRIPT_DIR/backup-qdrant.sh" \
        --environment="$ENVIRONMENT" \
        --output="$backup_file"

    success "Vectors backed up to: $backup_file"
    echo "$backup_file"
}

backup_storage() {
    log "📦 Backing up GCS storage..."

    local backup_bucket="gs://${GCP_PROJECT_ID}-ollama-backup-${ENVIRONMENT}"

    execute gcloud storage buckets create "$backup_bucket" \
        --location="$GCP_REGION" \
        --uniform-bucket-level-access \
        2>/dev/null || true

    execute bash "$SCRIPT_DIR/gcs-sync.sh" \
        --backup \
        --environment="$ENVIRONMENT" \
        --destination="$backup_bucket"

    success "Storage backed up to: $backup_bucket"
}

backup_terraform_state() {
    log "📦 Backing up Terraform state..."

    local state_backup="$BACKUP_DIR/terraform-state-${ENVIRONMENT}-${TIMESTAMP}.tar.gz"

    if [[ -d "$PROJECT_ROOT/docker/terraform" ]]; then
        tar -czf "$state_backup" \
            -C "$PROJECT_ROOT" docker/terraform/.terraform \
            docker/terraform/terraform.tfstate* \
            2>/dev/null || true

        success "Terraform state backed up to: $state_backup"
    fi
}

create_full_backup() {
    log "🔄 Creating full backup before operation..."

    backup_database
    backup_vectors
    backup_storage
    backup_terraform_state

    success "Full backup completed"
}

restore_database() {
    log "♻️ Restoring PostgreSQL database..."

    local backup_file="${1:}"

    if [[ -z "$backup_file" ]]; then
        # Find latest backup
        backup_file=$(find "$BACKUP_DIR" -name "postgres-${ENVIRONMENT}-*.sql.gz" -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
    fi

    if [[ ! -f "$backup_file" ]]; then
        error "No backup file found: $backup_file"
        return 1
    fi

    execute bash "$SCRIPT_DIR/restore-postgres.sh" \
        --environment="$ENVIRONMENT" \
        --backup="$backup_file"

    success "Database restored from: $backup_file"
}

restore_vectors() {
    log "♻️ Restoring Qdrant vectors..."

    local backup_file="${1:}"

    if [[ -z "$backup_file" ]]; then
        backup_file=$(find "$BACKUP_DIR" -name "qdrant-${ENVIRONMENT}-*.tar.gz" -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
    fi

    if [[ ! -f "$backup_file" ]]; then
        error "No backup file found: $backup_file"
        return 1
    fi

    execute bash "$SCRIPT_DIR/restore-qdrant.sh" \
        --environment="$ENVIRONMENT" \
        --backup="$backup_file"

    success "Vectors restored from: $backup_file"
}

# =============================================================================
# INFRASTRUCTURE OPERATIONS
# =============================================================================

terraform_init() {
    log "🔧 Initializing Terraform..."

    cd "$PROJECT_ROOT/docker/terraform"

    execute terraform init \
        -backend-config="bucket=${TF_STATE_BUCKET}" \
        -backend-config="prefix=${ENVIRONMENT}" \
        -upgrade

    success "Terraform initialized"
}

terraform_validate() {
    log "✓ Validating Terraform configuration..."

    cd "$PROJECT_ROOT/docker/terraform"

    execute terraform validate
    execute terraform fmt -check -recursive || execute terraform fmt -recursive

    success "Terraform configuration valid"
}

terraform_plan() {
    log "📋 Creating Terraform plan..."

    cd "$PROJECT_ROOT/docker/terraform"

    execute terraform plan \
        -var="environment=${ENVIRONMENT}" \
        -var="gcp_project_id=${GCP_PROJECT_ID}" \
        -var="gcp_region=${GCP_REGION}" \
        -out="tfplan-${ENVIRONMENT}"

    success "Terraform plan created"
}

terraform_apply() {
    log "🚀 Applying Terraform plan..."

    cd "$PROJECT_ROOT/docker/terraform"

    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY-RUN] Would apply Terraform changes"
    else
        execute terraform apply -auto-approve "tfplan-${ENVIRONMENT}"
    fi

    success "Terraform applied"
}

terraform_destroy() {
    log "💣 Destroying Terraform resources..."

    cd "$PROJECT_ROOT/docker/terraform"

    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY-RUN] Would destroy Terraform resources"
    else
        execute terraform destroy -auto-approve \
            -var="environment=${ENVIRONMENT}" \
            -var="gcp_project_id=${GCP_PROJECT_ID}" \
            -var="gcp_region=${GCP_REGION}"
    fi

    success "Terraform resources destroyed"
}

# =============================================================================
# DEPLOYMENT OPERATIONS
# =============================================================================

deploy_to_cloud_run() {
    log "📦 Deploying to Cloud Run..."

    local image="${1:-gcr.io/${GCP_PROJECT_ID}/ollama:latest}"

    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY-RUN] Would deploy: $image"
    else
        execute gcloud run deploy ollama-api \
            --image="$image" \
            --platform=managed \
            --region="$GCP_REGION" \
            --allow-unauthenticated \
            --set-env-vars="ENVIRONMENT=${ENVIRONMENT}" \
            --memory=4Gi \
            --cpu=2 \
            --timeout=3600 \
            --max-instances=10 \
            --labels="environment=${ENVIRONMENT},managed-by=terraform"
    fi

    success "Cloud Run deployment completed"
}

setup_load_balancer() {
    log "⚙️ Setting up Cloud Load Balancer..."

    execute bash "$SCRIPT_DIR/setup-gcp-ext-lb.sh" \
        --project="$GCP_PROJECT_ID" \
        --region="$GCP_REGION" \
        --environment="$ENVIRONMENT"

    success "Load Balancer configured"
}

setup_monitoring() {
    log "📊 Setting up monitoring and logging..."

    execute bash "$SCRIPT_DIR/setup-monitoring.sh" \
        --project="$GCP_PROJECT_ID" \
        --environment="$ENVIRONMENT"

    success "Monitoring configured"
}

setup_backups() {
    log "🔄 Setting up automated backups..."

    execute bash "$SCRIPT_DIR/setup-cron-backup.sh" \
        --environment="$ENVIRONMENT" \
        --frequency="daily"

    success "Backup cron jobs configured"
}

# =============================================================================
# VALIDATION & HEALTH CHECKS
# =============================================================================

health_check() {
    log "🏥 Running health checks..."

    execute bash "$SCRIPT_DIR/verify-production-health.sh" \
        --environment="$ENVIRONMENT" \
        --timeout=300

    success "Health checks passed"
}

validate_deployment() {
    log "✓ Validating deployment..."

    # Check Cloud Run service
    if ! gcloud run services describe ollama-api \
        --platform=managed \
        --region="$GCP_REGION" &>/dev/null; then
        error "Cloud Run service not found"
        return 1
    fi

    # Check database connectivity
    python3 "$SCRIPT_DIR/test_server.py" || return 1

    # Check API endpoints
    local api_url="https://elevatediq.ai/ollama"
    if ! curl -sf "$api_url/api/v1/health" >/dev/null; then
        error "API health check failed: $api_url"
        return 1
    fi

    success "Deployment validation passed"
}

# =============================================================================
# TEARDOWN OPERATIONS
# =============================================================================

cleanup_cloud_run() {
    log "Cleaning up Cloud Run services..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY-RUN] Would delete Cloud Run service"
    else
        gcloud run services delete ollama-api \
            --platform=managed \
            --region="$GCP_REGION" \
            --quiet || warning "Cloud Run service deletion failed or not found"
    fi
}

cleanup_load_balancer() {
    log "Cleaning up Load Balancer resources..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY-RUN] Would delete Load Balancer resources"
    else
        gcloud compute forwarding-rules delete ollama-https-lb \
            --global --quiet 2>/dev/null || warning "Forwarding rule not found"

        gcloud compute target-https-proxies delete ollama-https-proxy \
            --quiet 2>/dev/null || warning "HTTPS proxy not found"

        gcloud compute backend-buckets delete ollama-backend \
            --quiet 2>/dev/null || warning "Backend bucket not found"
    fi
}

cleanup_storage() {
    log "Cleaning up Cloud Storage..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY-RUN] Would delete Cloud Storage buckets"
    else
        gsutil -m rm -r "gs://${GCP_PROJECT_ID}-ollama-${ENVIRONMENT}-"* 2>/dev/null || warning "Storage cleanup failed or buckets not found"
    fi
}

cleanup_dns() {
    log "Cleaning up DNS entries..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY-RUN] Would delete DNS entries"
    else
        gcloud dns record-sets delete "ollama.${ENVIRONMENT}.elevatediq.ai." \
            --rrdatas="0.0.0.0" \
            --ttl=300 \
            --type=A \
            --zone=elevatediq \
            --quiet 2>/dev/null || warning "DNS record not found"
    fi
}

# =============================================================================
# MAIN WORKFLOW ORCHESTRATION
# =============================================================================

deploy_full() {
    log "🚀 STARTING FULL DEPLOYMENT (${ENVIRONMENT})"

    validate_prerequisites
    validate_environment
    create_full_backup
    terraform_init
    terraform_validate
    terraform_plan
    terraform_apply
    deploy_to_cloud_run
    setup_load_balancer
    setup_monitoring
    setup_backups
    health_check
    validate_deployment

    success "✅ DEPLOYMENT COMPLETE"
}

teardown_full() {
    log "💣 STARTING FULL TEARDOWN (${ENVIRONMENT})"

    validate_prerequisites
    validate_environment
    create_full_backup
    cleanup_cloud_run
    cleanup_load_balancer
    cleanup_dns
    cleanup_storage
    terraform_destroy

    success "✅ TEARDOWN COMPLETE"
}

restore_full() {
    log "♻️ STARTING FULL RESTORATION (${ENVIRONMENT})"

    validate_prerequisites
    validate_environment
    terraform_init
    terraform_validate
    terraform_plan
    terraform_apply
    deploy_to_cloud_run
    setup_load_balancer
    restore_database
    restore_vectors
    setup_monitoring
    health_check
    validate_deployment

    success "✅ RESTORATION COMPLETE"
}

# =============================================================================
# USAGE & MAIN EXECUTION
# =============================================================================

usage() {
    cat << EOF
Usage: $0 <environment> <action> [options]

Arguments:
  environment     dev|staging|prod (default: dev)
  action          deploy|teardown|restore|full-cycle (default: deploy)

Options:
  --dry-run       Show what would be executed without making changes

Examples:
  # Deploy to development
  $0 dev deploy

  # Dry-run teardown of staging
  $0 staging teardown --dry-run

  # Restore production
  $0 prod restore

  # Full cycle (deploy -> test -> teardown -> restore)
  $0 prod full-cycle

EOF
    exit 0
}

main() {
    if [[ "$ENVIRONMENT" == "--help" ]] || [[ "$ENVIRONMENT" == "-h" ]]; then
        usage
    fi

    log "╔════════════════════════════════════════════════════════════════╗"
    log "║          INFRASTRUCTURE LIFECYCLE MANAGEMENT                   ║"
    log "╚════════════════════════════════════════════════════════════════╝"
    log "Environment: $ENVIRONMENT"
    log "Action: $ACTION"
    log "Dry Run: $DRY_RUN"
    log "Log File: $LOG_FILE"

    case "$ACTION" in
        deploy)
            deploy_full
            ;;
        teardown)
            teardown_full
            ;;
        restore)
            restore_full
            ;;
        full-cycle)
            log "🔄 FULL CYCLE: Deploy → Validate → Teardown → Restore"
            deploy_full
            log "Sleeping 30s before teardown..."
            sleep 30
            teardown_full
            log "Sleeping 30s before restore..."
            sleep 30
            restore_full
            ;;
        *)
            error "Unknown action: $ACTION"
            usage
            exit 1
            ;;
    esac
}

main "$@"
