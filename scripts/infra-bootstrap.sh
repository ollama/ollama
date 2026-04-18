#!/usr/bin/env bash
# ==============================================================================
# Infra Bootstrap: Pre-flight Validation
# ==============================================================================
# This script validates pmo.yaml and tests the Terraform build configuration.

set -euo pipefail

# ANSI Color Codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TF_DIR="${PROJECT_ROOT}/docker/terraform/00-bootstrap"
PMO_FILE="${PROJECT_ROOT}/pmo.yaml"

log_info() { echo -e "${YELLOW}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# 0. Check Prerequisites
log_info "Checking prerequisites..."
PREREQS=("gcloud" "terraform" "grep")
for cmd in "${PREREQS[@]}"; do
    if ! command -v "$cmd" &> /dev/null; then
        log_error "Prerequisite '$cmd' not found. Please install it."
        exit 1
    fi
done

# 1. Validate pmo.yaml exists
if [[ ! -f "$PMO_FILE" ]]; then
    log_error "pmo.yaml not found at root"
    exit 1
fi

log_info "Validating 24-label schema in pmo.yaml..."

MANDATORY_LABELS=(
    "environment" "cost_center" "team" "managed_by"
    "created_by" "created_date" "lifecycle_state" "teardown_date" "retention_days"
    "product" "component" "tier" "compliance"
    "version" "stack" "backup_strategy" "monitoring_enabled"
    "budget_owner" "project_code" "monthly_budget_usd" "chargeback_unit"
    "git_repository" "git_branch" "auto_delete"
)

MISSING=0
# Quick check using grep (simulating yq if not installed)
for label in "${MANDATORY_LABELS[@]}"; do
    if ! grep -q "^${label}:" "$PMO_FILE"; then
        log_error "Missing label: $label"
        MISSING=$((MISSING + 1))
    fi
done

if [[ $MISSING -gt 0 ]]; then
    log_error "Found $MISSING missing mandatory labels. Bootstrap aborted."
    exit 1
fi

log_success "All 24 mandatory labels verified."

# 2. Extract labels to tfvars (Simplified extraction for demo)
log_info "Generating temporary tfvars for validation..."

CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null || echo "placeholder-project-id")
log_info "Target Project: ${CURRENT_PROJECT}"

# Construct labels map for terraform with sanitization
LABELS_JSON="{"
for label in "${MANDATORY_LABELS[@]}"; do
    VALUE=$(grep "^${label}:" "$PMO_FILE" | head -n1 | cut -d'"' -f2)
    # Sanitize value for GCP labels: only lowercase, digits, underscores, hyphens.
    # Replace all non-label-safe characters (including dots) with hyphens
    SANTOZED_VALUE=$(echo "$VALUE" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_-]/-/g')
    # Remove leading/trailing hyphens/underscores if they exist
    SANTOZED_VALUE=$(echo "$SANTOZED_VALUE" | sed 's/^[_-]*//;s/[_-]*$//')
    LABELS_JSON+="\"$label\": \"$SANTOZED_VALUE\","
done
LABELS_JSON="${LABELS_JSON%,}}" # Remove last comma and close brace

cat > "${TF_DIR}/validation.tfvars.json" <<EOF
{
  "project_id": "${CURRENT_PROJECT}",
  "project_labels": ${LABELS_JSON}
}
EOF

# 3. Test Terraform Build
log_info "Running Terraform initialization and plan..."

pushd "$TF_DIR" > /dev/null
terraform init

# Check if --apply flag is provided
APPLY=false
for arg in "$@"; do
    if [[ "$arg" == "--apply" ]]; then
        APPLY=true
    fi
done

if [[ "$APPLY" == "true" ]]; then
    log_info "Applying Terraform configuration..."
    terraform apply -var-file="validation.tfvars.json" -auto-approve
    log_success "Infrastructure bootstrap APPLIED successfully."
else
    terraform plan -var-file="validation.tfvars.json" -out=tfplan > /dev/null
    if [[ $? -eq 0 ]]; then
        log_success "Terraform configuration validated successfully."
        log_info "Clean up..."
        rm -f validation.tfvars.json tfplan
    else
        log_error "Terraform validation failed."
        rm -f validation.tfvars.json
        popd > /dev/null
        exit 1
    fi
fi

popd > /dev/null

log_success "Infrastructure bootstrap pre-flight checks PASSED."
echo "--------------------------------------------------------"
echo "Project 'ollama' is ready for Landing Zone onboarding."
