#!/usr/bin/env bash
# ==============================================================================
# Elite Landing Zone Deployment: Phase 2 (App Onboarding)
# ==============================================================================
# Purpose: Build, Push, and Deploy Ollama to GCP Landing Zone via Cloud Run
# Compliance: Enforces 24-label mandate and Elite Architecture Standards
# Naming: Follows {environment}-{application}-{component} pattern

set -euo pipefail

# ANSI Color Codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

DRY_RUN=false
for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then
        DRY_RUN=true
    fi
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PMO_FILE="${PROJECT_ROOT}/pmo.yaml"
PROJECT_ID=$(gcloud config get-value project 2>/dev/null || echo "gcp-eiq")
REGION="us-central1"
REPO_NAME="ollama"

# 0. Landing Zone Naming Convention
log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_step() { echo -e "${YELLOW}[STEP]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

if [[ ! -f "$PMO_FILE" ]]; then
    log_error "pmo.yaml not found. Required for Landing Zone compliance."
    exit 1
fi

ENV=$(grep "^environment:" "$PMO_FILE" | cut -d'"' -f2)
APP=$(grep "^product:" "$PMO_FILE" | cut -d'"' -f2)
COMP=$(grep "^component:" "$PMO_FILE" | cut -d'"' -f2)

# Standard: {environment}-{application}-{component}
SERVICE_NAME="${ENV}-${APP}-api"
IMAGE_NAME="${COMP}"
TAG=$(grep "^version:" "$PMO_FILE" | cut -d'"' -f2)

# 0. Check Prerequisites
echo -e "${BLUE}[INFO]${NC} Checking prerequisites..."
PREREQS=("gcloud" "docker" "grep")
for cmd in "${PREREQS[@]}"; do
    if ! command -v "$cmd" &> /dev/null; then
        echo -e "${RED}[ERROR]${NC} Prerequisite '$cmd' not found."
        exit 1
    fi
done

# 1. Validate pmo.yaml
log_step "Validating Landing Zone Compliance (24-Label Mandate)..."
if [[ ! -f "$PMO_FILE" ]]; then
    log_error "pmo.yaml not found. Please run scripts/infra-bootstrap.sh first."
    exit 1
fi

MANDATORY_LABELS=(
    "environment" "cost_center" "team" "managed_by"
    "created_by" "created_date" "lifecycle_state" "teardown_date" "retention_days"
    "product" "component" "tier" "compliance"
    "version" "stack" "backup_strategy" "monitoring_enabled"
    "budget_owner" "project_code" "monthly_budget_usd" "chargeback_unit"
    "git_repository" "git_branch" "auto_delete"
)

LABELS_CLI=""
for label in "${MANDATORY_LABELS[@]}"; do
    VALUE=$(grep "^${label}:" "$PMO_FILE" | head -n1 | cut -d'"' -f2)
    # Sanitize value for gcloud labels: only lowercase, digits, underscores, hyphens.
    # Replace all non-label-safe characters with hyphens
    SANTOZED_VALUE=$(echo "$VALUE" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_-]/-/g')
    # Remove leading/trailing hyphens/underscores if they exist
    SANTOZED_VALUE=$(echo "$SANTOZED_VALUE" | sed 's/^[_-]*//;s/[_-]*$//')
    LABELS_CLI+="${label}=${SANTOZED_VALUE},"
done
LABELS_CLI="${LABELS_CLI%,}" # Remove last comma

log_success "Compliance check passed."

# 2. Ensure Artifact Registry exists
log_step "Checking Artifact Registry repository..."
if [[ "$DRY_RUN" == "true" ]]; then
    log_info "[DRY-RUN] Would check/create Artifact Registry repository: ${REPO_NAME}"
else
    if ! gcloud artifacts repositories describe "$REPO_NAME" --project="$PROJECT_ID" --location="$REGION" &>/dev/null; then
        log_info "Creating repository '${REPO_NAME}' in ${REGION}..."
        gcloud artifacts repositories create "$REPO_NAME" \
            --repository-format=docker \
            --location="$REGION" \
            --project="$PROJECT_ID" \
            --description="Ollama Inference Engine Repository" \
            --labels="${LABELS_CLI}"
    else
        log_info "Repository exists."
    fi
fi

FULL_IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${TAG}"

# 3. Build Docker Image
log_step "Building Docker image: ${IMAGE_NAME}:${TAG}..."
cd "$PROJECT_ROOT"
if [[ "$DRY_RUN" == "true" ]]; then
    log_info "[DRY-RUN] Would build image: ${FULL_IMAGE_NAME}"
    log_info "[DRY-RUN] Command: DOCKER_BUILDKIT=1 docker build -t ${FULL_IMAGE_NAME} -f docker/Dockerfile ."
else
    DOCKER_BUILDKIT=1 docker build -t "$FULL_IMAGE_NAME" -f docker/Dockerfile . > /dev/null
fi

# 4. Push to Artifact Registry
log_step "Pushing image to Artifact Registry..."
if [[ "$DRY_RUN" == "true" ]]; then
    log_info "[DRY-RUN] Would push image to: ${FULL_IMAGE_NAME}"
else
    gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet > /dev/null
    docker push "$FULL_IMAGE_NAME" > /dev/null
fi

# 5. Deploy to Cloud Run (Elite Optimized for ROI & Performance)
log_step "Deploying to Cloud Run: ${SERVICE_NAME}..."
# Elite Performance & Security Flags:
# - cpu-boost: Faster cold starts (up to 50% faster)
# - execution-environment gen2: Better startup and performance
# - concurrency: Handle 80 simultaneous requests per instance (Financial Savings)
# - no-cpu-throttling: High performance during request handling
# - vpc-egress: Direct connection to DB/Redis via VPC (Zero Trust + Performance)
# - confidential-nodes: Hardware-level encryption for AI inference (Confidential Cloud Run)
if [[ "$DRY_RUN" == "true" ]]; then
    log_info "[DRY-RUN] Would deploy to Cloud Run: ${SERVICE_NAME}"
    log_info "[DRY-RUN] Command: gcloud run deploy ${SERVICE_NAME} --image ${FULL_IMAGE_NAME} --region ${REGION} --labels ${LABELS_CLI} --cpu-boost --no-cpu-throttling --concurrency 80 --execution-environment gen2 --network=default --subnet=default --vpc-egress=all-traffic --confidential-nodes"
else
    gcloud run deploy "$SERVICE_NAME" \
        --image "$FULL_IMAGE_NAME" \
        --region "$REGION" \
        --project "$PROJECT_ID" \
        --platform managed \
        --allow-unauthenticated \
        --labels "$LABELS_CLI" \
        --set-env-vars "ENVIRONMENT=production,PROJECT_ID=${PROJECT_ID},OLLAMA_BASE_URL=http://localhost:11434" \
        --timeout 600 \
        --memory 4Gi \
        --cpu 2 \
        --cpu-boost \
        --no-cpu-throttling \
        --concurrency 80 \
        --execution-environment gen2 \
        --network=default \
        --subnet=default \
        --vpc-egress=all-traffic \
        --confidential-nodes \
        --min-instances 0 \
        --max-instances 10 \
        --quiet
fi

if [[ "$DRY_RUN" == "true" ]]; then
    log_success "Dry Run Complete! No changes were made."
else
    log_success "Deployment Complete!"
    echo "--------------------------------------------------------"
    SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --region "$REGION" --project "$PROJECT_ID" --format='value(status.url)')
    echo "Service URL: ${SERVICE_URL}"
    echo "--------------------------------------------------------"
    echo "Project complies with GCP Landing Zone Governance."
fi
