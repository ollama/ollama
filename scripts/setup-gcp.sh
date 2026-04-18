#!/bin/bash
#===============================================================================
# GCP Infrastructure Setup Automation
# Sets up GCS backup bucket, service account, and Load Balancer
#===============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -f "${PROJECT_ROOT}/scripts/host-profile.sh" ]; then
    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/scripts/host-profile.sh"
    load_host_profile "${PROJECT_ROOT}"
fi

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
PROJECT_ID="${GCP_PROJECT:-$(gcloud config get-value project)}"
REGION="${GCP_REGION:-us-central1}"
ZONE="${GCP_ZONE:-us-central1-a}"
BUCKET_NAME="${GCS_BUCKET:-ollama-backups-${PROJECT_ID}}"
SERVICE_ACCOUNT_NAME="ollama-backup-sa"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
KEY_FILE="../secrets/gcp-service-account.json"

# Load Balancer configuration
LB_NAME="ollama-lb"
BACKEND_SERVICE_NAME="ollama-backend"
HEALTH_CHECK_NAME="ollama-health-check"
URL_MAP_NAME="ollama-url-map"
TARGET_PROXY_NAME="ollama-target-proxy"
FORWARDING_RULE_NAME="ollama-forwarding-rule"
SSL_CERT_NAME="ollama-ssl-cert"
DOMAIN="elevatediq.ai"
BACKEND_HOST="${BACKEND_HOST:-localhost}"
BACKEND_PORT="11000"

print_header() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}\n"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

#===============================================================================
# Validate Prerequisites
#===============================================================================

print_header "VALIDATING PREREQUISITES"

# Check gcloud authentication
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    print_error "Not authenticated with GCP. Run: gcloud auth login"
    exit 1
fi
print_success "GCP authentication valid"

# Check project
if [ -z "$PROJECT_ID" ]; then
    print_error "No GCP project set. Run: gcloud config set project PROJECT_ID"
    exit 1
fi
print_info "Using project: $PROJECT_ID"
print_info "Region: $REGION"

#===============================================================================
# Setup GCS Backup
#===============================================================================

print_header "SETTING UP GCS BACKUP"

# Create GCS bucket
print_info "Creating GCS bucket: gs://${BUCKET_NAME}"
if gsutil ls -b "gs://${BUCKET_NAME}" >/dev/null 2>&1; then
    print_warning "Bucket already exists"
else
    gsutil mb -p "$PROJECT_ID" -l "$REGION" "gs://${BUCKET_NAME}"
    print_success "Bucket created"
fi

# Enable versioning
print_info "Enabling versioning on bucket"
gsutil versioning set on "gs://${BUCKET_NAME}"
print_success "Versioning enabled"

# Set lifecycle policy (delete after 90 days)
print_info "Setting lifecycle policy"
cat > /tmp/lifecycle.json <<EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {"age": 90}
      }
    ]
  }
}
EOF
gsutil lifecycle set /tmp/lifecycle.json "gs://${BUCKET_NAME}"
print_success "Lifecycle policy set (90 day retention)"

#===============================================================================
# Setup Service Account
#===============================================================================

print_header "SETTING UP SERVICE ACCOUNT"

# Create service account
print_info "Creating service account: ${SERVICE_ACCOUNT_NAME}"
if gcloud iam service-accounts describe "$SERVICE_ACCOUNT_EMAIL" >/dev/null 2>&1; then
    print_warning "Service account already exists"
else
    gcloud iam service-accounts create "$SERVICE_ACCOUNT_NAME" \
        --display-name="Ollama Backup Service Account" \
        --description="Service account for Ollama backups to GCS"
    print_success "Service account created"
fi

# Grant bucket permissions
print_info "Granting Storage Object Admin role"
gsutil iam ch "serviceAccount:${SERVICE_ACCOUNT_EMAIL}:roles/storage.objectAdmin" \
    "gs://${BUCKET_NAME}"
print_success "Permissions granted"

# Create key file
print_info "Generating service account key"
mkdir -p "$(dirname "$KEY_FILE")"
if [ -f "$KEY_FILE" ]; then
    print_warning "Key file already exists at $KEY_FILE"
    read -p "Overwrite? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Skipping key generation"
    else
        gcloud iam service-accounts keys create "$KEY_FILE" \
            --iam-account="$SERVICE_ACCOUNT_EMAIL"
        chmod 600 "$KEY_FILE"
        print_success "Key file created at $KEY_FILE"
    fi
else
    gcloud iam service-accounts keys create "$KEY_FILE" \
        --iam-account="$SERVICE_ACCOUNT_EMAIL"
    chmod 600 "$KEY_FILE"
    print_success "Key file created at $KEY_FILE"
fi

#===============================================================================
# Update Environment Configuration
#===============================================================================

print_header "UPDATING CONFIGURATION"

# Update .env.production
if grep -q "GCS_BUCKET=" ../.env.production 2>/dev/null; then
    sed -i "s|GCS_BUCKET=.*|GCS_BUCKET=${BUCKET_NAME}|" ../.env.production
    print_success "Updated GCS_BUCKET in .env.production"
else
    echo "GCS_BUCKET=${BUCKET_NAME}" >> ../.env.production
    print_success "Added GCS_BUCKET to .env.production"
fi

if grep -q "GCS_SERVICE_ACCOUNT_KEY=" ../.env.production 2>/dev/null; then
    sed -i "s|GCS_SERVICE_ACCOUNT_KEY=.*|GCS_SERVICE_ACCOUNT_KEY=${KEY_FILE}|" ../.env.production
else
    echo "GCS_SERVICE_ACCOUNT_KEY=${KEY_FILE}" >> ../.env.production
    print_success "Added GCS_SERVICE_ACCOUNT_KEY to .env.production"
fi

#===============================================================================
# Setup GCP Load Balancer
#===============================================================================

print_header "SETTING UP GCP LOAD BALANCER"

print_warning "Note: GCP Load Balancer requires a static external IP and will incur charges"
print_warning "This setup creates a global HTTPS load balancer"
read -p "Continue with Load Balancer setup? (y/N): " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "Skipping Load Balancer setup"
    print_info "You can run this script again later to complete the setup"
else
    # Reserve static IP
    print_info "Reserving global static IP"
    if gcloud compute addresses describe "${LB_NAME}-ip" --global >/dev/null 2>&1; then
        LB_IP=$(gcloud compute addresses describe "${LB_NAME}-ip" --global --format="value(address)")
        print_warning "Static IP already exists: $LB_IP"
    else
        gcloud compute addresses create "${LB_NAME}-ip" --global
        LB_IP=$(gcloud compute addresses describe "${LB_NAME}-ip" --global --format="value(address)")
        print_success "Reserved static IP: $LB_IP"
    fi

    print_info ""
    print_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    print_info "IMPORTANT: Add DNS A record for your domain:"
    print_info "  Domain:  ${DOMAIN}"
    print_info "  Record:  ollama.${DOMAIN}"
    print_info "  Type:    A"
    print_info "  Value:   ${LB_IP}"
    print_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    print_info ""

    read -p "Have you added the DNS record? (y/N): " -n 1 -r
    echo

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "DNS record not added. Please add it and re-run this script"
        print_info "You can continue with the rest of the setup manually"
        exit 0
    fi

    # Create health check
    print_info "Creating health check"
    if gcloud compute health-checks describe "$HEALTH_CHECK_NAME" >/dev/null 2>&1; then
        print_warning "Health check already exists"
    else
        gcloud compute health-checks create http "$HEALTH_CHECK_NAME" \
            --port="$BACKEND_PORT" \
            --request-path="/health" \
            --check-interval=30s \
            --timeout=10s \
            --unhealthy-threshold=3 \
            --healthy-threshold=2
        print_success "Health check created"
    fi

    # Create backend service
    print_info "Creating backend service"
    if gcloud compute backend-services describe "$BACKEND_SERVICE_NAME" --global >/dev/null 2>&1; then
        print_warning "Backend service already exists"
    else
        gcloud compute backend-services create "$BACKEND_SERVICE_NAME" \
            --protocol=HTTP \
            --health-checks="$HEALTH_CHECK_NAME" \
            --global \
            --port-name=http \
            --timeout=30s
        print_success "Backend service created"
    fi

    print_warning ""
    print_warning "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    print_warning "MANUAL STEP REQUIRED:"
    print_warning "You need to add your backend instance to the backend service."
    print_warning ""
    print_warning "For a serverless NEG (recommended for external backends):"
    print_warning "1. Create a Network Endpoint Group (NEG)"
    print_warning "2. Add it to the backend service"
    print_warning ""
    print_warning "For instance groups:"
    print_warning "Run: gcloud compute backend-services add-backend $BACKEND_SERVICE_NAME \\"
    print_warning "  --instance-group=YOUR_INSTANCE_GROUP \\"
    print_warning "  --instance-group-zone=$ZONE \\"
    print_warning "  --global"
    print_warning "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    print_warning ""

    # Create URL map
    print_info "Creating URL map"
    if gcloud compute url-maps describe "$URL_MAP_NAME" >/dev/null 2>&1; then
        print_warning "URL map already exists"
    else
        gcloud compute url-maps create "$URL_MAP_NAME" \
            --default-service="$BACKEND_SERVICE_NAME"
        print_success "URL map created"
    fi

    # Create managed SSL certificate
    print_info "Creating managed SSL certificate"
    if gcloud compute ssl-certificates describe "$SSL_CERT_NAME" >/dev/null 2>&1; then
        print_warning "SSL certificate already exists"
    else
        gcloud compute ssl-certificates create "$SSL_CERT_NAME" \
            --domains="ollama.${DOMAIN}" \
            --global
        print_success "SSL certificate created (provisioning may take up to 20 minutes)"
    fi

    # Create target HTTPS proxy
    print_info "Creating target HTTPS proxy"
    if gcloud compute target-https-proxies describe "$TARGET_PROXY_NAME" >/dev/null 2>&1; then
        print_warning "Target proxy already exists"
    else
        gcloud compute target-https-proxies create "$TARGET_PROXY_NAME" \
            --url-map="$URL_MAP_NAME" \
            --ssl-certificates="$SSL_CERT_NAME"
        print_success "Target proxy created"
    fi

    # Create forwarding rule
    print_info "Creating global forwarding rule"
    if gcloud compute forwarding-rules describe "$FORWARDING_RULE_NAME" --global >/dev/null 2>&1; then
        print_warning "Forwarding rule already exists"
    else
        gcloud compute forwarding-rules create "$FORWARDING_RULE_NAME" \
            --global \
            --target-https-proxy="$TARGET_PROXY_NAME" \
            --address="${LB_NAME}-ip" \
            --ports=443
        print_success "Forwarding rule created"
    fi

    print_success "Load Balancer setup complete!"
fi

#===============================================================================
# Summary
#===============================================================================

print_header "SETUP COMPLETE"

echo -e "${GREEN}✓ GCS Backup Configuration:${NC}"
echo -e "  Bucket: gs://${BUCKET_NAME}"
echo -e "  Service Account: ${SERVICE_ACCOUNT_EMAIL}"
echo -e "  Key File: ${KEY_FILE}"
echo -e "  Versioning: Enabled"
echo -e "  Lifecycle: 90 day retention"

if [ -n "$LB_IP" ]; then
    echo -e ""
    echo -e "${GREEN}✓ Load Balancer Configuration:${NC}"
    echo -e "  Public IP: ${LB_IP}"
    echo -e "  Domain: ollama.${DOMAIN}"
    echo -e "  Backend: ${BACKEND_HOST}:${BACKEND_PORT}"
    echo -e "  SSL: Managed certificate (provisioning...)"
    echo -e ""
    echo -e "${YELLOW}⚠ Next Steps:${NC}"
    echo -e "  1. Verify DNS record: dig ollama.${DOMAIN}"
    echo -e "  2. Wait for SSL certificate provisioning (up to 20 minutes)"
    echo -e "  3. Add backend to backend service (see manual step above)"
    echo -e "  4. Test: curl https://ollama.${DOMAIN}/health"
fi

echo -e ""
echo -e "${BLUE}To test backup:${NC}"
echo -e "  cd ${PROJECT_ROOT}"
echo -e "  source .env.production"
echo -e "  ./scripts/sync-to-gcs.sh"

echo -e ""
print_success "GCP infrastructure setup complete! 🚀"
