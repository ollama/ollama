#!/bin/bash
# Rollback Script - Emergency Recovery
# Purpose: Quickly rollback to previous production version
# Date: January 13, 2026
# Status: Critical recovery procedure

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ID="${GCP_PROJECT_ID:-ollama-prod}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="ollama"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ============================================================================
# FUNCTIONS
# ============================================================================

log_info() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

# ============================================================================
# EMERGENCY ROLLBACK
# ============================================================================

echo ""
echo "================================================================================"
echo "🚨 EMERGENCY ROLLBACK PROCEDURE"
echo "================================================================================"
echo ""

# Verify authorization
read -p "Are you sure you want to rollback production? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    log_info "Rollback cancelled"
    exit 0
fi

echo ""
log_warn "INITIATING EMERGENCY ROLLBACK"
echo "────────────────────────────────────────────────────────────────────────────"

# Step 1: Get previous revision
echo ""
log_info "Step 1: Identifying previous stable revision..."

# Get list of revisions
REVISIONS=$(gcloud run services describe "${SERVICE_NAME}" \
    --platform managed \
    --region "${REGION}" \
    --project "${PROJECT_ID}" \
    --format json | jq -r '.status.traffic[].revisionName' 2>/dev/null || echo "")

if [ -z "$REVISIONS" ]; then
    log_error "Could not retrieve revision history"
fi

log_info "Current revisions: $REVISIONS"

# Step 2: Route traffic back to previous revision
echo ""
log_info "Step 2: Rerouting all traffic to previous revision..."

gcloud run services update-traffic "${SERVICE_NAME}" \
    --to-revisions PREVIOUS=100 \
    --platform managed \
    --region "${REGION}" \
    --project "${PROJECT_ID}"

log_info "Traffic rerouted to previous revision (100%)"

# Step 3: Wait and verify
echo ""
log_info "Step 3: Waiting for DNS propagation..."
sleep 30

# Verify service is responding
log_info "Step 4: Verifying service is responding..."
MAX_ATTEMPTS=30
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
        --platform managed \
        --region "${REGION}" \
        --project "${PROJECT_ID}" \
        --format 'value(status.url)')

    if curl -f -s -m 5 "${SERVICE_URL}/api/v1/health" > /dev/null 2>&1; then
        log_info "✓ Service is responding normally"
        break
    fi

    ATTEMPT=$((ATTEMPT + 1))
    log_warn "Service not responding yet (attempt $ATTEMPT/$MAX_ATTEMPTS), waiting..."
    sleep 5
done

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    log_error "Service failed to respond after rollback"
fi

# Step 4: Verify error rate is normal
echo ""
log_info "Step 5: Checking error rate..."

ERROR_COUNT=$(gcloud logging read \
    "resource.labels.service_name=\"${SERVICE_NAME}\" AND severity>=ERROR" \
    --limit 100 \
    --format json \
    --project "${PROJECT_ID}" 2>/dev/null | grep -c "severity" || echo "0")

if [ "$ERROR_COUNT" -gt 5 ]; then
    log_warn "High error rate detected: $ERROR_COUNT errors"
else
    log_info "Error rate normal: $ERROR_COUNT recent errors"
fi

# Step 5: Notify team
echo ""
log_info "Step 6: Preparing notification..."

TIMESTAMP=$(date -u +'%Y-%m-%d %H:%M:%S UTC')
cat > "ROLLBACK_REPORT.md" << EOF
# Emergency Rollback Report

**Date**: ${TIMESTAMP}
**Service**: ${SERVICE_NAME}
**Region**: ${REGION}
**Status**: ✅ COMPLETE

## Rollback Summary

- ✅ Previous revision identified
- ✅ All traffic routed to previous revision
- ✅ Service verified responding
- ✅ Error rate checked

## Current State

- **Service URL**: ${SERVICE_URL}
- **Traffic**: 100% on previous revision
- **Error Count**: ${ERROR_COUNT}

## Action Items

1. [ ] Investigate the issue that caused the rollback
2. [ ] Review deployment logs
3. [ ] Fix identified issues
4. [ ] Test thoroughly before redeploying
5. [ ] Post-mortem with team

## Contact On-Call

If service is still not responding, contact the on-call engineer.

EOF

log_info "Rollback report: ROLLBACK_REPORT.md"

# ============================================================================
# COMPLETION
# ============================================================================

echo ""
echo "================================================================================"
echo "✅ EMERGENCY ROLLBACK - COMPLETE"
echo "================================================================================"
echo ""
echo "Status: Production rolled back to previous version"
echo "URL: ${SERVICE_URL}"
echo ""
echo "Next Steps:"
echo "  1. Verify service stability for 15-30 minutes"
echo "  2. Investigate root cause of deployment issue"
echo "  3. Apply fixes"
echo "  4. Test thoroughly"
echo "  5. Redeploy when ready"
echo ""
echo "Documentation: ROLLBACK_REPORT.md"
echo ""
