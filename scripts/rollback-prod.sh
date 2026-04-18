#!/bin/bash
# Rollback script for production deployment
# Emergency rollback procedure for Ollama production deployments

set -e

echo "🔄 Initiating Production Rollback..."
echo ""

# ============================================================
# Configuration
# ============================================================
CLUSTER="prod-gke"
REGION="us-central1"
NAMESPACE="production"
DEPLOYMENT="ollama-api"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ============================================================
# Helper Functions
# ============================================================
print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# ============================================================
# Rollback Procedure
# ============================================================
echo "Cluster: $CLUSTER"
echo "Namespace: $NAMESPACE"
echo "Deployment: $DEPLOYMENT"
echo ""

# Get current cluster context
current_cluster=$(kubectl config current-context 2>/dev/null || echo "unknown")
if [[ ! "$current_cluster" =~ "$CLUSTER" ]]; then
    echo -n "🔗 Connecting to cluster $CLUSTER ... "
    gcloud container clusters get-credentials $CLUSTER --region=$REGION > /dev/null 2>&1
    print_success "Connected"
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "Current Deployment Status"
echo "════════════════════════════════════════════════════════════"
echo ""

# Get deployment info
echo "📋 Current Deployment:"
kubectl get deployment $DEPLOYMENT -n $NAMESPACE -o wide || print_error "Deployment not found"

echo ""
echo "📋 Current Rollout Status:"
kubectl rollout history deployment/$DEPLOYMENT -n $NAMESPACE || print_error "No rollout history found"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "Rollback Options"
echo "════════════════════════════════════════════════════════════"
echo ""

echo "1. Rollback to previous revision (recommended)"
echo "2. Rollback to specific revision (advanced)"
echo "3. Cancel current rollout (if in progress)"
echo "4. Exit without rollback"
echo ""

read -p "Select option (1-4): " option

case $option in
    1)
        echo ""
        echo "🔄 Rolling back to previous revision..."

        # Perform rollback
        kubectl rollout undo deployment/$DEPLOYMENT -n $NAMESPACE

        # Wait for rollout to complete
        echo "⏳ Waiting for rollback to complete (timeout: 5 minutes)..."
        kubectl rollout status deployment/$DEPLOYMENT -n $NAMESPACE --timeout=5m

        # Verify rollback
        echo ""
        echo "✅ Rollback Status:"
        kubectl get deployment $DEPLOYMENT -n $NAMESPACE -o wide
        kubectl get pods -n $NAMESPACE -l app=$DEPLOYMENT

        print_success "Rollback completed successfully"

        # Log the rollback
        gcloud logging write ollama-rollbacks \
            "Manual rollback to previous revision triggered" \
            --severity=WARNING \
            --resource=global

        echo ""
        echo "📝 Next Steps:"
        echo "   1. Monitor error rates and metrics"
        echo "   2. Investigate root cause of deployment failure"
        echo "   3. Create incident ticket with findings"
        echo "   4. Fix issues in code and create PR"
        echo ""
        ;;

    2)
        echo ""
        echo "📋 Available Revisions:"
        kubectl rollout history deployment/$DEPLOYMENT -n $NAMESPACE

        echo ""
        read -p "Enter revision number to rollback to: " revision

        if [ -z "$revision" ]; then
            print_error "No revision selected"
            exit 1
        fi

        echo "🔄 Rolling back to revision $revision..."

        kubectl rollout undo deployment/$DEPLOYMENT -n $NAMESPACE --to-revision="$revision"

        echo "⏳ Waiting for rollback to complete (timeout: 5 minutes)..."
        kubectl rollout status deployment/$DEPLOYMENT -n $NAMESPACE --timeout=5m

        print_success "Rollback to revision $revision completed"

        gcloud logging write ollama-rollbacks \
            "Manual rollback to revision $revision triggered" \
            --severity=WARNING \
            --resource=global
        ;;

    3)
        echo ""
        echo "⏹️  Canceling current rollout..."

        kubectl rollout pause deployment/$DEPLOYMENT -n $NAMESPACE

        print_success "Rollout paused"
        print_warning "Note: Some pods may still be updating. Resume or undo as needed."
        ;;

    4)
        print_warning "Rollback cancelled by user"
        exit 0
        ;;

    *)
        print_error "Invalid option"
        exit 1
        ;;
esac

echo ""
echo "════════════════════════════════════════════════════════════"
echo "Post-Rollback Verification"
echo "════════════════════════════════════════════════════════════"
echo ""

# Check pod status
echo "Pod Status:"
kubectl get pods -n $NAMESPACE -l app=$DEPLOYMENT

echo ""
echo "Deployment Status:"
kubectl get deployment $DEPLOYMENT -n $NAMESPACE -o wide

echo ""
echo "Recent Logs:"
kubectl logs -n $NAMESPACE -l app=$DEPLOYMENT --tail=20 --all-containers=true || true

echo ""
echo "════════════════════════════════════════════════════════════"
echo "Metrics Check"
echo "════════════════════════════════════════════════════════════"
echo ""

# Check metrics if available
echo "Checking recent metrics..."
gcloud monitoring timeseries list \
    --filter='metric.type="custom.googleapis.com/ollama/api_error_rate"' \
    --limit=1 \
    --format='value(points[0].value)' 2>/dev/null || print_warning "Metrics not available"

echo ""
print_success "Rollback procedure complete"
echo ""
echo "📞 Support:"
echo "   • Review error logs: kubectl logs -n $NAMESPACE -l app=$DEPLOYMENT"
echo "   • Check metrics: gcloud monitoring timeseries list"
echo "   • Create incident in issue tracker"
echo "   • Post in #deployments Slack channel"
