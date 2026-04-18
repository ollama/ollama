#!/bin/bash
# GCP Deployment Script for Ollama
# Builds, pushes, and deploys Ollama to Cloud Run with GCP Load Balancer

set -e

PROJECT_ID="project-131055855980"
REGION="us-central1"
SERVICE_NAME="ollama-api"
IMAGE_NAME="ollama"
VERSION="1.0.0"
GCR_IMAGE="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${VERSION}"

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================="
echo "Ollama GCP Deployment Script"
echo "==========================================${NC}"
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Service: $SERVICE_NAME"
echo "Image: $GCR_IMAGE"
echo ""

# Step 1: Build Docker image
echo -e "${YELLOW}[1/5] Building Docker image...${NC}"
docker build \
    -t $IMAGE_NAME:$VERSION \
    -t $IMAGE_NAME:latest \
    -f docker/Dockerfile \
    --build-arg PYTHON_VERSION=3.12 \
    --build-arg BASE_IMAGE=python:3.12-slim \
    . > /dev/null 2>&1 || {
    echo "  ✗ Docker build failed"
    exit 1
}
echo -e "${GREEN}  ✓ Docker image built successfully${NC}"

# Step 2: Tag for GCP Container Registry
echo -e "${YELLOW}[2/5] Tagging image for GCR...${NC}"
docker tag $IMAGE_NAME:$VERSION $GCR_IMAGE
docker tag $IMAGE_NAME:latest gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest
echo -e "${GREEN}  ✓ Image tagged${NC}"

# Step 3: Configure Docker authentication for GCP
echo -e "${YELLOW}[3/5] Configuring GCP authentication...${NC}"
gcloud auth configure-docker gcr.io --quiet
echo -e "${GREEN}  ✓ Authentication configured${NC}"

# Step 4: Push to GCP Container Registry
echo -e "${YELLOW}[4/5] Pushing image to GCR...${NC}"
docker push $GCR_IMAGE || {
    echo "  ✗ Docker push failed"
    exit 1
}
docker push gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest || true
echo -e "${GREEN}  ✓ Image pushed to GCR${NC}"

# Step 5: Deploy to Cloud Run
echo -e "${YELLOW}[5/5] Deploying to Cloud Run...${NC}"

# Get Firebase credentials from Secret Manager
FIREBASE_CREDS_MOUNT="/run/secrets/firebase-credentials"

gcloud run deploy $SERVICE_NAME \
    --image $GCR_IMAGE \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --timeout 600 \
    --max-instances 20 \
    --min-instances 1 \
    --port 8000 \
    --set-env-vars "\
ENVIRONMENT=production,\
FIREBASE_PROJECT_ID=project-131055855980,\
GCP_PROJECT_ID=project-131055855980,\
FIREBASE_ENABLED=true,\
PUBLIC_API_ENDPOINT=https://elevatediq.ai/ollama,\
GCP_LOAD_BALANCER_IP=0.0.0.0,\
LOG_LEVEL=info" \
    --set-secrets "FIREBASE_CREDENTIALS_PATH=${FIREBASE_CREDS_MOUNT}:firebase-service-account@latest" \
    --project=$PROJECT_ID \
    --quiet || {
    echo "  ✗ Cloud Run deployment failed"
    exit 1
}

echo -e "${GREEN}  ✓ Deployed to Cloud Run${NC}"

# Get Cloud Run service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
    --region $REGION \
    --project=$PROJECT_ID \
    --format='value(status.url)' 2>/dev/null || echo "")

echo ""
echo -e "${GREEN}=========================================="
echo "✅ Deployment Complete!"
echo "==========================================${NC}"
echo ""
echo "Cloud Run Service URL: ${SERVICE_URL:-not available}"
echo "Public Endpoint: https://elevatediq.ai/ollama"
echo ""
echo "Next steps:"
echo "1. Configure GCP Load Balancer to route traffic"
echo "2. Test health check: curl https://elevatediq.ai/ollama/health"
echo "3. Verify OAuth: curl -H 'Authorization: Bearer \$TOKEN' https://elevatediq.ai/ollama/api/v1/health"
echo ""
echo "To view logs:"
echo "  gcloud logging read \"resource.type=cloud_run_revision\" --limit 50 --format json"
echo ""
echo "To monitor metrics:"
echo "  gcloud monitoring dashboards list --project=$PROJECT_ID"
echo ""
