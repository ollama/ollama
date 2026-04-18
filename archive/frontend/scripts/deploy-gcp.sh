#!/bin/bash
# GCP Cloud Run Deployment Script
# Deploys frontend to GCP with Landing Zone compliance

set -e

echo "🚀 Deploying Ollama Frontend to GCP Cloud Run"
echo "==============================================="

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-""}
REGION=${GCP_REGION:-"us-central1"}
SERVICE_NAME=${SERVICE_NAME:-"ollama-frontend"}
ENVIRONMENT=${ENVIRONMENT:-"production"}

# Validate configuration
if [ -z "$PROJECT_ID" ]; then
    echo "❌ Error: GCP_PROJECT_ID not set"
    echo "   Set with: export GCP_PROJECT_ID=your-project-id"
    exit 1
fi

if [ -z "$NEXT_PUBLIC_FIREBASE_API_KEY" ]; then
    echo "❌ Error: NEXT_PUBLIC_FIREBASE_API_KEY not set"
    echo "   Set Firebase environment variables before deploying"
    exit 1
fi

echo "✅ Project ID: $PROJECT_ID"
echo "✅ Region: $REGION"
echo "✅ Service: $SERVICE_NAME"
echo "✅ Environment: $ENVIRONMENT"

# Build and push Docker image
echo ""
echo "🐳 Building Docker image..."
IMAGE_TAG="gcr.io/$PROJECT_ID/$SERVICE_NAME:$ENVIRONMENT"
docker build -t "$IMAGE_TAG" .

echo ""
echo "📤 Pushing image to GCR..."
docker push "$IMAGE_TAG"

# Deploy to Cloud Run
echo ""
echo "🚢 Deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE_TAG" \
  --platform managed \
  --region "$REGION" \
  --allow-unauthenticated \
  --min-instances 1 \
  --max-instances 10 \
  --memory 512Mi \
  --cpu 1 \
  --timeout 60 \
  --set-env-vars "NEXT_PUBLIC_API_URL=https://elevatediq.ai/ollama" \
  --set-env-vars "NEXT_PUBLIC_FIREBASE_API_KEY=$NEXT_PUBLIC_FIREBASE_API_KEY" \
  --set-env-vars "NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=$NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN" \
  --set-env-vars "NEXT_PUBLIC_FIREBASE_PROJECT_ID=$NEXT_PUBLIC_FIREBASE_PROJECT_ID" \
  --set-env-vars "NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=$NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET" \
  --set-env-vars "NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=$NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID" \
  --set-env-vars "NEXT_PUBLIC_FIREBASE_APP_ID=$NEXT_PUBLIC_FIREBASE_APP_ID" \
  --set-env-vars "NODE_ENV=production" \
  --set-env-vars "NEXT_TELEMETRY_DISABLED=1" \
  --labels "environment=$ENVIRONMENT,application=ollama,component=frontend,managed-by=gcloud,team=ollama-engineering"

# Get service URL
echo ""
echo "🔍 Fetching service URL..."
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format='value(status.url)')

echo ""
echo "✅ Deployment complete!"
echo ""
echo "Service URL: $SERVICE_URL"
echo "Domain: elevatediq.ai/ollama (configure Load Balancer)"
echo ""
echo "Next steps:"
echo "  1. Configure GCP Load Balancer to route elevatediq.ai/ollama to $SERVICE_URL"
echo "  2. Update Cloud Armor security policies"
echo "  3. Configure CDN for static assets"
echo "  4. Set up monitoring and alerting"
