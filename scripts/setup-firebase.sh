#!/bin/bash
# Firebase Service Account Setup Script
# Configures Firebase authentication for Ollama deployment

set -e

PROJECT_ID="project-131055855980"
SERVICE_ACCOUNT_NAME="ollama-service"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "=========================================="
echo "Firebase Service Account Setup"
echo "=========================================="
echo "Project ID: $PROJECT_ID"
echo "Service Account: $SERVICE_ACCOUNT_EMAIL"
echo ""

# Step 1: Create service account if doesn't exist
echo "[1/5] Creating service account..."
if gcloud iam service-accounts describe $SERVICE_ACCOUNT_EMAIL --project=$PROJECT_ID &>/dev/null; then
    echo "  ✓ Service account already exists"
else
    gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME \
        --display-name="Ollama API Service Account" \
        --project=$PROJECT_ID
    echo "  ✓ Service account created"
fi

# Step 2: Grant Firebase Admin role
echo "[2/5] Granting Firebase Admin role..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member=serviceAccount:$SERVICE_ACCOUNT_EMAIL \
    --role=roles/firebase.admin \
    --quiet &>/dev/null || true
echo "  ✓ Firebase Admin role granted"

# Step 3: Grant Cloud Datastore User role
echo "[3/5] Granting Cloud Datastore User role..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member=serviceAccount:$SERVICE_ACCOUNT_EMAIL \
    --role=roles/datastore.user \
    --quiet &>/dev/null || true
echo "  ✓ Cloud Datastore User role granted"

# Step 4: Create and store service account key
echo "[4/5] Creating service account key..."
KEY_FILE="/tmp/firebase-service-account.json"
gcloud iam service-accounts keys create "$KEY_FILE" \
    --iam-account=$SERVICE_ACCOUNT_EMAIL \
    --project=$PROJECT_ID

# Step 5: Store in GCP Secret Manager
echo "[5/5] Storing in GCP Secret Manager..."
gcloud secrets create firebase-service-account \
    --data-file="$KEY_FILE" \
    --replication-policy="automatic" \
    --project=$PROJECT_ID \
    --quiet &>/dev/null || \
gcloud secrets versions add firebase-service-account \
    --data-file="$KEY_FILE" \
    --project=$PROJECT_ID \
    --quiet

# Clean up
rm "$KEY_FILE"

echo ""
echo "=========================================="
echo "✅ Firebase Setup Complete!"
echo "=========================================="
echo ""
echo "Service Account Email: $SERVICE_ACCOUNT_EMAIL"
echo "Firebase Service Account Secret: firebase-service-account"
echo ""
echo "Next steps:"
echo "1. Run Docker build: docker build -t ollama:1.0.0 ."
echo "2. Push to GCP: docker tag ollama:1.0.0 gcr.io/$PROJECT_ID/ollama:1.0.0"
echo "3. Deploy to Cloud Run: gcloud run deploy ollama-api ..."
echo ""
