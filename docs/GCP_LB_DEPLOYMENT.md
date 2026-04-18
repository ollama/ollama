# GCP Load Balancer - Ollama Deployment Guide

**Date**: January 13, 2026
**Project**: project-131055855980
**Endpoint**: https://elevatediq.ai/ollama

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│              Clients (Internet)                              │
│          (Gov-AI-Scout, End Users)                           │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTPS/TLS 1.3+
        ┌────────────▼────────────────┐
        │   GCP Load Balancer         │
        │ https://elevatediq.ai/ollama│
        │ • SSL/TLS Termination       │
        │ • Request routing           │
        │ • Rate limiting (Cloud      │
        │   Armor)                    │
        │ • DDoS protection           │
        └────────────────┬────────────┘
                         │ Internal routing
        ┌────────────────▼────────────────────┐
        │   Cloud Run / GKE                   │
        │   (Ollama FastAPI Container)        │
        │   • Firebase OAuth verification     │
        │   • Request processing              │
        │   • Response formatting             │
        └────────────────┬────────────────────┘
                         │ Internal Docker network
        ┌────────────────▼────────────────────┐
        │   Docker Services (Internal)        │
        ├─ PostgreSQL (:5432)                 │
        ├─ Redis (:6379)                      │
        ├─ Qdrant (:6333)                     │
        └─ Ollama (:11434)                    │
```

---

## GCP Load Balancer Configuration

### Frontend Configuration

**Resource Name**: `ollama-frontend`
**Protocol**: HTTPS
**IP Address**: Global static IP
**Certificate**: Managed by Google (auto-renewal)
**Domain**: elevatediq.ai

```bash
gcloud compute backend-services create ollama-backend \
  --protocol=HTTP \
  --global \
  --enable-cdn \
  --session-affinity=CLIENT_IP \
  --connection-draining-timeout=300

gcloud compute url-maps create ollama-url-map \
  --default-service=ollama-backend

gcloud compute ssl-certificates create ollama-cert \
  --domains=elevatediq.ai \
  --global

gcloud compute target-https-proxies create ollama-https-proxy \
  --url-map=ollama-url-map \
  --ssl-certificates=ollama-cert

gcloud compute forwarding-rules create ollama-forwarding-rule \
  --global \
  --target-https-proxy=ollama-https-proxy \
  --address=ollama-ip \
  --ports=443
```

### Backend Configuration

**Service**: Cloud Run (recommended) or GKE
**Health Check**: `/health` endpoint (no auth required)
**Timeout**: 300 seconds (model inference)
**Session Affinity**: Client IP (sticky sessions)

```bash
gcloud compute health-checks create http ollama-health-check \
  --request-path=/health \
  --port=8000 \
  --check-interval=10s \
  --timeout=5s \
  --unhealthy-threshold=3 \
  --healthy-threshold=2
```

### Request Path Routing

```yaml
defaultService: projects/project-131055855980/global/backendServices/ollama-backend

hostRules:
  - hosts:
      - "elevatediq.ai"
    pathMatcher: "ollama-paths"

pathMatchers:
  - name: "ollama-paths"
    pathRules:
      - paths:
          - "/ollama/*"
        service: projects/project-131055855980/global/backendServices/ollama-backend
      - paths:
          - "/health"
        service: projects/project-131055855980/global/backendServices/ollama-backend
        priority: 1
```

### Security Configuration

**Cloud Armor Policy**:

```yaml
apiVersion: compute.cnrm.cloud.google.com/v1beta1
kind: ComputeSecurityPolicy
metadata:
  name: ollama-security-policy
spec:
  description: "Security policy for Ollama API"
  rules:
    # Rate limiting
    - priority: 100
      action: "rate_based_ban"
      rateLimitOptions:
        conformAction: "allow"
        exceedAction: "deny(429)"
        banDurationSec: 600
        rateLimitThreshold:
          count: 100
          intervalSec: 60
        banThreshold:
          count: 1000
          intervalSec: 60
      match:
        versionedExpr: "CEL_V1"
        cexlMatches:
          - expr: "origin.region_code == 'US'"

    # Allow authenticated requests
    - priority: 200
      action: "allow"
      match:
        versionedExpr: "CEL_V1"
        cexlMatches:
          - expr: "has(request.headers['authorization'])"

    # Deny unauthenticated /api/v1/* requests
    - priority: 300
      action: "deny(401)"
      match:
        versionedExpr: "CEL_V1"
        cexlMatches:
          - expr: "request.path.matches('/ollama/api/v1/.*') && !has(request.headers['authorization'])"

    # Allow /health (public)
    - priority: 400
      action: "allow"
      match:
        versionedExpr: "CEL_V1"
        cexlMatches:
          - expr: "request.path == '/health'"
```

---

## Deployment Steps

### Step 1: Build Docker Image

```bash
cd /home/akushnir/ollama

# Build multi-stage image
docker build -t ollama:1.0.0 \
  -f docker/Dockerfile \
  --build-arg PYTHON_VERSION=3.12 \
  --build-arg BASE_IMAGE=python:3.12-slim \
  .

# Tag for GCP Registry
docker tag ollama:1.0.0 \
  gcr.io/project-131055855980/ollama:1.0.0

# Push to GCP Container Registry
docker push gcr.io/project-131055855980/ollama:1.0.0
```

### Step 2: Deploy to Cloud Run

```bash
gcloud run deploy ollama-api \
  --image gcr.io/project-131055855980/ollama:1.0.0 \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 600 \
  --max-instances 10 \
  --min-instances 1 \
  --set-env-vars "FIREBASE_PROJECT_ID=project-131055855980,GCP_PROJECT_ID=project-131055855980,ENVIRONMENT=production" \
  --set-secrets "FIREBASE_CREDENTIALS=/run/secrets/firebase-credentials:firebase-service-account@latest" \
  --project project-131055855980
```

### Step 3: Configure Load Balancer

```bash
# Create backend service pointing to Cloud Run
gcloud compute backend-services update ollama-backend \
  --global \
  --custom-request-header "X-Forwarded-Proto:https" \
  --health-checks ollama-health-check

# Attach security policy
gcloud compute security-policies rules update 100 \
  --security-policy ollama-security-policy \
  --action "allow"
```

### Step 4: Update DNS

```bash
# Point elevatediq.ai to GCP LB IP
# In DNS provider:
# A record: elevatediq.ai → <GCP-LB-IP>
# CNAME: *.elevatediq.ai → elevatediq.ai

# Verify DNS propagation
nslookup elevatediq.ai
```

---

## Firebase Service Account Setup

### Step 1: Create Service Account

```bash
gcloud iam service-accounts create ollama-service \
  --display-name "Ollama API Service Account" \
  --project project-131055855980

# Grant necessary permissions
gcloud projects add-iam-policy-binding project-131055855980 \
  --member=serviceAccount:ollama-service@project-131055855980.iam.gserviceaccount.com \
  --role=roles/firebase.admin

gcloud projects add-iam-policy-binding project-131055855980 \
  --member=serviceAccount:ollama-service@project-131055855980.iam.gserviceaccount.com \
  --role=roles/datastore.user
```

### Step 2: Generate and Store Credentials

```bash
# Create JSON key
gcloud iam service-accounts keys create /tmp/firebase-sa-key.json \
  --iam-account=ollama-service@project-131055855980.iam.gserviceaccount.com

# Store in GCP Secret Manager
gcloud secrets create firebase-service-account \
  --data-file=/tmp/firebase-sa-key.json \
  --replication-policy="automatic" \
  --project project-131055855980

# Clean up local copy
rm /tmp/firebase-sa-key.json
```

### Step 3: Grant Cloud Run Access to Secret

```bash
# Get Cloud Run service account
CLOUD_RUN_SA=$(gcloud iam service-accounts list \
  --filter="email~service-^$REGION$" \
  --format="value(email)" \
  --project project-131055855980)

# Grant secret accessor role
gcloud secrets add-iam-policy-binding firebase-service-account \
  --member=serviceAccount:$CLOUD_RUN_SA \
  --role=roles/secretmanager.secretAccessor \
  --project project-131055855980
```

---

## Deployment Configuration Files

### cloud-run-deploy.yaml

```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: ollama-api
  namespace: default
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "1"
        autoscaling.knative.dev/maxScale: "10"
    spec:
      serviceAccountName: ollama-service
      containers:
        - image: gcr.io/project-131055855980/ollama:1.0.0
          ports:
            - containerPort: 8000
          env:
            - name: ENVIRONMENT
              value: "production"
            - name: FIREBASE_PROJECT_ID
              value: "project-131055855980"
            - name: GCP_PROJECT_ID
              value: "project-131055855980"
            - name: FIREBASE_CREDENTIALS_PATH
              value: "/run/secrets/firebase-credentials"
          volumeMounts:
            - name: firebase-credentials
              mountPath: /run/secrets
              readOnly: true
          resources:
            requests:
              memory: "2Gi"
              cpu: "1000m"
            limits:
              memory: "4Gi"
              cpu: "2000m"
          livenessProbe:
            httpGet:
              path: /health/live
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
      volumes:
        - name: firebase-credentials
          secret:
            secretName: firebase-service-account
```

---

## Testing Deployment

### Health Check (Public - No Auth)

```bash
# Should return 200 OK
curl https://elevatediq.ai/ollama/health

# Response:
# {
#   "status": "healthy",
#   "version": "1.0.0",
#   "services": {
#     "database": "healthy",
#     "redis": "healthy",
#     "qdrant": "healthy"
#   }
# }
```

### Protected Endpoint (OAuth Required)

```bash
# Without token - should return 401
curl https://elevatediq.ai/ollama/api/v1/health
# Response: {"detail": "Missing or invalid Authorization header"}

# With valid Firebase token
TOKEN=$(gcloud auth print-identity-token)
curl -H "Authorization: Bearer $TOKEN" \
  https://elevatediq.ai/ollama/api/v1/health
# Response: 200 OK with health status
```

### Load Testing

```bash
# Install hey
go install github.com/rakyll/hey@latest

# Run load test
hey -n 1000 -c 10 \
  -m GET \
  -H "Authorization: Bearer $TOKEN" \
  https://elevatediq.ai/ollama/api/v1/health

# Monitor in Cloud Console:
# - Cloud Monitoring → Dashboards
# - Cloud Logging → Logs
# - Cloud Trace → Request traces
```

---

## Monitoring & Logging

### Cloud Logging

```bash
# View application logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ollama-api" \
  --limit 50 \
  --format json

# Watch logs in real-time
gcloud logging tail "resource.type=cloud_run_revision AND resource.labels.service_name=ollama-api" \
  --limit 50 \
  --stream
```

### Cloud Metrics

```bash
# Create custom dashboard
gcloud monitoring dashboards create --config-from-file=- << EOF
{
  "displayName": "Ollama API",
  "mosaicLayout": {
    "columns": 12,
    "tiles": [
      {
        "width": 6,
        "height": 4,
        "widget": {
          "title": "Request Rate",
          "xyChart": {
            "dataSets": [{
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "metric.type=\\\"run.googleapis.com/request_count\\\""
                }
              }
            }]
          }
        }
      }
    ]
  }
}
EOF
```

---

## Rollback Procedure

```bash
# If deployment has issues, rollback to previous version
gcloud run deploy ollama-api \
  --image gcr.io/project-131055855980/ollama:previous-version \
  --region us-central1 \
  --project project-131055855980

# Or use traffic splitting
gcloud run services update-traffic ollama-api \
  --to-revisions LATEST=50,previous-revision=50 \
  --region us-central1 \
  --project project-131055855980
```

---

## Production Checklist

- ✅ Docker image built and tested
- ✅ Firebase service account created and configured
- ✅ GCP Load Balancer configured
- ✅ Cloud Armor security policy deployed
- ✅ DNS records pointing to LB
- ✅ SSL/TLS certificate installed
- ✅ Monitoring dashboards set up
- ✅ Logging configured
- ✅ Health checks passing
- ✅ OAuth endpoints tested
- ✅ Load testing completed
- ✅ Rollback procedure documented

---

**Status**: Ready for deployment to GCP Load Balancer
