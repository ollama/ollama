# Production Deployment Guide - Ollama Elite AI Platform

**Last Updated**: January 13, 2026
**Status**: Phase 4 - GCP Deployment Ready
**Deployment Target**: https://elevatediq.ai/ollama
**OAuth Provider**: Google Firebase (project-131055855980)
**Integration Partner**: Gov-AI-Scout

---

## Quick Start Deployment

### Prerequisites Check
```bash
cd /home/akushnir/ollama

# Verify environment
echo "Python version:" && python --version
echo "Docker version:" && docker --version
echo "GCloud version:" && gcloud --version

# Verify credentials
gcloud auth list
gcloud config get-value project
```

### Automated Deployment

```bash
# 1. Setup Firebase Service Account
chmod +x scripts/setup-firebase.sh
./scripts/setup-firebase.sh

# 2. Build and deploy to GCP
chmod +x scripts/deploy-gcp.sh
./scripts/deploy-gcp.sh

# 3. Verify deployment
curl https://elevatediq.ai/ollama/health
```

---

## Manual Deployment Steps

### Step 1: Build Docker Image

```bash
cd /home/akushnir/ollama

# Build with specific Python version
docker build \
  -t ollama:1.0.0 \
  -t ollama:latest \
  --build-arg PYTHON_VERSION=3.12 \
  --build-arg BASE_IMAGE=python:3.12-slim \
  -f docker/Dockerfile \
  .

# Verify image
docker run --rm ollama:1.0.0 --version
```

**Expected**: Image built successfully, ~500MB-1GB in size

### Step 2: Push to GCP Container Registry

```bash
PROJECT_ID="project-131055855980"

# Authenticate with GCP
gcloud auth configure-docker gcr.io

# Tag image
docker tag ollama:1.0.0 gcr.io/$PROJECT_ID/ollama:1.0.0
docker tag ollama:latest gcr.io/$PROJECT_ID/ollama:latest

# Push image
docker push gcr.io/$PROJECT_ID/ollama:1.0.0
docker push gcr.io/$PROJECT_ID/ollama:latest

# Verify in GCP
gcloud container images list-tags gcr.io/$PROJECT_ID/ollama
```

**Expected**: Image available in GCP Container Registry

### Step 3: Create Firebase Service Account

```bash
PROJECT_ID="project-131055855980"
SERVICE_ACCOUNT="ollama-service"
EMAIL="${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com"

# Create service account
gcloud iam service-accounts create $SERVICE_ACCOUNT \
  --display-name="Ollama API Service" \
  --project=$PROJECT_ID

# Grant roles
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member=serviceAccount:$EMAIL \
  --role=roles/firebase.admin \
  --quiet

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member=serviceAccount:$EMAIL \
  --role=roles/datastore.user \
  --quiet

# Create and store key
KEY_FILE="/tmp/firebase-sa-key.json"
gcloud iam service-accounts keys create $KEY_FILE \
  --iam-account=$EMAIL \
  --project=$PROJECT_ID

# Store in Secret Manager
gcloud secrets create firebase-service-account \
  --data-file=$KEY_FILE \
  --replication-policy="automatic" \
  --project=$PROJECT_ID || \
gcloud secrets versions add firebase-service-account \
  --data-file=$KEY_FILE \
  --project=$PROJECT_ID

rm $KEY_FILE
```

**Expected**: Service account created, Firebase credentials stored in Secret Manager

### Step 4: Deploy to Cloud Run

```bash
PROJECT_ID="project-131055855980"
REGION="us-central1"
SERVICE="ollama-api"
IMAGE="gcr.io/${PROJECT_ID}/ollama:1.0.0"

gcloud run deploy $SERVICE \
  --image=$IMAGE \
  --platform managed \
  --region=$REGION \
  --allow-unauthenticated \
  --memory=4Gi \
  --cpu=2 \
  --timeout=600 \
  --max-instances=20 \
  --min-instances=1 \
  --port=8000 \
  --set-env-vars="\
ENVIRONMENT=production,\
FIREBASE_PROJECT_ID=project-131055855980,\
GCP_PROJECT_ID=project-131055855980,\
FIREBASE_ENABLED=true,\
PUBLIC_API_ENDPOINT=https://elevatediq.ai/ollama,\
LOG_LEVEL=info" \
  --set-secrets="FIREBASE_CREDENTIALS_PATH=/run/secrets/firebase-credentials:firebase-service-account@latest" \
  --project=$PROJECT_ID

# Get service URL
gcloud run services describe $SERVICE \
  --region=$REGION \
  --project=$PROJECT_ID \
  --format='value(status.url)'
```

**Expected**: Service deployed and running on Cloud Run

### Step 5: Configure GCP Load Balancer

```bash
PROJECT_ID="project-131055855980"
REGION="us-central1"
SERVICE="ollama-api"

# Get Cloud Run service URL
SERVICE_URL=$(gcloud run services describe $SERVICE \
  --region=$REGION \
  --project=$PROJECT_ID \
  --format='value(status.url)' 2>/dev/null)

echo "Cloud Run Service URL: $SERVICE_URL"

# Create backend service (use gcloud console for full configuration)
# See: GCP_LB_DEPLOYMENT.md for complete setup
```

### Step 6: Configure DNS

Update DNS provider to point `elevatediq.ai` to GCP Load Balancer IP:

```bash
# Get Load Balancer IP
gcloud compute addresses list \
  --global \
  --project=$PROJECT_ID \
  --filter="name=ollama-ip"

# Then update DNS:
# A record: elevatediq.ai → <LOAD_BALANCER_IP>
# CNAME: *.elevatediq.ai → elevatediq.ai

# Verify propagation
nslookup elevatediq.ai
```

**Expected**: DNS resolves to GCP Load Balancer IP

---

## Testing Deployment

### Health Check Tests

```bash
# 1. Public endpoint (no auth required)
curl -v https://elevatediq.ai/ollama/health

# Expected response:
# HTTP/2 200
# {
#   "status": "healthy",
#   "version": "1.0.0",
#   "services": {
#     "database": "healthy",
#     "redis": "healthy",
#     "qdrant": "healthy"
#   }
# }

# 2. Protected endpoint (auth required)
curl -v https://elevatediq.ai/ollama/api/v1/health
# Expected: 401 Unauthorized (no token)

# 3. With valid token
TOKEN=$(gcloud auth print-identity-token)
curl -v -H "Authorization: Bearer $TOKEN" \
  https://elevatediq.ai/ollama/api/v1/health
# Expected: 200 OK with authenticated user info
```

### API Endpoint Tests

```bash
# Get auth token
TOKEN=$(gcloud auth print-identity-token)

# Test text generation
curl -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "prompt": "Explain machine learning in one sentence",
    "max_tokens": 100
  }' \
  https://elevatediq.ai/ollama/api/v1/generate

# Test embeddings
curl -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nomic-embed-text",
    "text": "Climate change policy"
  }' \
  https://elevatediq.ai/ollama/api/v1/embeddings
```

### Load Testing

```bash
# Install hey (if not already installed)
go install github.com/rakyll/hey@latest

# Get auth token
TOKEN=$(gcloud auth print-identity-token)

# Run load test (1000 requests, 50 concurrent)
hey -n 1000 -c 50 \
  -m GET \
  -H "Authorization: Bearer $TOKEN" \
  https://elevatediq.ai/ollama/api/v1/health

# Expected results:
# - Success rate: 100%
# - Average latency < 200ms
# - 99th percentile latency < 500ms
# - No timeout errors
```

---

## Monitoring & Observability

### View Logs

```bash
PROJECT_ID="project-131055855980"

# View recent logs
gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="ollama-api"' \
  --limit=50 \
  --format=json \
  --project=$PROJECT_ID

# Watch logs in real-time
gcloud logging tail \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="ollama-api"' \
  --limit=50 \
  --stream \
  --project=$PROJECT_ID

# Filter by severity
gcloud logging read \
  'severity=ERROR AND resource.type="cloud_run_revision"' \
  --limit=50 \
  --project=$PROJECT_ID
```

### View Metrics

```bash
# List available metrics
gcloud monitoring metrics-descriptors list --project=$PROJECT_ID

# Example metrics to monitor:
# - run.googleapis.com/request_count (requests per second)
# - run.googleapis.com/request_latencies (latency distribution)
# - run.googleapis.com/container_memory_usage_bytes (memory usage)
# - run.googleapis.com/container_cpu_utilization (CPU usage)
```

### Create Custom Dashboard

Dashboard creation via Cloud Console:
1. Go to Cloud Monitoring → Dashboards
2. Create new dashboard
3. Add widgets for:
   - Request rate
   - Request latency (p50, p95, p99)
   - Error rate
   - Cache hit rate
   - CPU/Memory utilization

---

## Troubleshooting

### Service Fails to Start

```bash
# Check deployment status
gcloud run describe ollama-api --region=us-central1

# View recent revision
gcloud run revisions list --service=ollama-api --region=us-central1

# Check logs for errors
gcloud logging read \
  'resource.type="cloud_run_revision"' \
  --limit=100 \
  --format=json | grep -i error
```

### Health Check Failing

```bash
# Test from local machine
curl -I https://elevatediq.ai/ollama/health

# Test directly on Cloud Run service
gcloud run services describe ollama-api \
  --region=us-central1 \
  --format='value(status.url)'

# If 503 error, check container health in logs
```

### Performance Issues

```bash
# Monitor resource usage
gcloud monitoring read \
  'resource.type="cloud_run_revision" AND metric.type="run.googleapis.com/container_cpu_utilization"' \
  --format=json

# Check for scaling events
gcloud logging read \
  'resource.type="cloud_run_revision" AND (jsonPayload.message=~".*scaled.*" OR jsonPayload.message=~".*replica.*")' \
  --limit=50
```

### OAuth Token Issues

```bash
# Verify token is valid
TOKEN=$(gcloud auth print-identity-token)
curl -H "Authorization: Bearer $TOKEN" \
  https://elevatediq.ai/ollama/api/v1/health

# If 401, refresh token
gcloud auth application-default print-access-token

# Check token expiration
echo $TOKEN | jq -R 'split(".") | .[1] | @base64d | fromjson'
```

---

## Rollback Procedure

### If Issues Occur

```bash
PROJECT_ID="project-131055855980"
REGION="us-central1"
SERVICE="ollama-api"

# 1. View deployment history
gcloud run revisions list \
  --service=$SERVICE \
  --region=$REGION \
  --project=$PROJECT_ID

# 2. Rollback to previous version
PREVIOUS_REVISION="ollama-api-previous-id"
gcloud run rollbacks execute \
  --service=$SERVICE \
  --region=$REGION \
  --project=$PROJECT_ID

# Or deploy previous image directly
gcloud run deploy $SERVICE \
  --image=gcr.io/$PROJECT_ID/ollama:previous-version \
  --region=$REGION \
  --project=$PROJECT_ID
```

---

## Post-Deployment

### Notify Stakeholders

1. **Slack**: #ollama-deployment channel
   - "✅ Ollama v1.0.0 deployed to production"
   - "📍 Endpoint: https://elevatediq.ai/ollama"
   - "🔑 OAuth: Google Firebase (project-131055855980)"
   - "👥 Integration: Gov-AI-Scout pattern"

2. **Email**: akushnir@bioenergystrategies.com
   - Deployment summary
   - Known issues (if any)
   - Next steps

### Update Documentation

- [ ] API documentation current
- [ ] Integration guide complete
- [ ] Troubleshooting guide updated
- [ ] On-call runbook created
- [ ] SLA documentation published

### Monitor for 24 Hours

- [ ] Watch error rates
- [ ] Monitor performance metrics
- [ ] Check scaling behavior
- [ ] Verify integrations working
- [ ] Confirm OAuth working properly

---

## Success Criteria

✅ **Deployment Successful When**:
- Public endpoint returns 200: `https://elevatediq.ai/ollama/health`
- Protected endpoint requires valid token
- Text generation working with Firebase JWT
- Embeddings generation functional
- No error spikes in logs
- Performance metrics within SLA:
  - Availability: > 99.9%
  - Latency p99: < 500ms
  - Error rate: < 1%
- Gov-AI-Scout can authenticate and use API

---

**Deployment Complete ✅**

**Contact**: akushnir@bioenergystrategies.com
**Support**: See troubleshooting section above
**Next Review**: January 20, 2026
