# Public Endpoint Deployment Guide: elevatediq.ai/ollama

**Status**: Production-ready for GCP Load Balancer deployment  
**Endpoint**: `https://elevatediq.ai/ollama`  
**Last Updated**: January 12, 2026

---

## Overview

This guide covers deploying Ollama to the public-facing endpoint `elevatediq.ai/ollama` through a Google Cloud Load Balancer.

---

## Architecture

```
┌──────────────┐
│   Client     │ (Browser, SDK, API)
│ elevatediq   │
└──────┬───────┘
       │ HTTPS/TLS
       ▼
┌──────────────────────────────┐
│   GCP Cloud Load Balancer    │
│   - SSL/TLS Termination      │
│   - Rate Limiting            │
│   - DDoS Protection (Armor)  │
│   - Health Checks            │
│   - Session Affinity         │
└──────┬──────────────────────┘
       │ HTTP (internal VPC)
       ▼
┌──────────────────────────────┐
│  Backend Services (GKE/VMs)  │
│  - Ollama API (Port 8000)    │
│  - Multiple replicas         │
│  - Auto-scaling              │
└──────────────────────────────┘
```

---

## Prerequisites

✅ Google Cloud Project with billing  
✅ Domain: `elevatediq.ai` (DNS configured)  
✅ SSL Certificate (auto-managed or imported)  
✅ gcloud CLI installed  
✅ Kubernetes or VM resources  

---

## Step 1: Prepare Ollama Backend

### Environment Configuration

Create `.env` for production:

```bash
# API Configuration
OLLAMA_PUBLIC_URL=https://elevatediq.ai/ollama
OLLAMA_DOMAIN=elevatediq.ai
OLLAMA_HOST=0.0.0.0:8000

# Database
DATABASE_URL=postgresql://ollama:password@postgres:5432/ollama

# Cache
REDIS_URL=redis://redis:6379/0

# Security
API_KEY_AUTH_ENABLED=true
CORS_ORIGINS=["https://elevatediq.ai","https://*.elevatediq.ai"]
TLS_ENABLED=false  # GCP LB handles TLS

# Performance
MAX_CONCURRENT_REQUESTS=32
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=100
```

### Configuration File

```yaml
# config/production.yaml
server:
  host: "0.0.0.0"
  port: 8000
  public_url: "https://elevatediq.ai/ollama"
  domain: "elevatediq.ai"
  workers: 8
  log_level: "INFO"

security:
  api_key_auth_enabled: true
  cors_origins:
    - "https://elevatediq.ai"
    - "https://*.elevatediq.ai"
  tls_enabled: false  # LB handles TLS

rate_limiting:
  enabled: true
  requests_per_minute: 100
  burst: 150
```

---

## Step 2: Build Docker Image

### Dockerfile

```dockerfile
# docker/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements/core.txt .
RUN pip install --no-cache-dir -r core.txt

# Copy application
COPY ollama/ ./ollama/
COPY config/ ./config/

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start server
CMD ["python", "-m", "ollama.api.server"]
```

### Build and Push

```bash
# Build image
docker build -t ollama:prod -f docker/Dockerfile .

# Tag for GCP registry
docker tag ollama:prod gcr.io/YOUR_PROJECT/ollama:prod

# Push to GCP Container Registry
docker push gcr.io/YOUR_PROJECT/ollama:prod
```

---

## Step 3: Deploy to GCP

### Option A: Using GKE (Kubernetes)

#### Create Kubernetes Manifests

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ollama
  labels:
    app: ollama
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ollama
  template:
    metadata:
      labels:
        app: ollama
    spec:
      containers:
      - name: ollama
        image: gcr.io/YOUR_PROJECT/ollama:prod
        ports:
        - containerPort: 8000
        env:
        - name: OLLAMA_PUBLIC_URL
          value: "https://elevatediq.ai/ollama"
        - name: OLLAMA_DOMAIN
          value: "elevatediq.ai"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: ollama-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: ollama-secrets
              key: redis-url
        - name: API_KEY_AUTH_ENABLED
          value: "true"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            cpu: 4
            memory: 8Gi
          limits:
            cpu: 8
            memory: 16Gi
---
apiVersion: v1
kind: Service
metadata:
  name: ollama
spec:
  type: LoadBalancer
  selector:
    app: ollama
  ports:
  - protocol: TCP
    port: 8000
    targetPort: 8000
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ollama
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ollama
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

#### Deploy

```bash
# Create GKE cluster
gcloud container clusters create ollama-cluster \
  --zone=us-central1-a \
  --num-nodes=3 \
  --machine-type=n1-standard-8

# Get credentials
gcloud container clusters get-credentials ollama-cluster \
  --zone=us-central1-a

# Create secrets
kubectl create secret generic ollama-secrets \
  --from-literal=database-url="postgresql://..." \
  --from-literal=redis-url="redis://..."

# Deploy
kubectl apply -f k8s/

# Verify
kubectl get pods
kubectl get svc ollama
```

### Option B: Using Compute Engine (VMs)

```bash
# Create instance template
gcloud compute instance-templates create ollama-template \
  --machine-type=n1-standard-8 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB \
  --metadata=startup-script='
    #!/bin/bash
    apt-get update
    apt-get install -y docker.io
    docker pull gcr.io/YOUR_PROJECT/ollama:prod
    docker run -d \
      -p 8000:8000 \
      -e OLLAMA_PUBLIC_URL=https://elevatediq.ai/ollama \
      -e DATABASE_URL=$DATABASE_URL \
      gcr.io/YOUR_PROJECT/ollama:prod
  '

# Create instance group
gcloud compute instance-groups managed create ollama-ig \
  --template=ollama-template \
  --size=3 \
  --zone=us-central1-a

# Configure autoscaling
gcloud compute instance-groups managed set-autoscaling ollama-ig \
  --min-num-replicas=2 \
  --max-num-replicas=10 \
  --target-cpu-utilization=0.6 \
  --zone=us-central1-a
```

---

## Step 4: Configure Load Balancer

See [docs/gcp-load-balancer.md](gcp-load-balancer.md) for detailed LB setup.

Quick summary:

```bash
# Create health check
gcloud compute health-checks create http ollama-health \
  --port=8000 \
  --request-path=/health

# Create backend service
gcloud compute backend-services create ollama-backend \
  --global \
  --protocol=HTTP \
  --port-name=http \
  --health-checks=ollama-health

# Create URL map
gcloud compute url-maps create ollama-url-map \
  --default-service=ollama-backend

# Create SSL certificate (auto-managed)
gcloud compute ssl-certificates create ollama-cert \
  --domains=elevatediq.ai,api.elevatediq.ai

# Create HTTPS proxy
gcloud compute target-https-proxies create ollama-https-proxy \
  --url-map=ollama-url-map \
  --ssl-certificates=ollama-cert

# Create forwarding rule
gcloud compute forwarding-rules create ollama-https-rule \
  --global \
  --target-https-proxy=ollama-https-proxy \
  --address=ollama-ip \
  --ports=443
```

---

## Step 5: Configure DNS

Update your DNS provider to point to the load balancer IP:

```dns
elevatediq.ai       A    <LOAD_BALANCER_IP>
api.elevatediq.ai   A    <LOAD_BALANCER_IP>
```

---

## Step 6: Test Public Endpoint

### Health Check

```bash
curl -v https://elevatediq.ai/ollama/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "production",
  "public_url": "https://elevatediq.ai/ollama"
}
```

### API Request

```bash
# With API key
API_KEY="your-api-key"

curl -X POST https://elevatediq.ai/ollama/api/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "model": "llama2",
    "prompt": "Hello world",
    "stream": false
  }'
```

### Python Client

```python
from ollama import Client

# Connect to public endpoint
client = Client(
    base_url="https://elevatediq.ai/ollama",
    api_key="your-api-key"
)

# Generate
response = client.generate(
    model="llama2",
    prompt="Explain local AI"
)
print(response)

# Chat
response = client.chat(
    model="llama2",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response)
```

---

## Step 7: Monitoring & Alerts

### Cloud Monitoring

```bash
# View metrics in Cloud Monitoring console
gcloud monitoring dashboards create --config-from-file=- <<EOF
{
  "displayName": "Ollama Public API",
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
                  "filter": "metric.type=\"http.server.request.duration\" resource.type=\"http_load_balancer\""
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

### Logging

```bash
# View logs
gcloud logging read "resource.type=http_load_balancer" \
  --limit 50 \
  --format=json

# Create log-based metric
gcloud logging metrics create ollama_errors \
  --description="Ollama API errors" \
  --log-filter='resource.type="http_load_balancer" severity="ERROR"'
```

### Alerts

```bash
# Create alert policy
gcloud alpha monitoring policies create \
  --notification-channels=<CHANNEL_ID> \
  --display-name="Ollama High Error Rate" \
  --condition-display-name="Error rate > 5%" \
  --condition-threshold-value=0.05 \
  --condition-threshold-duration=300s \
  --condition-threshold-filter='metric.type="compute.googleapis.com/instance/cpu/utilization"'
```

---

## Step 8: Enable Cloud Armor (DDoS Protection)

```bash
# Create security policy
gcloud compute security-policies create ollama-armor \
  --type=CLOUD_ARMOR

# Add rate limiting
gcloud compute security-policies rules create 100 \
  --security-policy=ollama-armor \
  --action=rate-based-ban \
  --rate-limit-options=enforced-cap-elements=CLIENT_IP \
  --rate-limit-options=rate-limit-threshold-count=1000 \
  --rate-limit-options=rate-limit-threshold-interval-sec=60 \
  --ban-duration-sec=600

# Attach to backend
gcloud compute backend-services update ollama-backend \
  --global \
  --security-policy=ollama-armor
```

---

## Performance Optimization

### Enable CDN

```bash
gcloud compute backend-services update ollama-backend \
  --global \
  --enable-cdn
```

### Session Affinity

```bash
gcloud compute backend-services update ollama-backend \
  --global \
  --session-affinity=CLIENT_IP \
  --affinityCookieTtlSec=3600
```

### Custom Cache Settings

```bash
gcloud compute backend-services update ollama-backend \
  --global \
  --cache-mode=CACHE_ALL_STATIC \
  --client-ttl=3600 \
  --default-ttl=3600
```

---

## Troubleshooting

### Check Backend Health

```bash
gcloud compute backend-services get-health ollama-backend --global
```

### View Load Balancer Metrics

```bash
# CPU utilization
gcloud monitoring time-series list \
  --filter='metric.type="compute.googleapis.com/instance/cpu/utilization"'

# Network traffic
gcloud monitoring time-series list \
  --filter='metric.type="compute.googleapis.com/instance/network/in_bytes_count"'
```

### SSH into Backend VM

```bash
# List instances
gcloud compute instances list

# SSH
gcloud compute ssh ollama-instance-1 --zone=us-central1-a

# Check service
sudo systemctl status ollama
sudo journalctl -u ollama -n 100
```

### Test Connectivity

```bash
# From VM
curl http://localhost:8000/health

# From another machine
curl -H "X-API-Key: $API_KEY" https://elevatediq.ai/ollama/health
```

---

## Security Checklist

- ✅ TLS/HTTPS enabled (GCP managed certificate)
- ✅ API key authentication required
- ✅ CORS properly configured
- ✅ Rate limiting enabled
- ✅ DDoS protection (Cloud Armor)
- ✅ Security headers set (HSTS, etc.)
- ✅ Audit logging enabled
- ✅ Secrets in Cloud Secret Manager
- ✅ VPC network isolation
- ✅ Cloud IAM access controls

---

## Cost Optimization

1. **Use committed use discounts** for instances
2. **Enable CDN** to reduce egress
3. **Auto-scaling** to match demand
4. **Spot instances** for non-critical workloads
5. **Monitor costs** with Cloud Billing

---

## Rollback Plan

### Blue-Green Deployment

```bash
# Keep two versions running
# Update load balancer to switch traffic

# Current (blue)
gcloud compute backend-services update ollama-backend-v1 --global

# New (green)
gcloud compute backend-services update ollama-backend-v2 --global

# Switch traffic
gcloud compute url-maps update ollama-url-map \
  --default-service=ollama-backend-v2
```

---

## Maintenance

### Regular Tasks

- [ ] Monitor error rates and latencies
- [ ] Review and update API documentation
- [ ] Rotate API keys periodically
- [ ] Update dependencies
- [ ] Review Cloud Armor rules
- [ ] Analyze performance metrics
- [ ] Check certificate expiration

---

**Version**: 1.0.0  
**Status**: Production Ready  
**Last Updated**: January 12, 2026  
**Support**: Contact elevatediq.ai engineering
