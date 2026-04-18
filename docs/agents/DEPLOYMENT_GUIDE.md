# Agentic GCP Infrastructure - Deployment Guide

**Date**: January 26, 2026
**Status**: ✅ READY FOR DEPLOYMENT
**Audience**: Infrastructure/DevOps engineers

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Pre-Deployment Validation](#pre-deployment-validation)
4. [Container Image Build](#container-image-build)
5. [Terraform Deployment](#terraform-deployment)
6. [Post-Deployment Verification](#post-deployment-verification)
7. [Monitoring Setup](#monitoring-setup)
8. [Troubleshooting](#troubleshooting)
9. [Cost Analysis](#cost-analysis)
10. [Rollback Procedures](#rollback-procedures)

---

## Overview

This guide provides step-by-step instructions for deploying the Agentic GCP infrastructure to production. The deployment includes:

- **24 GCP Resources**: Cloud Run (2), Cloud Tasks (1), Firestore (1), Pub/Sub (2), BigQuery (1), KMS (3), Monitoring (8+)
- **Infrastructure as Code**: Terraform 1.0+ with 4 modules
- **Container Services**: Agent service + Orchestrator service
- **Deployment Time**: ~45 minutes (15 min setup + 10 min terraform + 20 min testing)
- **Monthly Cost**: ~$120 (150 vCPU-hours + storage)

### Architecture

```
┌────────────────────────────────┐
│     External Clients           │
│    (Internet / Partners)       │
└───────────┬────────────────────┘
            │
            │ HTTPS/TLS 1.3+
            ▼
┌────────────────────────────────┐
│   GCP Load Balancer            │
│ https://elevatediq.ai/ollama   │
│ - Auth (API key)               │
│ - Rate limiting                │
│ - DDoS protection              │
└───────────┬────────────────────┘
            │
            │ Mutual TLS 1.3+
            ▼
┌────────────────────────────────┐
│  Docker Container Network      │
├────────────────────────────────┤
│ ✓ Cloud Run: Agent Service     │
│   (prod-ollama-agents-service) │
│   - CPU: 2 | Mem: 2Gi          │
│   - Instances: 0-10            │
│                                │
│ ✓ Cloud Run: Orchestrator      │
│   (prod-ollama-orchestrator)   │
│   - CPU: 1 | Mem: 1Gi          │
│   - Instances: 0-5             │
│                                │
│ ✓ Cloud Tasks: Task Queue      │
│   (prod-ollama-agent-tasks)    │
│   - Concurrency: 100           │
│   - Retry: Exponential backoff │
│                                │
│ ✓ Firestore: State DB          │
│   (prod-ollama-agents)         │
│   - P-I-T Recovery enabled     │
│   - CMEK encrypted             │
│                                │
│ ✓ Pub/Sub: Results Topic       │
│   (prod-ollama-agent-results)  │
│   - Retention: 24h             │
│   - DLQ: prod-ollama-agent-dlq │
│                                │
│ ✓ BigQuery: Analytics          │
│   (prod_ollama_agents)         │
│   - CMEK encrypted             │
│   - 7-year retention           │
└────────────────────────────────┘
```

---

## Prerequisites

### 1. GCP Project Setup

```bash
# Set variables
export GCP_PROJECT_ID="gcp-eiq"
export GCP_REGION="us-central1"
export GCP_ZONE="us-central1-a"

# Authenticate with GCP
gcloud auth login
gcloud config set project ${GCP_PROJECT_ID}

# Verify project
gcloud projects describe ${GCP_PROJECT_ID}
```

### 2. Required Tools

```bash
# Terraform 1.0+ (latest recommended)
terraform version

# Google Cloud SDK
gcloud version

# Docker 24+
docker --version

# Git with GPG signing
git --version
git config --list | grep gpg

# Python 3.12
python3 --version

# Google Cloud CLI plugins
gcloud components install gke-gcloud-auth-plugin
```

### 3. GCP Permissions

Ensure your user/service account has these roles:

```
roles/compute.admin                    # Compute resources
roles/container.admin                  # Cloud Run
roles/cloudtasks.admin                # Cloud Tasks
roles/datastore.admin                 # Firestore
roles/pubsub.admin                    # Pub/Sub
roles/bigquery.admin                  # BigQuery
roles/cloudkms.admin                  # KMS
roles/monitoring.admin                # Cloud Monitoring
roles/logging.admin                   # Cloud Logging
roles/serviceusage.serviceUsageAdmin  # Service usage
```

### 4. Pre-existing Resources

Verify KMS keyrings exist:

```bash
# List KMS keyrings
gcloud kms keyrings list --location=${GCP_REGION}

# Expected keyrings:
# - artifact-keys
# - pubsub-keys
# - bigquery-keys

# If not exists, create them:
gcloud kms keyrings create artifact-keys --location=${GCP_REGION}
gcloud kms keyrings create pubsub-keys --location=${GCP_REGION}
gcloud kms keyrings create bigquery-keys --location=${GCP_REGION}
```

---

## Pre-Deployment Validation

### 1. Code Quality Checks

```bash
cd /home/akushnir/ollama

# Type checking (100% coverage required)
mypy ollama/ --strict
# Expected: No errors

# Linting (ruff)
ruff check ollama/
# Expected: Clean or fixable with --fix

# Tests (≥90% coverage required)
pytest tests/ -v --cov=ollama --cov-report=term-missing
# Expected: All tests pass, coverage ≥90%

# Security audit
pip-audit
# Expected: No vulnerabilities
```

### 2. Terraform Validation

```bash
cd docker/terraform/04-agentic

# Validate Terraform
terraform validate
# Expected: Success

# Check formatting
terraform fmt -check -recursive
# Expected: No changes needed (or apply fmt)

# Lint with tflint
tflint --init
tflint
# Expected: No critical issues
```

### 3. Folder Structure Validation

```bash
cd /home/akushnir/ollama

# Validate Elite Filesystem Standards
python scripts/validate_folder_structure.py --strict
# Expected: No critical errors

# Check specific components
python scripts/validate_folder_structure.py --strict --verbose
```

### 4. Git & Commit Validation

```bash
# Verify GPG signing is enabled
git config user.signingkey
# Expected: Your GPG key ID

# Verify recent commits are signed
git log --oneline --show-signature -5
# Expected: All commits show "gpg: Good signature from..."
```

---

## Container Image Build

### Step 1: Prepare Dockerfile

Ensure `docker/Dockerfile` includes:

```dockerfile
FROM python:3.12-slim

ARG SERVICE=agent
ARG PYTHON_VERSION=3.12

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ollama/ /app/ollama
COPY ollama/agents/ /app/ollama/agents

# Set environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV SERVICE=${SERVICE}

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Entry point
CMD ["uvicorn", "ollama.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 2: Build Agent Service Image

```bash
export GCP_PROJECT_ID="gcp-eiq"
export GCP_REGION="us-central1"
export REGISTRY="${GCP_REGION}-docker.pkg.dev"
export REPO="ollama-agentic"
export TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Build agent service
docker build \
    --build-arg PYTHON_VERSION=3.12 \
    --build-arg SERVICE=agent \
    -t ${REGISTRY}/${GCP_PROJECT_ID}/${REPO}/agent-service:latest \
    -t ${REGISTRY}/${GCP_PROJECT_ID}/${REPO}/agent-service:${TIMESTAMP} \
    -f docker/Dockerfile \
    .

# Verify image
docker inspect ${REGISTRY}/${GCP_PROJECT_ID}/${REPO}/agent-service:latest
```

### Step 3: Build Orchestrator Service Image

```bash
# Build orchestrator service
docker build \
    --build-arg PYTHON_VERSION=3.12 \
    --build-arg SERVICE=orchestrator \
    -t ${REGISTRY}/${GCP_PROJECT_ID}/${REPO}/orchestrator-service:latest \
    -t ${REGISTRY}/${GCP_PROJECT_ID}/${REPO}/orchestrator-service:${TIMESTAMP} \
    -f docker/Dockerfile \
    .

# Verify image
docker inspect ${REGISTRY}/${GCP_PROJECT_ID}/${REPO}/orchestrator-service:latest
```

### Step 4: Push to Artifact Registry

```bash
# Configure Docker authentication
gcloud auth configure-docker ${GCP_REGION}-docker.pkg.dev

# Create repository (if not exists)
gcloud artifacts repositories create ${REPO} \
    --repository-format=docker \
    --location=${GCP_REGION} \
    --description="Ollama Agentic Services" || true

# Push agent service
docker push ${REGISTRY}/${GCP_PROJECT_ID}/${REPO}/agent-service:latest
docker push ${REGISTRY}/${GCP_PROJECT_ID}/${REPO}/agent-service:${TIMESTAMP}

# Push orchestrator service
docker push ${REGISTRY}/${GCP_PROJECT_ID}/${REPO}/orchestrator-service:latest
docker push ${REGISTRY}/${GCP_PROJECT_ID}/${REPO}/orchestrator-service:${TIMESTAMP}

# Verify images in registry
gcloud artifacts docker images list ${REGISTRY}/${GCP_PROJECT_ID}/${REPO}
```

---

## Terraform Deployment

### Step 1: Prepare Terraform Variables

Create `docker/terraform/04-agentic/terraform.tfvars`:

```hcl
# GCP Configuration
gcp_project_id = "gcp-eiq"
gcp_region     = "us-central1"

# Deployment Configuration
environment = "prod"
application = "ollama"

# Service Configuration
agent_service_config = {
  cpu              = "2"
  memory           = "2Gi"
  max_instances    = 10
  timeout_seconds  = 600
  http1_enabled    = true
}

orchestrator_config = {
  cpu              = "1"
  memory           = "1Gi"
  max_instances    = 5
  timeout_seconds  = 300
  http1_enabled    = true
}

# Container Images
agent_service_image       = "us-central1-docker.pkg.dev/gcp-eiq/ollama-agentic/agent-service:latest"
orchestrator_service_image = "us-central1-docker.pkg.dev/gcp-eiq/ollama-agentic/orchestrator-service:latest"

# Database Configuration
firestore_location = "us-central1"
bigquery_location  = "us"

# KMS Encryption Keys
artifact_key_id  = "artifact-key"
pubsub_key_id    = "pubsub-key"
bigquery_key_id  = "bigquery-key"

# Mandatory PMO Labels
mandatory_labels = {
  # Organizational
  environment       = "prod"
  team              = "ai-platform"
  application       = "ollama"
  component         = "agentic-infrastructure"

  # Lifecycle
  owner             = "ai-team@elevatediq.ai"
  cost_center       = "ai-infrastructure"
  manager           = "engineering-manager"
  lifecycle_status  = "active"
  teardown_date     = ""

  # Business
  business_unit     = "ai-products"
  compliance        = "fedramp"
  data_sensitivity  = "internal"
  business_value    = "strategic"

  # Technical
  language          = "python"
  framework         = "fastapi"
  database_type    = "firestore,bigquery"
  version           = "1.0.0"

  # Financial
  cost_model        = "on-demand"
  budget_quarterly  = "5000"
  cost_attribution  = "ai-infrastructure"
  financial_owner   = "finance-team@elevatediq.ai"

  # Git Attribution
  git_repo          = "github.com/kushin77/ollama"
  git_branch        = "main"
  git_commit_id     = "$(git rev-parse HEAD)"
}

# Optional: Custom monitoring configuration
alert_email = "alerts@elevatediq.ai"
```

### Step 2: Initialize Terraform

```bash
cd docker/terraform/04-agentic

# Initialize Terraform
terraform init

# Verify backend
terraform show -no-color | head -20
```

### Step 3: Plan Deployment

```bash
# Generate plan
terraform plan -out=tfplan

# Review plan (should show ~24 resources to create)
# Expected resources:
# - 2x Cloud Run services
# - 1x Cloud Tasks queue
# - 1x Firestore database
# - 2x Pub/Sub topics
# - 1x BigQuery dataset
# - 3x KMS keys
# - 8+ Monitoring policies
# - 4x Service accounts
# + additional supporting resources
```

### Step 4: Apply Terraform

```bash
# Review plan again (CRITICAL STEP)
terraform plan

# Apply with explicit approval
terraform apply tfplan

# Wait for completion (10-15 minutes)
# Monitor output:
# - Check for "Apply complete! Resources: XX added..."
# - Verify all resources created successfully
```

### Step 5: Verify Deployment

```bash
# Get outputs
terraform output

# Expected outputs:
# - agent_service_url
# - orchestrator_service_url
# - firestore_database_name
# - pubsub_topic_names
# - bigquery_dataset_id
# - kms_key_names
# - service_account_emails

# Verify Cloud Run services
gcloud run services list --region=us-central1

# Verify Cloud Tasks queues
gcloud tasks queues list

# Verify Firestore
gcloud firestore databases list

# Verify Pub/Sub topics
gcloud pubsub topics list
```

---

## Post-Deployment Verification

### 1. Health Checks

```bash
# Get agent service URL
AGENT_URL=$(terraform output -raw agent_service_url)

# Test health endpoint (should return 200)
curl -i ${AGENT_URL}/health

# Expected response:
# HTTP/2 200
# {
#   "status": "healthy",
#   "timestamp": "2026-01-26T10:30:00Z",
#   "version": "1.0.0"
# }
```

### 2. API Validation

```bash
# Create test API key
API_KEY="test-key-$(uuidgen)"

# Store in Secret Manager
echo -n "${API_KEY}" | gcloud secrets create ollama-test-key \
    --data-file=-

# Test inference endpoint
curl -X POST ${AGENT_URL}/api/v1/generate \
    -H "Authorization: Bearer ${API_KEY}" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama3.2",
        "prompt": "Hello, world!",
        "max_tokens": 100
    }'

# Expected response:
# {
#   "success": true,
#   "data": {
#     "text": "...",
#     "tokens_used": 42,
#     "inference_time_ms": 1250
#   }
# }
```

### 3. Firestore Validation

```bash
# Query Firestore to verify database
gcloud firestore documents list --collection-ids=agents

# Expected: Returns collection (may be empty if no tasks yet)
```

### 4. Pub/Sub Validation

```bash
# Verify topics and subscriptions
gcloud pubsub topics list
gcloud pubsub subscriptions list

# Test publish (optional)
gcloud pubsub topics publish prod-ollama-agent-results \
    --message='{"test": "message"}'
```

### 5. BigQuery Validation

```bash
# Query BigQuery dataset
bq ls --project_id=gcp-eiq prod_ollama_agents

# Expected: Shows execution_logs table

# Check schema
bq show --schema prod_ollama_agents.execution_logs
```

---

## Monitoring Setup

### 1. Cloud Monitoring Dashboard

```bash
# Create dashboard (optional - Terraform creates one)
gcloud monitoring dashboards create --config-from-file=- << 'EOF'
{
  "displayName": "Ollama Agentic Infrastructure",
  "mosaicLayout": {
    "tiles": [
      {
        "xPos": 0,
        "yPos": 0,
        "width": 6,
        "height": 4,
        "widget": {
          "title": "Agent Service Requests",
          "xyChart": {
            "dataSets": [{
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "metric.type=\"run.googleapis.com/request_count\" resource.type=\"cloud_run_revision\"",
                  "aggregation": {
                    "alignmentPeriod": "60s",
                    "perSeriesAligner": "ALIGN_RATE"
                  }
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

### 2. Alert Policies

Verify auto-created alerts:

```bash
# List alert policies
gcloud alpha monitoring policies list

# Expected policies:
# - Error Rate > 1%
# - P99 Latency > 10s
# - Service Unavailable
```

### 3. Notification Channels

```bash
# Create Slack channel (if not exists)
gcloud alpha monitoring channels create \
    --display-name="Ollama Alerts" \
    --type=slack \
    --channel-labels="channel_name=#prod-ollama-agents-alerts"

# Verify channels
gcloud alpha monitoring channels list
```

---

## Troubleshooting

### Issue: Cloud Run service fails to deploy

**Symptoms**:

```
ERROR: (gcloud.run.deploy) Cloud Run error: Container Timeout
```

**Solutions**:

```bash
# Check Cloud Run logs
gcloud run logs read prod-ollama-agents-service --region=us-central1 --limit=50

# Check image availability
gcloud container images describe ${REGISTRY}/${GCP_PROJECT_ID}/${REPO}/agent-service:latest

# Rebuild and push image
docker build ... -t ${REGISTRY}/${GCP_PROJECT_ID}/${REPO}/agent-service:latest
docker push ${REGISTRY}/${GCP_PROJECT_ID}/${REPO}/agent-service:latest

# Re-apply Terraform
terraform apply
```

### Issue: Firestore connection timeout

**Symptoms**:

```
Error: DEADLINE_EXCEEDED when connecting to Firestore
```

**Solutions**:

```bash
# Verify Firestore database exists
gcloud firestore databases list

# Check network connectivity
gcloud compute networks describe default

# Verify IAM permissions
gcloud projects get-iam-policy ${GCP_PROJECT_ID}

# Restart Cloud Run services
gcloud run services update prod-ollama-agents-service --region=us-central1
```

### Issue: KMS key not found

**Symptoms**:

```
Error: Failed to create object: Key [artifact-key] not found
```

**Solutions**:

```bash
# Create missing KMS key
gcloud kms keyrings create artifact-keys --location=us-central1
gcloud kms keys create artifact-key \
    --location=us-central1 \
    --keyring=artifact-keys \
    --purpose=encryption

# Update Terraform variables
# In terraform.tfvars, update artifact_key_id

# Re-apply Terraform
terraform apply
```

### Issue: Pub/Sub message delivery fails

**Symptoms**:

```
Pub/Sub subscription has dead-letter messages
```

**Solutions**:

```bash
# Check subscription
gcloud pubsub subscriptions describe prod-ollama-agent-results-subscription

# Seek to latest
gcloud pubsub subscriptions seek prod-ollama-agent-results-subscription --time=now

# Check DLQ
gcloud pubsub topics pull prod-ollama-agent-dlq --auto-ack --limit=10
```

---

## Cost Analysis

### Monthly Breakdown (~$120/month)

| Component         | Usage            | Rate                    | Cost            |
| ----------------- | ---------------- | ----------------------- | --------------- |
| **Cloud Run**     | 150 vCPU-hours   | $0.00002400/vCPU-second | $43.20          |
| **Firestore**     | 10GB, 100K ops   | Variable                | $25.00          |
| **Pub/Sub**       | 1M messages, 5GB | Variable                | $15.00          |
| **BigQuery**      | 10GB, 1T scans   | Variable                | $20.00          |
| **Cloud Tasks**   | 50K tasks        | $0.40 per 1M            | $0.02           |
| **Logging**       | 100GB logs       | $0.50 per GB            | $5.00           |
| **Monitoring**    | Basic            | Included                | $0.00           |
| **KMS**           | 3 keys, 50K ops  | Variable                | $6.00           |
| **Load Balancer** | Forwarding rules | $0.025/hour             | $18.00          |
| **Storage**       | 50GB             | $0.020/GB               | $1.00           |
| **Total**         |                  |                         | **~$120/month** |

### Cost Optimization Tips

1. **Auto-scaling**: Set min instances to 0 for dev/staging
2. **Cold start tuning**: Use smaller instances for non-critical services
3. **Batch operations**: Use Cloud Tasks for batch processing
4. **Archive logs**: Move old logs to Cloud Storage (Coldline)
5. **Commit filtering**: Only pay for commits to specific models

---

## Rollback Procedures

### Fast Rollback (< 5 minutes)

```bash
# Revert to previous state
git revert <commit-hash>
git push origin main

# OR hard reset (if not pushed to production)
git reset --hard HEAD~1
git push origin main --force-with-lease

# Redeploy previous version
terraform apply -var="image_tag=previous-stable"
```

### Full Rollback (complete destroy)

```bash
cd docker/terraform/04-agentic

# Plan destruction
terraform plan -destroy -out=destroy_plan

# Review and approve
terraform destroy

# Verify cleanup
gcloud run services list --region=us-central1
# Expected: Services deleted

# Clean up data (if needed)
gcloud firestore databases list
gcloud pubsub topics list
```

---

## Success Criteria

Deployment is successful when:

- ✅ All 24 Terraform resources created
- ✅ Cloud Run services running (health check passing)
- ✅ API endpoints responding (200 OK)
- ✅ Firestore database accessible
- ✅ Pub/Sub topics receiving messages
- ✅ BigQuery dataset ready for queries
- ✅ Monitoring dashboards showing metrics
- ✅ Alert policies active and testable
- ✅ Performance baselines achieved (P99 < 10s)
- ✅ Error rate < 0.1%

---

## Support & Escalation

**For deployment issues**:

- Check logs: `gcloud run logs read [SERVICE_NAME] --region=us-central1`
- Check status: `terraform show`
- Check monitoring: GCP Console → Cloud Monitoring → Dashboards

**For security concerns**:

- Review IAM: `gcloud projects get-iam-policy ${GCP_PROJECT_ID}`
- Audit logs: `gcloud logging read --resource-names=...`
- Security summary: GCP Console → Security → Overview

---

**Document Version**: 1.0.0
**Last Updated**: January 26, 2026
**Maintained By**: AI Platform Team
**Status**: ✅ PRODUCTION READY
