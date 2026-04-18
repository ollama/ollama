# Agentic GCP Deployment Guide

**Status**: ✅ Production-Ready  
**Version**: 0.1.0  
**Date**: January 26, 2026

---

## Overview

This guide walks through deploying the Ollama agentic infrastructure to GCP following the Landing Zone 8-point mandate.

### What Gets Deployed

- **Cloud Run Services**: Agent execution + Orchestrator
- **Cloud Tasks Queue**: Task distribution and retry logic
- **Firestore Database**: Agent state and conversation history
- **Pub/Sub Topics**: Result streaming and dead-letter queue
- **BigQuery Dataset**: Execution logs and analytics
- **Monitoring Dashboards**: Real-time performance tracking
- **IAM & Workload Identity**: Zero-trust authentication
- **Service Accounts**: Least-privilege access control

---

## Prerequisites

### GCP Project Setup

```bash
# 1. Set project
export GCP_PROJECT="gcp-eiq"
gcloud config set project $GCP_PROJECT

# 2. Enable required APIs
gcloud services enable \
  run.googleapis.com \
  cloudtasks.googleapis.com \
  firestore.googleapis.com \
  pubsub.googleapis.com \
  bigquery.googleapis.com \
  monitoring.googleapis.com \
  logging.googleapis.com \
  artifactregistry.googleapis.com \
  cloudkms.googleapis.com \
  secretmanager.googleapis.com

# 3. Verify APIs enabled
gcloud services list --enabled | grep -E "run|tasks|firestore|pubsub|bigquery"
```

### Local Tools

```bash
# Terraform
terraform version  # Must be >= 1.0

# Google Cloud SDK
gcloud version

# Docker (for building images)
docker version

# Python
python3 --version  # Must be >= 3.11
```

### KMS Keys (for CMEK encryption)

```bash
# Create KMS keyring
gcloud kms keyrings create prod-keys --location us-central1

# Create encryption keys
gcloud kms keys create artifact-key \
  --location us-central1 \
  --keyring prod-keys \
  --purpose encryption

gcloud kms keys create pubsub-key \
  --location us-central1 \
  --keyring prod-keys \
  --purpose encryption

gcloud kms keys create bq-key \
  --location us-central1 \
  --keyring prod-keys \
  --purpose encryption
```

---

## Step 1: Build & Push Container Images

### Build Agent Service Image

```bash
# Navigate to ollama root
cd /home/akushnir/ollama

# Build agent service Docker image
docker build \
  --file docker/Dockerfile \
  --tag us-central1-docker.pkg.dev/$GCP_PROJECT/prod-ollama-agents/agent:0.1.0 \
  --tag us-central1-docker.pkg.dev/$GCP_PROJECT/prod-ollama-agents/agent:latest \
  --build-arg SERVICE=agents \
  .

# Push to Artifact Registry
docker push us-central1-docker.pkg.dev/$GCP_PROJECT/prod-ollama-agents/agent:0.1.0
docker push us-central1-docker.pkg.dev/$GCP_PROJECT/prod-ollama-agents/agent:latest
```

### Build Orchestrator Service Image

```bash
# Build orchestrator service Docker image
docker build \
  --file docker/Dockerfile \
  --tag us-central1-docker.pkg.dev/$GCP_PROJECT/prod-ollama-agents/orchestrator:0.1.0 \
  --tag us-central1-docker.pkg.dev/$GCP_PROJECT/prod-ollama-agents/orchestrator:latest \
  --build-arg SERVICE=orchestrator \
  .

# Push to Artifact Registry
docker push us-central1-docker.pkg.dev/$GCP_PROJECT/prod-ollama-agents/orchestrator:0.1.0
docker push us-central1-docker.pkg.dev/$GCP_PROJECT/prod-ollama-agents/orchestrator:latest
```

---

## Step 2: Create Terraform Variables

```bash
# Create terraform.tfvars
cat > docker/terraform/04-agentic/terraform.tfvars <<'EOF'
project_id = "gcp-eiq"
gcp_project = "gcp-eiq"
region      = "us-central1"
environment = "production"

# Container Images
agent_image_uri       = "us-central1-docker.pkg.dev/gcp-eiq/prod-ollama-agents/agent:latest"
orchestrator_image_uri = "us-central1-docker.pkg.dev/gcp-eiq/prod-ollama-agents/orchestrator:latest"

# KMS Keys
artifact_kms_key = "projects/gcp-eiq/locations/us-central1/keyRings/prod-keys/cryptoKeys/artifact-key"
pubsub_kms_key   = "projects/gcp-eiq/locations/us-central1/keyRings/prod-keys/cryptoKeys/pubsub-key"
bq_kms_key       = "projects/gcp-eiq/locations/us-central1/keyRings/prod-keys/cryptoKeys/bq-key"

# Monitoring
slack_channel = "#prod-ollama-agents-alerts"
ollama_service_url = "http://ollama:11434"

# Mandatory 24 Labels (Landing Zone)
resource_labels = {
  environment         = "production"
  cost_center         = "ai-infrastructure"
  team                = "ai-platform"
  managed_by          = "terraform"
  created_by          = "infrastructure-team@elevatediq.ai"
  created_date        = "2026-01-26"
  lifecycle_state     = "active"
  teardown_date       = "none"
  retention_days      = "3650"
  product             = "ollama"
  component           = "agents"
  tier                = "critical"
  compliance          = "fedramp"
  version             = "0.1.0"
  stack               = "python-3.11-fastapi-gcp"
  backup_strategy     = "continuous"
  monitoring_enabled  = "true"
  budget_owner        = "infrastructure-team@elevatediq.ai"
  project_code        = "OLLAMA-2026-001"
  monthly_budget_usd  = "5000"
  chargeback_unit     = "ai-infrastructure"
  git_repository      = "github.com/kushin77/ollama"
  git_branch          = "main"
  auto_delete         = "false"
}
EOF
```

---

## Step 3: Validate Terraform Configuration

```bash
# Initialize Terraform
cd docker/terraform/04-agentic
terraform init

# Validate configuration
terraform validate

# Format code
terraform fmt -recursive

# Run linter
tflint

# Plan deployment (review before applying)
terraform plan -out=tfplan
```

### Review Terraform Plan

The plan should show:
- ✅ Cloud Run services (agents + orchestrator)
- ✅ Cloud Tasks queue
- ✅ Firestore database
- ✅ Pub/Sub topics and subscriptions
- ✅ BigQuery dataset
- ✅ Service accounts and IAM bindings
- ✅ Monitoring policies and alerts

---

## Step 4: Apply Terraform Configuration

```bash
# Deploy agentic infrastructure
terraform apply tfplan

# Capture outputs for next steps
terraform output -json > agentic_deployment.json
```

### Verify Deployment

```bash
# Check Cloud Run services
gcloud run services list --filter="prod-ollama-agents or prod-ollama-orchestrator"

# Check service accounts
gcloud iam service-accounts list --filter="agents or orchestrator"

# Check Cloud Tasks queue
gcloud tasks queues describe prod-ollama-agent-tasks --location us-central1

# Check Firestore database
gcloud firestore databases list

# Check BigQuery dataset
bq ls -d prod_ollama_agents
```

---

## Step 5: Create Secret Manager Secrets

```bash
# Create secrets for agent configuration
gcloud secrets create ollama-agents-config \
  --data-file=- <<EOF
{
  "agents": [
    {
      "agent_id": "agent-reasoning-v1",
      "model": "llama3.2",
      "capabilities": ["reasoning", "planning", "tool_use"],
      "max_tokens": 2048
    },
    {
      "agent_id": "agent-research-v1",
      "model": "neural-chat",
      "capabilities": ["memory", "multi_step"],
      "max_tokens": 4096
    }
  ]
}
EOF

# Create orchestrator configuration
gcloud secrets create ollama-orchestrator-config \
  --data-file=- <<EOF
{
  "max_concurrent_tasks": 100,
  "queue_name": "prod-ollama-agent-tasks",
  "agents_service_url": "https://prod-ollama-agents-service-<hash>-uc.a.run.app",
  "retry_config": {
    "max_attempts": 3,
    "initial_backoff": "1s",
    "max_backoff": "60s"
  }
}
EOF

# Grant service accounts access to secrets
gcloud secrets add-iam-policy-binding ollama-agents-config \
  --member=serviceAccount:prod-ollama-agents@gcp-eiq.iam.gserviceaccount.com \
  --role=roles/secretmanager.secretAccessor

gcloud secrets add-iam-policy-binding ollama-orchestrator-config \
  --member=serviceAccount:prod-ollama-orchestrator@gcp-eiq.iam.gserviceaccount.com \
  --role=roles/secretmanager.secretAccessor
```

---

## Step 6: Configure Cloud Load Balancer

The agentic service is exposed through the existing GCP Load Balancer:

```bash
# Get service URLs from Terraform output
AGENTS_SERVICE_URL=$(terraform output -raw agents_service_url)
ORCHESTRATOR_SERVICE_URL=$(terraform output -raw orchestrator_service_url)

# These are automatically routed through the GCP LB:
# https://elevatediq.ai/ollama/api/v1/agents (agents service)
# https://elevatediq.ai/ollama/api/v1/orchestrator (orchestrator service)

echo "Agent Service: $AGENTS_SERVICE_URL"
echo "Orchestrator: $ORCHESTRATOR_SERVICE_URL"
```

---

## Step 7: Validate Agentic Infrastructure

### Run Compliance Validation

```bash
# Validate Landing Zone compliance
python scripts/validate_landing_zone_compliance.py --strict

# Validate Terraform labels
python scripts/validate_landing_zone_compliance.py --terraform
```

### Run Health Checks

```bash
# Check agent service health
curl -H "Authorization: Bearer $API_KEY" \
  https://elevatediq.ai/ollama/api/v1/agents/health

# Check orchestrator health
curl -H "Authorization: Bearer $API_KEY" \
  https://elevatediq.ai/ollama/api/v1/orchestrator/health

# List available agents
curl -H "Authorization: Bearer $API_KEY" \
  https://elevatediq.ai/ollama/api/v1/agents
```

### Run Smoke Tests

```bash
# Create a test task
curl -X POST \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "agent-reasoning-v1",
    "prompt": "What is 2+2?",
    "context": {"max_tokens": 100}
  }' \
  https://elevatediq.ai/ollama/api/v1/agents/agent-reasoning-v1/execute

# Poll for result
curl -H "Authorization: Bearer $API_KEY" \
  https://elevatediq.ai/ollama/api/v1/agents/tasks/{task_id}
```

---

## Step 8: Configure Monitoring & Alerting

### Create Monitoring Dashboards

```bash
# Dashboards are auto-created by Terraform
# Access in Cloud Console:
gcloud monitoring dashboards list --filter="agentic or agents"
```

### Test Alert Policies

```bash
# Verify alert policies are active
gcloud alpha monitoring policies list --filter="agentic"

# View recent alerts
gcloud logging read --limit 10 "severity >= WARNING"
```

---

## Post-Deployment Checklist

- [ ] All Cloud Run services are green (healthy)
- [ ] Service accounts created with proper IAM roles
- [ ] Firestore database initialized and accessible
- [ ] Pub/Sub topics and subscriptions created
- [ ] BigQuery dataset with execution table created
- [ ] KMS keys operational and used for encryption
- [ ] Monitoring dashboards displaying data
- [ ] Alert policies configured and tested
- [ ] Health checks passing (agent + orchestrator)
- [ ] Smoke test completed successfully
- [ ] Landing Zone compliance validation passing
- [ ] Logs flowing to Cloud Logging
- [ ] Metrics exported to Cloud Monitoring
- [ ] All 24 mandatory labels applied to resources
- [ ] Terraform state secured in GCS (CMEK)

---

## Rollback Procedure

If issues occur, rollback is simple:

```bash
# Destroy agentic infrastructure
cd docker/terraform/04-agentic
terraform destroy

# Confirm destruction:
# - Cloud Run services deleted
# - Cloud Tasks queue deleted
# - Firestore database deleted (data retained in backups)
# - Pub/Sub topics deleted
# - BigQuery dataset deleted (data retained if configured)
# - Service accounts deleted

# Restart with clean state
terraform init
terraform apply tfplan
```

---

## Troubleshooting

### Issue: Service Account Lacks Permissions

**Symptom**: 403 Forbidden errors when service calls other GCP services

**Solution**:
```bash
# Grant missing roles to service account
gcloud projects add-iam-policy-binding $GCP_PROJECT \
  --member=serviceAccount:prod-ollama-agents@gcp-eiq.iam.gserviceaccount.com \
  --role=roles/logging.logWriter
```

### Issue: Images Not Found in Artifact Registry

**Symptom**: Cloud Run service stuck in pending state

**Solution**:
```bash
# Push images to Artifact Registry
docker push us-central1-docker.pkg.dev/$GCP_PROJECT/prod-ollama-agents/agent:latest

# Update Terraform with correct image URI
# Re-apply Terraform configuration
```

### Issue: Firestore Quota Exceeded

**Symptom**: Firestore operations return quota exceeded errors

**Solution**:
```bash
# Check current usage
gcloud firestore usage describe

# Request quota increase in Cloud Console
# (Search for "Quotas" in Cloud Console)
```

---

## Cost Estimation

Based on typical usage (assuming 1000 tasks/day, average 5s per task):

| Component | Monthly Cost | Notes |
|-----------|--------------|-------|
| Cloud Run (agents) | ~$50 | Auto-scales, ~250 vCPU-hours/month |
| Cloud Run (orchestrator) | ~$20 | Lighter workload, ~100 vCPU-hours/month |
| Cloud Tasks | ~$10 | ~30K task executions |
| Firestore | ~$15 | ~1GB storage, 100K document reads |
| Pub/Sub | ~$5 | ~100GB ingestion |
| BigQuery | ~$10 | Queries and storage |
| Cloud Logging | ~$5 | Log ingestion and retention |
| Cloud Monitoring | ~$5 | Metrics and alerting |
| **Total** | **~$120** | Varies by usage |

---

## References

- [Cloud Run Deployment Guide](https://cloud.google.com/run/docs/deploying)
- [Terraform GCP Provider](https://registry.terraform.io/providers/hashicorp/google/latest/docs)
- [Landing Zone Standards](./docs/LANDING_ZONE_QUICK_REFERENCE.md)
- [API Documentation](./docs/agents/API.md)

---

**Last Updated**: January 26, 2026  
**Status**: ✅ Production-Ready (Landing Zone Compliant)
