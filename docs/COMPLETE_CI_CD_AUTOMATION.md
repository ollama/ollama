# Complete CI/CD Automation Guide

## Overview

This guide covers the fully-automated CI/CD pipeline for Ollama, including:

- **Complete automation** for deploy, teardown, and restore
- **GitHub Actions workflows** for continuous integration and deployment
- **Local development automation** for rapid iteration
- **Continuous monitoring** for production systems
- **Disaster recovery** procedures

---

## 🚀 Quick Start

### Local Development (5 minutes)

```bash
# Start everything locally
./scripts/local-dev-automation.sh start

# Run tests
./scripts/local-dev-automation.sh test

# Full reset with fresh data
./scripts/local-dev-automation.sh reset

# View available services
./scripts/local-dev-automation.sh ports
```

### Production Deployment (Automated)

```bash
# Deploy to production (via GitHub Actions)
git push origin main

# Trigger manual deployment
gh workflow run full-ci-cd.yml \
  -f environment=prod \
  -f action=deploy
```

### Infrastructure Lifecycle

```bash
# Deploy new infrastructure
./scripts/infrastructure-lifecycle.sh prod deploy

# Backup before critical operations
./scripts/infrastructure-lifecycle.sh prod backup

# Full teardown with backups
./scripts/infrastructure-lifecycle.sh prod teardown

# Restore from latest backup
./scripts/infrastructure-lifecycle.sh prod restore

# Full cycle: deploy → test → teardown → restore
./scripts/infrastructure-lifecycle.sh prod full-cycle
```

---

## 📋 CI/CD Pipeline Architecture

### GitHub Actions Workflow: `full-ci-cd.yml`

```
┌─────────────────────────────────────────────────────────────┐
│           PHASE 1: VALIDATION & SECURITY SCANS              │
│  - Folder structure validation                              │
│  - Landing Zone compliance check                            │
│  - Linting, type checking                                   │
│  - Unit tests + coverage                                    │
│  - Security audit (pip-audit, Snyk)                         │
│  - Terraform validation (tfsec)                             │
└───────────────────┬─────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────┐
│            PHASE 2: BUILD ARTIFACTS                         │
│  - Build Docker image                                       │
│  - Push to GCR                                              │
│  - Cache Docker layers                                      │
└───────────────────┬─────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────┐
│      PHASE 3: DEPLOYMENT DECISION ENGINE                    │
│  - Determine action (deploy/teardown/restore)               │
│  - Select environment (dev/staging/prod)                    │
│  - Check branch protection                                  │
└───────┬────────────────────────────────────────────┬────────┘
        │                                            │
        │ DEPLOY                                    │ TEARDOWN/RESTORE
        │                                            │
┌───────▼──────────────────┐       ┌────────────────▼──────┐
│   PHASE 4A: DEPLOY       │       │ PHASE 4B/C: TEARDOWN  │
│  - Terraform apply       │       │ - Backup all data     │
│  - Deploy to Cloud Run   │       │ - Terraform destroy   │
│  - Setup LB & DNS        │       │ - Cleanup resources   │
│  - Setup monitoring      │       │ - Remove DNS entries  │
│  - Health checks         │       │ or RESTORE:           │
└───────┬──────────────────┘       │ - Terraform apply     │
        │                          │ - Restore data        │
        │                          │ - Verify health       │
        │                  ┌───────▼─────────────────┐
        │                  │ PHASE 5: POST-DEPLOY    │
        └─────────────────►│ - Smoke tests            │
                           │ - Health checks          │
                           │ - Load testing (k6)      │
                           └───────┬─────────────────┘
                                   │
                           ┌───────▼──────────────────┐
                           │ PHASE 6: NOTIFICATIONS   │
                           │ - Slack alert            │
                           │ - GitHub deployment      │
                           │ - Success summary        │
                           └────────────────────────┘
```

---

## 🎯 Automation Scripts

### 1. Infrastructure Lifecycle (`infrastructure-lifecycle.sh`)

Complete infrastructure management with full disaster recovery.

**Usage:**

```bash
./scripts/infrastructure-lifecycle.sh <environment> <action> [--dry-run]

# Arguments:
#   environment: dev|staging|prod
#   action: deploy|teardown|restore|full-cycle
#   --dry-run: Show what would happen without making changes
```

**Examples:**

```bash
# Deploy to production
./scripts/infrastructure-lifecycle.sh prod deploy

# Dry-run teardown
./scripts/infrastructure-lifecycle.sh staging teardown --dry-run

# Restore from backup
./scripts/infrastructure-lifecycle.sh prod restore

# Full cycle test (deploy → test → teardown → restore)
./scripts/infrastructure-lifecycle.sh dev full-cycle
```

**What it does:**

**Deploy:**

1. Validates prerequisites (GCP credentials, tools)
2. Creates full backup
3. Initializes & validates Terraform
4. Plans & applies infrastructure
5. Deploys to Cloud Run
6. Configures load balancer & monitoring
7. Runs health checks
8. Validates deployment

**Teardown:**

1. Creates full backup (CRITICAL!)
2. Stops Cloud Run services
3. Removes load balancer
4. Destroys all Terraform resources
5. Cleans up DNS entries
6. Removes Cloud Storage buckets

**Restore:**

1. Deploys infrastructure from IaC
2. Restores PostgreSQL database
3. Restores Qdrant vectors
4. Restores Cloud Storage
5. Verifies health checks

**Full-Cycle:**

1. Deploys full stack
2. Validates deployment
3. Tears down (with backups)
4. Restores from backups
5. Verifies restoration

### 2. Local Development Automation (`local-dev-automation.sh`)

Complete local environment automation with Docker Compose.

**Usage:**

```bash
./scripts/local-dev-automation.sh <action> [options]

# Actions:
#   start, stop, restart, logs, ps
#   setup, reset, teardown
#   test, lint, type-check, all-checks
#   health, shell [service], ports, metrics
#   migrate, seed, reset-db
```

**Examples:**

```bash
# Start development environment
./scripts/local-dev-automation.sh start

# Run all quality checks
./scripts/local-dev-automation.sh all-checks

# Open shell to PostgreSQL container
./scripts/local-dev-automation.sh shell postgres

# Reset database
./scripts/local-dev-automation.sh reset-db

# Full setup from scratch
./scripts/local-dev-automation.sh setup
```

**What it does:**

**Start:**

1. Validates Docker installation
2. Creates .env files
3. Starts all containers
4. Checks service health
5. Shows available endpoints

**Setup:**

1. Full Docker cleanup
2. Rebuild services
3. Run migrations
4. Seed test data
5. Run all tests
6. Show port mappings

**Reset:**

1. Stops containers
2. Resets database
3. Restarts containers
4. Verifies health

### 3. Continuous Monitoring (`continuous-monitoring.sh`)

Real-time health checks and metrics collection.

**Usage:**

```bash
./scripts/continuous-monitoring.sh <environment> <action> [options]

# Actions:
#   check: Single health check cycle
#   continuous [N]: Continuous monitoring (N=interval in seconds)
#   health, alerts, dashboard, history
#   api, database, redis, system, prometheus
```

**Examples:**

```bash
# Single health check
./scripts/continuous-monitoring.sh prod check

# Continuous monitoring (60-second interval)
./scripts/continuous-monitoring.sh prod continuous 60

# Show monitoring dashboard
./scripts/continuous-monitoring.sh prod dashboard

# Show recent alerts
./scripts/continuous-monitoring.sh prod alerts
```

**What it does:**

**Check:**

- API health & endpoint validation
- Database connectivity & connection pool
- Redis health
- System resources (CPU, memory, disk)
- Prometheus metrics (requests, errors, latency, cache)

**Continuous:**

- Runs checks on interval
- Generates health reports
- Collects metrics
- Sends alerts to Slack
- Updates CloudWatch logs

**Dashboard:**

- Real-time status display
- Auto-refreshes every 10 seconds
- Shows API, DB, system status
- Displays recent alerts

---

## 🔄 GitHub Actions Workflow

### Trigger Conditions

```yaml
# Automatically triggered:
push:
  branches: [main, develop, staging]

pull_request:
  branches: [main, develop]

# Manual trigger with options:
workflow_dispatch:
  environment: dev|staging|prod
  action: deploy|teardown|restore|full-cycle
```

### Environment Mapping

| Branch    | Environment   | Action        |
| --------- | ------------- | ------------- |
| `main`    | `prod`        | `deploy`      |
| `staging` | `staging`     | `deploy`      |
| `develop` | `dev`         | `deploy`      |
| Manual    | User selected | User selected |

### Concurrency Control

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}-${{ inputs.environment }}
  cancel-in-progress: false
```

- One deployment per environment at a time
- Prevents concurrent deployments
- Previous jobs continue if new push received

---

## 📊 Monitoring & Alerts

### Thresholds

| Metric            | Threshold | Alert    |
| ----------------- | --------- | -------- |
| API Response Time | > 500ms   | WARNING  |
| Error Rate        | > 1%      | WARNING  |
| CPU Usage         | > 80%     | WARNING  |
| Memory Usage      | > 85%     | WARNING  |
| Disk Usage        | > 90%     | CRITICAL |
| DB Connections    | > 100     | WARNING  |

### Alert Destinations

- **Slack**: Real-time notifications
- **CloudWatch**: Persistent logging
- **GitHub**: Deployment status
- **Prometheus**: Metrics collection
- **Grafana**: Visualization

---

## 🔒 Disaster Recovery

### Backup Strategy

**Automated Backups** (Before every deployment):

```bash
./scripts/infrastructure-lifecycle.sh prod deploy
# Automatically creates:
# - PostgreSQL dump: backups/postgres-prod-TIMESTAMP.sql.gz
# - Qdrant backup: backups/qdrant-prod-TIMESTAMP.tar.gz
# - GCS sync: gs://project-ollama-backup-prod/
# - Terraform state: backups/terraform-state-prod-TIMESTAMP.tar.gz
```

**Manual Backup**:

```bash
./scripts/infrastructure-lifecycle.sh prod backup
```

**Cron-based Backups** (Daily):

```bash
./scripts/setup-cron-backup.sh --environment=prod --frequency=daily
```

### Recovery Procedures

**Full Infrastructure Restore**:

```bash
# Restore everything from backup
./scripts/infrastructure-lifecycle.sh prod restore
```

**Partial Restore - Database Only**:

```bash
bash scripts/restore-postgres.sh \
  --environment=prod \
  --backup=backups/postgres-prod-TIMESTAMP.sql.gz
```

**Partial Restore - Vectors Only**:

```bash
bash scripts/restore-qdrant.sh \
  --environment=prod \
  --backup=backups/qdrant-prod-TIMESTAMP.tar.gz
```

---

## 🧪 Testing & Validation

### Local Testing

```bash
# Run unit tests
./scripts/local-dev-automation.sh test

# Run type checking
./scripts/local-dev-automation.sh type-check

# Run linting
./scripts/local-dev-automation.sh lint

# Run all checks
./scripts/local-dev-automation.sh all-checks
```

### Post-Deployment Testing

Automatic smoke tests run after deployment:

```bash
# API health checks
bash scripts/verify-production-health.sh --environment=prod

# Load testing (k6)
k6 run load-tests/k6-load-test.js \
  --vus=10 --duration=60s

# Integration tests
pytest tests/integration/ -v
```

---

## 📝 Configuration

### Environment Variables

**GCP Configuration**:

```bash
export GCP_PROJECT_ID="your-project-id"
export GCP_REGION="us-central1"
export TF_STATE_BUCKET="your-project-tf-state"
```

**Monitoring**:

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/..."
export PROMETHEUS_ENDPOINT="http://localhost:9090"
export GRAFANA_ENDPOINT="http://localhost:3000"
```

**API**:

```bash
export API_ENDPOINT="https://elevatediq.ai/ollama"
```

### Secrets (GitHub Actions)

```bash
# Required secrets in GitHub Actions:
GCP_SA_KEY              # GCP Service Account JSON
GCP_PROJECT_ID          # GCP Project ID
TF_STATE_BUCKET         # Terraform state bucket
SLACK_WEBHOOK_URL       # Slack webhook (optional)
SNYK_TOKEN              # Snyk security scanning
```

---

## 🚨 Troubleshooting

### Deployment Fails

1. **Check logs**:

   ```bash
   # GitHub Actions logs (in UI)
   # or
   gh run view <run-id> --log
   ```

2. **Local dry-run**:

   ```bash
   ./scripts/infrastructure-lifecycle.sh prod deploy --dry-run
   ```

3. **Manual Terraform**:
   ```bash
   cd docker/terraform
   terraform plan -var="environment=prod"
   ```

### Health Check Failures

1. **Run manual health check**:

   ```bash
   ./scripts/continuous-monitoring.sh prod check
   ```

2. **Check API**:

   ```bash
   curl -v https://elevatediq.ai/ollama/api/v1/health
   ```

3. **Check database**:
   ```bash
   psql postgresql://user:pass@host/ollama -c "SELECT 1"
   ```

### Restore Issues

1. **List available backups**:

   ```bash
   ls -lh ./backups/
   ```

2. **Manual database restore**:

   ```bash
   bash scripts/restore-postgres.sh \
     --environment=prod \
     --backup=backups/postgres-prod-TIMESTAMP.sql.gz
   ```

3. **Check backup integrity**:
   ```bash
   gunzip -t backups/postgres-prod-TIMESTAMP.sql.gz
   ```

---

## 📚 Additional Resources

- **Terraform Modules**: `docker/terraform/`
- **Docker Compose**: `docker/docker-compose.*.yml`
- **Scripts**: `scripts/`
- **Tests**: `tests/`
- **Documentation**: `docs/`

---

## ✅ Checklist for Production Deployment

- [ ] All tests passing locally
- [ ] Security audit clean
- [ ] Landing Zone compliance verified
- [ ] Backups created
- [ ] Monitoring configured
- [ ] Alerts configured
- [ ] DNS records ready
- [ ] Load Balancer configured
- [ ] SSL certificates valid
- [ ] Disaster recovery tested
- [ ] Team notified
- [ ] Deployment window scheduled
- [ ] Rollback plan documented

---

## 🎓 Training & Onboarding

### For New Team Members

1. **Local Setup** (15 min):

   ```bash
   ./scripts/local-dev-automation.sh setup
   ```

2. **Understand Workflows** (30 min):
   - Read `.github/workflows/full-ci-cd.yml`
   - Review automation scripts

3. **Test Deployment** (30 min):

   ```bash
   # Test in dev environment first
   ./scripts/infrastructure-lifecycle.sh dev deploy --dry-run
   ```

4. **Understand Monitoring** (20 min):
   ```bash
   ./scripts/continuous-monitoring.sh dev dashboard
   ```

---

**Document Version**: 1.0
**Last Updated**: January 18, 2026
**Maintained By**: Engineering Team
**Status**: Production Ready ✅
