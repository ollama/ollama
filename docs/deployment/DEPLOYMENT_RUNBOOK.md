# 📖 OLLAMA ELITE AI PLATFORM - COMPREHENSIVE DEPLOYMENT RUNBOOK

**Version**: 2.0.0 | **Date**: January 13, 2026 | **Status**: 🟢 PRODUCTION READY

---

## 🎯 Complete Deployment Guide

This runbook covers the entire Ollama Elite AI Platform deployment from zero to production-grade system with monitoring, CI/CD, and scalability.

---

## 📋 Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Initial Setup](#initial-setup)
3. [Infrastructure Setup](#infrastructure-setup)
4. [Application Deployment](#application-deployment)
5. [Git Hooks & CI/CD](#git-hooks--cicd)
6. [Database Setup](#database-setup)
7. [Monitoring & Alerting](#monitoring--alerting)
8. [DNS Configuration](#dns-configuration)
9. [Verification & Testing](#verification--testing)
10. [Troubleshooting](#troubleshooting)
11. [Post-Deployment](#post-deployment)

---

## Pre-Deployment Checklist

Before starting deployment, verify:

- [ ] GCP account with appropriate permissions
- [ ] Docker installed locally
- [ ] gcloud CLI configured
- [ ] kubectl installed (for Kubernetes)
- [ ] Git repository initialized
- [ ] Environment variables documented
- [ ] Database credentials prepared
- [ ] DNS domain registered and accessible
- [ ] SSL certificate requirements understood
- [ ] Team members notified

---

## Initial Setup

### 1. Clone Repository

```bash
git clone https://github.com/kushin77/ollama.git
cd ollama
```

### 2. Create Python Virtual Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip setuptools wheel
pip install -e .
pip install -r requirements/dev.txt  # Development dependencies
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your configuration
cat > .env << 'EOF'
# GCP Configuration
GCP_PROJECT=elevatediq
GCP_REGION=us-central1

# Application
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=info

# Endpoints
PUBLIC_API_ENDPOINT=https://ollama.elevatediq.ai
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8080

# Database
DATABASE_URL=postgresql+asyncpg://user:password@cloud-sql-proxy:5432/ollama
REDIS_URL=redis://redis:6379/0
QDRANT_URL=http://qdrant:6333

# Services
OLLAMA_BASE_URL=http://ollama:11434

# Security
API_KEY_PREFIX=sk_
REQUIRE_API_KEY=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
EOF
```

### 4. Set up Git Hooks

```bash
chmod +x .githooks/*.sh
cd .githooks
./setup.sh
cd ..

# Verify hooks are installed
ls -la .git/hooks/
```

---

## Infrastructure Setup

### 1. Create GCP Project (if new)

```bash
gcloud projects create ollama-production --name="Ollama Elite AI"
gcloud config set project ollama-production
```

### 2. Enable Required APIs

```bash
gcloud services enable \
  run.googleapis.com \
  sql-component.googleapis.com \
  redis.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com \
  artifactregistry.googleapis.com \
  compute.googleapis.com \
  container.googleapis.com
```

### 3. Create Service Account

```bash
# Create service account
gcloud iam service-accounts create ollama-service \
  --display-name="Ollama AI Service Account"

# Grant required roles
gcloud projects add-iam-policy-binding elevatediq \
  --member=serviceAccount:ollama-service@elevatediq.iam.gserviceaccount.com \
  --role=roles/run.admin

gcloud projects add-iam-policy-binding elevatediq \
  --member=serviceAccount:ollama-service@elevatediq.iam.gserviceaccount.com \
  --role=roles/cloudsql.client

gcloud projects add-iam-policy-binding elevatediq \
  --member=serviceAccount:ollama-service@elevatediq.iam.gserviceaccount.com \
  --role=roles/redis.admin
```

### 4. Create Cloud SQL Instance

```bash
# Create PostgreSQL instance
gcloud sql instances create ollama-db \
  --database-version=POSTGRES_15 \
  --tier=db-f1-micro \
  --region=us-central1 \
  --storage-type=SSD \
  --storage-size=100GB

# Create database
gcloud sql databases create ollama --instance=ollama-db

# Create database user
gcloud sql users create ollama \
  --instance=ollama-db \
  --password=<SECURE_PASSWORD>
```

### 5. Create Redis Instance

```bash
gcloud redis instances create ollama-cache \
  --size=1 \
  --region=us-central1 \
  --redis-version=7.0
```

---

## Application Deployment

### 1. Build Docker Image

```bash
docker build -t gcr.io/elevatediq/ollama:v1.0.0 \
  -f Dockerfile.minimal .

docker tag gcr.io/elevatediq/ollama:v1.0.0 \
  gcr.io/elevatediq/ollama:latest
```

### 2. Push to Container Registry

```bash
# Configure Docker authentication
gcloud auth configure-docker

# Push image
docker push gcr.io/elevatediq/ollama:v1.0.0
docker push gcr.io/elevatediq/ollama:latest
```

### 3. Deploy to Cloud Run

```bash
gcloud run deploy ollama-service \
  --image gcr.io/elevatediq/ollama:v1.0.0 \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --timeout 60 \
  --min-instances 1 \
  --max-instances 5 \
  --set-env-vars "$(cat .env | grep -v '^#' | tr '\n' ',')" \
  --project elevatediq
```

---

## Git Hooks & CI/CD

### 1. Configure Local Git Hooks

```bash
# Initialize hooks
cd .githooks
./setup.sh
cd ..

# Verify hooks work by making a test commit
git add .
git commit -m "test: verify hooks work"  # Will run all checks
```

### 2. Set up GitHub Actions

```bash
# Push to GitHub (workflows auto-run)
git push origin main

# Verify workflows in GitHub UI:
# Settings > Actions > Workflows

# Monitor workflow runs:
# Actions tab in GitHub
```

### 3. Configure Deployment Secrets

```bash
# Add secrets in GitHub Settings > Secrets:
# - GCP_PROJECT: elevatediq
# - WIF_PROVIDER: <Workload Identity Federation provider>
# - WIF_SERVICE_ACCOUNT: <Service account email>
```

---

## Database Setup

### 1. Run Migrations

```bash
# Create migration
alembic revision -m "initial schema"

# Apply migrations
alembic upgrade head

# Verify migrations
alembic current
```

### 2. Create Database User

```python
# Using SQLAlchemy
from ollama.models import User, Base
from ollama.database import engine

# Create all tables
Base.metadata.create_all(engine)

# Create initial admin user
admin = User(
    username="admin",
    email="admin@elevatediq.ai",
    display_name="Admin User",
    password_hash="<bcrypt_hashed_password>",
    is_admin=True
)
```

### 3. Seed Initial Data

```bash
python scripts/seed_models.py  # Populate supported models
python scripts/create_admin.py  # Create admin account
```

---

## Monitoring & Alerting

### 1. Set up Prometheus

```bash
# Deploy Prometheus
helm install prometheus prometheus-community/kube-prometheus-stack \
  --values monitoring/prometheus-values.yml \
  -n monitoring --create-namespace

# Port forward to access UI
kubectl port-forward -n monitoring svc/prometheus-operated 9090:9090
# Access at: http://localhost:9090
```

### 2. Set up Grafana

```bash
# Deploy Grafana
helm install grafana grafana/grafana \
  -n monitoring \
  --set adminPassword=<SECURE_PASSWORD>

# Port forward
kubectl port-forward -n monitoring svc/grafana 3000:80
# Access at: http://localhost:3000
```

### 3. Create Alert Rules

```bash
# Apply alert rules
kubectl apply -f monitoring/alerts.yml

# Verify alerts
kubectl get prometheusrule -n monitoring
```

### 4. Configure Cloud Monitoring

```bash
# Run setup script
chmod +x scripts/setup-monitoring.sh
./scripts/setup-monitoring.sh
```

---

## DNS Configuration

### 1. Create DNS Records

Add CNAME record to your DNS provider:

```
Name:  ollama
Type:  CNAME
Value: ghs.googlehosted.com
TTL:   300
```

### 2. Create Domain Mapping

```bash
gcloud beta run domain-mappings create \
  --service=ollama-service \
  --domain=ollama.elevatediq.ai \
  --region=us-central1 \
  --project=elevatediq
```

### 3. Verify DNS

```bash
# Check DNS propagation
nslookup ollama.elevatediq.ai
dig ollama.elevatediq.ai

# Test endpoint
curl https://ollama.elevatediq.ai/health
```

---

## Verification & Testing

### 1. Health Checks

```bash
# Check service health
curl https://ollama.elevatediq.ai/health

# Check API status
curl https://ollama.elevatediq.ai/api/v1/health

# Check models endpoint
curl https://ollama.elevatediq.ai/api/v1/models

# Check metrics
curl https://ollama.elevatediq.ai/metrics
```

### 2. Run Test Suite

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Full test suite with coverage
pytest tests/ --cov=ollama --cov-report=html
```

### 3. Load Testing

```bash
# Install load testing tool
pip install locust

# Run load test
locust -f load_test.py --host https://ollama.elevatediq.ai

# Generate report
locust -f load_test.py --host https://ollama.elevatediq.ai \
  -u 100 -r 10 --run-time 5m --headless --csv=results
```

---

## Troubleshooting

### Service Not Responding

```bash
# Check Cloud Run status
gcloud run services describe ollama-service \
  --region=us-central1 --project=elevatediq

# Check recent revisions
gcloud run revisions list \
  --region=us-central1 --project=elevatediq

# View logs
gcloud run logs read ollama-service \
  --region=us-central1 --project=elevatediq --limit=50

# Follow logs in real-time
gcloud run logs read ollama-service \
  --region=us-central1 --project=elevatediq --follow
```

### Database Connection Issues

```bash
# Test connection
psql -h <CLOUD_SQL_IP> -U ollama -d ollama -c "SELECT 1"

# Check connection pooling
psql -h <CLOUD_SQL_IP> -U postgres -d postgres \
  -c "SELECT * FROM pg_stat_activity"

# View active connections
SELECT datname, count(*) as connection_count
FROM pg_stat_activity
GROUP BY datname;
```

### Performance Issues

```bash
# Check service metrics
gcloud monitoring time-series list \
  --filter='resource.service_name=ollama-service' \
  --project=elevatediq

# Profile application
python -m cProfile -s cumulative main.py > profile.txt

# View slowest queries
SELECT query, mean_time, calls FROM pg_stat_statements
ORDER BY mean_time DESC LIMIT 10;
```

---

## Post-Deployment

### 1. Backup Configuration

```bash
# Backup database
pg_dump --host <SQL_PROXY> --user ollama ollama > backup.sql

# Backup configurations
tar -czf config-backup.tar.gz monitoring/ k8s/ config/

# Store backups
gsutil cp backup.sql gs://ollama-backups/
gsutil cp config-backup.tar.gz gs://ollama-backups/
```

### 2. Document System

```bash
# Update README with deployment info
# Update runbooks with customizations
# Document any manual changes
# Create postmortems for any incidents
```

### 3. Monitor Regularly

```bash
# Set up daily health checks
0 6 * * * curl -f https://ollama.elevatediq.ai/health || alert_team

# Monitor costs
gcloud billing accounts list
gcloud compute project-info describe --project=elevatediq

# Review logs weekly
# Review metrics weekly
# Review alerts weekly
```

---

## Quick Reference Commands

```bash
# View service
gcloud run services describe ollama-service --region=us-central1

# Scale service
gcloud run services update ollama-service \
  --max-instances 10 --region=us-central1

# Update service image
gcloud run deploy ollama-service \
  --image gcr.io/elevatediq/ollama:latest --region=us-central1

# View real-time logs
gcloud run logs read ollama-service --follow --region=us-central1

# SSH into database
gcloud sql connect ollama-db --user=ollama

# Get service URL
gcloud run services describe ollama-service --region=us-central1 \
  --format 'value(status.url)'

# List all API keys
SELECT * FROM api_keys WHERE revoked = false;

# View usage metrics
SELECT model, COUNT(*) as requests, AVG(latency_ms) as avg_latency
FROM usage WHERE created_at > now() - interval '1 day'
GROUP BY model ORDER BY requests DESC;
```

---

## Support & Escalation

### Emergency Contact

- **On-Call**: [Team contact info]
- **Escalation**: [Escalation procedure]

### Incident Response

1. **Detect**: Alerts trigger in Cloud Monitoring
2. **Assess**: Check logs and metrics
3. **Mitigate**: Apply temporary fix or rollback
4. **Resolve**: Investigate root cause
5. **Document**: Create postmortem

### Maintenance Windows

- **Weekly**: Tuesday 2-3 AM UTC
- **Scheduled**: 2 weeks notice required
- **Emergency**: Immediate

---

**End of Runbook**

For latest updates, see: [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)
