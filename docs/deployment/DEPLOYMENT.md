# Ollama Deployment Guide

**Version**: 0.1.0
**Last Updated**: January 18, 2026
**Target Environments**: Development, Staging, Production

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Configuration Management](#configuration-management)
- [Secrets Management](#secrets-management)
- [Database Migrations](#database-migrations)
- [Deployment Procedures](#deployment-procedures)
  - [Local Development](#local-development)
  - [Staging Deployment](#staging-deployment)
  - [Production Deployment](#production-deployment)
- [Health Checks & Verification](#health-checks--verification)
- [Rollback Procedures](#rollback-procedures)
- [Troubleshooting](#troubleshooting)
- [Production Checklist](#production-checklist)

---

## Prerequisites

### System Requirements

**Hardware**:
- CPU: 4+ cores (8+ recommended for production)
- RAM: 8GB minimum (16GB+ recommended)
- Disk: 50GB+ available space
- Network: 1 Gbps+ connection
- GPU: Optional (NVIDIA with CUDA support for faster inference)

**Software**:
- Docker: 24.0+
- Docker Compose: 2.20+
- Python: 3.11+ (for local development)
- Git: 2.40+ with GPG signing configured
- curl/wget: For health checks

### Access Requirements

**Production**:
- GCP Project access: `gcp-landing-zone`
- Service account credentials
- GCP Secret Manager access
- GitHub repository access (kushin77/ollama)

**Development**:
- Local Docker daemon running
- Network access to Docker Hub
- 10GB+ disk space for Docker images

---

## Environment Setup

### 1. Clone Repository

```bash
# Clone repository
git clone https://github.com/kushin77/ollama.git
cd ollama

# Verify folder structure
ls -la
# Expected: ollama/, tests/, docs/, docker/, config/, scripts/
```

### 2. Configure Environment

#### Development Environment

```bash
# Copy environment template
cp .env.example .env.dev

# Get your real local IP (NEVER use localhost)
export REAL_IP=$(hostname -I | awk '{print $1}')  # Linux
# OR for macOS
export REAL_IP=$(ipconfig getifaddr en0)

# Edit .env.dev
cat > .env.dev <<EOF
# Environment
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=debug

# API Configuration (use real IP, NEVER localhost)
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
PUBLIC_API_URL=http://$REAL_IP:8000
API_KEY=dev-key-for-testing-only

# Database
DATABASE_URL=postgresql://ollama_user:dev_password@postgres:5432/ollama_dev
POSTGRES_USER=ollama_user
POSTGRES_PASSWORD=dev_password
POSTGRES_DB=ollama_dev

# Redis Cache
REDIS_URL=redis://redis:6379/0
REDIS_PASSWORD=dev_redis_password

# Qdrant Vector Database
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=dev_qdrant_key

# Ollama
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODELS_PATH=/root/.ollama/models

# Monitoring
PROMETHEUS_ENABLED=false
JAEGER_ENABLED=false
EOF

# Source environment
export $(cat .env.dev | grep -v '^#' | xargs)
```

#### Production Environment

```bash
# Production uses GCP Secret Manager (NEVER .env files)
# Secrets managed via GCP console or Terraform

# Example secret structure in GCP Secret Manager:
# projects/<project-id>/secrets/ollama-database-url
# projects/<project-id>/secrets/ollama-redis-password
# projects/<project-id>/secrets/ollama-api-keys
```

---

## Configuration Management

### Configuration Hierarchy

1. **Default values** (code)
2. **Environment variables** (.env files)
3. **GCP Secret Manager** (production only)
4. **Command-line arguments** (override all)

### Configuration Files

```
config/
├── development.yaml    # Dev environment settings
├── staging.yaml        # Staging environment settings
├── production.yaml     # Production settings
└── alembic.ini         # Database migration config
```

### Example: production.yaml

```yaml
# Production configuration
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  reload: false
  debug: false

database:
  pool_size: 20
  max_overflow: 10
  pool_timeout: 30
  pool_recycle: 3600

redis:
  max_connections: 50
  socket_timeout: 5
  socket_connect_timeout: 5

ollama:
  timeout: 30
  max_retries: 3
  num_parallel: 4
  max_loaded_models: 3

monitoring:
  prometheus_enabled: true
  jaeger_enabled: true
  log_level: "INFO"

security:
  api_key_required: true
  rate_limit_requests: 100
  rate_limit_window: 60
  cors_origins:
    - "https://elevatediq.ai"
    - "https://elevatediq.ai/ollama"
```

---

## Secrets Management

### Development (Local)

Use `.env.dev` file (gitignored):

```bash
# Generate secure API key
openssl rand -hex 32

# Store in .env.dev
echo "API_KEY=sk-$(openssl rand -hex 32)" >> .env.dev
```

### Production (GCP Secret Manager)

**Create Secrets**:

```bash
# Authenticate with GCP
gcloud auth login
gcloud config set project gcp-landing-zone

# Create secrets
echo -n "postgresql://user:password@host:5432/ollama" | \
  gcloud secrets create ollama-database-url --data-file=-

echo -n "super-secret-redis-password" | \
  gcloud secrets create ollama-redis-password --data-file=-

echo -n "sk-production-api-key-here" | \
  gcloud secrets create ollama-api-keys --data-file=-
```

**Access Secrets in Application**:

```python
from google.cloud import secretmanager

def get_secret(secret_id):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# Usage
DATABASE_URL = get_secret("ollama-database-url")
```

**Grant Access**:

```bash
# Grant service account access to secrets
gcloud secrets add-iam-policy-binding ollama-database-url \
  --member="serviceAccount:ollama-sa@gcp-landing-zone.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

---

## Database Migrations

### Using Alembic

**Initialize** (first time only):

```bash
# Already done (alembic/ directory exists)
# alembic init alembic
```

**Create Migration**:

```bash
# Auto-generate migration from model changes
alembic revision --autogenerate -m "Add users table"

# Manual migration
alembic revision -m "Add index on conversations.user_id"
```

**Apply Migrations**:

```bash
# Development
docker-compose exec api alembic upgrade head

# Production
docker-compose -f docker/docker-compose.prod.yml exec prod-ollama-api alembic upgrade head
```

**Rollback Migration**:

```bash
# Rollback one version
alembic downgrade -1

# Rollback to specific version
alembic downgrade <revision_id>

# Rollback all
alembic downgrade base
```

**View Migration History**:

```bash
# Current version
alembic current

# Migration history
alembic history --verbose

# Show pending migrations
alembic upgrade head --sql  # Dry run, show SQL only
```

---

## Deployment Procedures

### Local Development

**1. Start Services**:

```bash
# Get real IP (MANDATORY - never use localhost)
export REAL_IP=$(hostname -I | awk '{print $1}')

# Create dev environment file
cp .env.example .env.dev
sed -i "s|PUBLIC_API_URL=.*|PUBLIC_API_URL=http://$REAL_IP:8000|" .env.dev

# Start all services
docker-compose up -d

# Watch logs
docker-compose logs -f api
```

**2. Run Migrations**:

```bash
# Apply database migrations
docker-compose exec api alembic upgrade head
```

**3. Verify Deployment**:

```bash
# Health check (using real IP)
curl http://$REAL_IP:8000/health

# List models
curl -H "Authorization: Bearer dev-key-for-testing-only" \
     http://$REAL_IP:8000/api/v1/models
```

**4. Development Workflow**:

```bash
# Code changes auto-reload (mounted volumes)
# Edit files in ollama/ directory
# FastAPI auto-reloads on file changes

# Run tests
docker-compose exec api pytest tests/ -v

# Type checking
docker-compose exec api mypy ollama/ --strict

# Linting
docker-compose exec api ruff check ollama/
```

**5. Stop Services**:

```bash
# Stop all containers
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v
```

---

### Staging Deployment

**Not yet implemented**. Future staging environment will use:
- GCP Cloud Run or Kubernetes
- Separate staging database
- Staging subdomain: `staging-ollama.elevatediq.ai`

---

### Production Deployment

#### Prerequisites Checklist

- [ ] All tests passing: `pytest tests/ -v`
- [ ] Type checking clean: `mypy ollama/ --strict`
- [ ] Security audit clean: `pip-audit`
- [ ] GCP secrets configured
- [ ] Database backup recent (< 24 hours)
- [ ] Load testing completed
- [ ] Rollback plan documented
- [ ] Monitoring dashboards ready
- [ ] On-call engineer notified

#### Deployment Steps

**1. Pre-Deployment**:

```bash
# Backup database
docker-compose exec postgres pg_dump -U ollama_user ollama > \
  backups/ollama-$(date +%Y%m%d-%H%M%S).sql

# Tag current version
git tag -s v0.1.0 -m "Production release v0.1.0"
git push origin v0.1.0

# Build production image
docker build -t ollama:1.0.0 -f docker/Dockerfile .

# Security scan
docker scan ollama:1.0.0
```

**2. Deploy to Production**:

```bash
# Pull latest code
git pull origin main

# Stop current services (zero-downtime not yet implemented)
docker-compose -f docker/docker-compose.prod.yml down

# Start new version
docker-compose -f docker/docker-compose.prod.yml up -d

# Run migrations
docker-compose -f docker/docker-compose.prod.yml exec prod-ollama-api \
  alembic upgrade head

# Verify health
curl https://elevatediq.ai/ollama/health
```

**3. Post-Deployment Verification**:

```bash
# Health check
curl https://elevatediq.ai/ollama/health

# API functionality
curl -H "Authorization: Bearer <prod-api-key>" \
     https://elevatediq.ai/ollama/api/v1/models

# Monitor logs (first 5 minutes)
docker-compose -f docker/docker-compose.prod.yml logs -f api

# Check metrics
curl http://localhost:9090/metrics | grep ollama_
```

**4. Monitor for Issues**:

```bash
# Watch error rate
watch -n 5 'docker-compose -f docker/docker-compose.prod.yml logs api | grep ERROR'

# Check database connections
docker-compose -f docker/docker-compose.prod.yml exec postgres \
  psql -U ollama_user -d ollama -c "SELECT count(*) FROM pg_stat_activity;"

# Monitor system resources
docker stats
```

---

## Health Checks & Verification

### Health Check Endpoints

**Basic Health**:
```bash
curl https://elevatediq.ai/ollama/health
# Expected: {"status": "healthy", "version": "0.1.0"}
```

**Detailed Health**:
```bash
curl https://elevatediq.ai/ollama/health/detailed
# Expected: Full service status
```

### Verification Checklist

#### API Layer
- [ ] Health endpoint responds (< 100ms)
- [ ] Authentication working (401 for invalid key)
- [ ] Rate limiting active (429 after 100 requests)
- [ ] CORS headers present

#### Database Layer
- [ ] Database connection pool healthy
- [ ] Migrations applied successfully
- [ ] Sample queries execute (< 100ms)

#### Cache Layer
- [ ] Redis connection established
- [ ] Cache set/get operations working
- [ ] TTL expiration functioning

#### Model Layer
- [ ] Ollama service responding
- [ ] Models loaded successfully
- [ ] Sample inference completes

#### Monitoring
- [ ] Prometheus metrics endpoint accessible
- [ ] Grafana dashboards displaying data
- [ ] Alerts configured and firing (test)

---

## Rollback Procedures

### Rollback Decision Criteria

Rollback if:
- Error rate > 5% for 5 minutes
- Critical functionality broken
- Data corruption detected
- Performance degraded > 50%
- Security vulnerability discovered

### Rollback Steps

**1. Immediate Rollback** (< 10 seconds):

```bash
# Stop current version
docker-compose -f docker/docker-compose.prod.yml down

# Start previous version
docker-compose -f docker/docker-compose.prod.yml up -d --scale api=1
```

**2. Database Rollback** (if migrations applied):

```bash
# Rollback to previous migration
docker-compose exec prod-ollama-api alembic downgrade -1

# OR restore from backup
docker-compose exec postgres psql -U ollama_user -d ollama < \
  backups/ollama-20260118-100000.sql
```

**3. Verify Rollback**:

```bash
# Health check
curl https://elevatediq.ai/ollama/health

# Test API
curl -H "Authorization: Bearer <prod-api-key>" \
     https://elevatediq.ai/ollama/api/v1/models

# Monitor logs
docker-compose -f docker/docker-compose.prod.yml logs -f api
```

**4. Post-Rollback**:

```bash
# Notify team
# Update incident report
# Schedule post-mortem
# Plan fix for next deployment
```

---

## Troubleshooting

### Common Issues

#### 1. Database Connection Failures

**Symptoms**:
```
sqlalchemy.exc.OperationalError: could not connect to server
```

**Solutions**:
```bash
# Check database is running
docker-compose ps postgres

# Check connection string
echo $DATABASE_URL

# Test connection manually
docker-compose exec postgres psql -U ollama_user -d ollama

# Restart database
docker-compose restart postgres
```

#### 2. Ollama Service Unavailable

**Symptoms**:
```
httpx.ConnectError: [Errno 111] Connection refused
```

**Solutions**:
```bash
# Check Ollama is running
docker-compose ps ollama

# Check Ollama logs
docker-compose logs ollama

# Restart Ollama
docker-compose restart ollama

# Pull models if missing
docker-compose exec ollama ollama pull llama3.2
```

#### 3. Redis Connection Issues

**Symptoms**:
```
redis.exceptions.ConnectionError: Error connecting to Redis
```

**Solutions**:
```bash
# Check Redis is running
docker-compose ps redis

# Test Redis connection
docker-compose exec redis redis-cli ping
# Expected: PONG

# Check Redis memory
docker-compose exec redis redis-cli info memory

# Restart Redis
docker-compose restart redis
```

#### 4. API Returns 500 Errors

**Symptoms**:
```json
{"error": {"code": "INTERNAL_ERROR", "message": "Unexpected error"}}
```

**Solutions**:
```bash
# Check API logs
docker-compose logs api | tail -100

# Check Python exceptions
docker-compose logs api | grep -i "traceback" -A 20

# Verify environment variables
docker-compose exec api env | grep -E "(DATABASE|REDIS|OLLAMA)"

# Restart API
docker-compose restart api
```

---

## Production Checklist

### Pre-Deployment

- [ ] **Code Quality**
  - [ ] All tests passing
  - [ ] Type checking clean (mypy)
  - [ ] Linting clean (ruff)
  - [ ] Security audit clean (pip-audit)
  - [ ] Code review completed
  - [ ] GPG signed commits

- [ ] **Configuration**
  - [ ] Environment variables verified
  - [ ] Secrets rotated (if needed)
  - [ ] GCP secrets accessible
  - [ ] CORS origins configured
  - [ ] Rate limits set correctly

- [ ] **Database**
  - [ ] Migrations tested in staging
  - [ ] Backup completed (< 24h old)
  - [ ] Rollback procedure tested
  - [ ] Connection pool sized appropriately

- [ ] **Infrastructure**
  - [ ] GCP Load Balancer configured
  - [ ] Cloud Armor rules active
  - [ ] Firewall rules verified
  - [ ] SSL certificates valid (> 30 days)
  - [ ] DNS records correct

- [ ] **Monitoring**
  - [ ] Dashboards configured
  - [ ] Alerts active
  - [ ] Log aggregation working
  - [ ] On-call rotation updated

### Post-Deployment

- [ ] **Verification**
  - [ ] Health check passing
  - [ ] API endpoints functional
  - [ ] Models loading successfully
  - [ ] Authentication working
  - [ ] Rate limiting active

- [ ] **Monitoring**
  - [ ] No error spikes
  - [ ] Latency within SLA
  - [ ] Resource usage normal
  - [ ] No alerts firing

- [ ] **Documentation**
  - [ ] Deployment notes recorded
  - [ ] CHANGELOG.md updated
  - [ ] Team notified
  - [ ] Customer communication sent (if applicable)

---

## Support

- **On-Call Engineer**: Check PagerDuty rotation
- **Team Slack**: #ai-infrastructure
- **Email**: ai-infrastructure@elevatediq.ai
- **Documentation**: https://github.com/kushin77/ollama/docs

---

**Last Updated**: January 18, 2026
**Version**: 0.1.0
**Next Review**: February 18, 2026
