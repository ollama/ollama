# Deployment Guide

## Table of Contents
- [Local Development](#local-development)
- [Docker Compose](#docker-compose)
- [On-Prem Deployment Model](#on-prem-deployment-model)
- [Production Deployment](#production-deployment)
- [Environment Variables](#environment-variables)
- [Database Migrations](#database-migrations)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## Local Development

### Prerequisites
- Python 3.12+
- PostgreSQL 15+
- Redis 7+
- Qdrant (latest)
- Ollama (latest)

### Setup
```bash
# Clone repository
git clone <repository-url>
cd ollama

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env

# Edit .env with your settings
nano .env

# Initialize database
alembic upgrade head

# Run tests
pytest tests/ -v --cov=ollama

# Start development server
uvicorn ollama.main:app --reload --host 0.0.0.0 --port 8000
```

## Docker Compose

### Development Environment

Start all services with hot-reload:

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down

# Reset volumes (clean start)
docker-compose down -v
```

**Services Available:**
- API Gateway: http://localhost:8000
- API Docs: http://localhost:8000/docs
- PostgreSQL: localhost:5432
- Redis: localhost:6379
- Qdrant: http://localhost:6333
- Ollama: http://localhost:11434
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
- Jaeger UI: http://localhost:16686

### Production Environment

```bash
# Create production .env file
cp .env.example .env.prod
nano .env.prod

# Start production stack
docker-compose -f docker/docker-compose.prod.yml up -d

# Check health
curl http://localhost:8000/health

# View logs
docker-compose -f docker/docker-compose.prod.yml logs -f

# Scale API workers
docker-compose -f docker/docker-compose.prod.yml up -d --scale api=4
```

## On-Prem Execution Index

Use [On-Prem Execution Index](operations/ON_PREM_EXECUTION_INDEX.md) as the shared entry point for target-server-local workflows. The underlying SSOT remains [On-Prem Deployment Model](operations/ON_PREM_DEPLOYMENT_MODEL.md).

Keep host-specific values in the checked-in inventories and keep runtime state in declared volumes or other IaC-managed storage.

## Production Deployment

The cloud-oriented examples below are retained for legacy reference. For on-prem bare-metal and development-node runs, prefer the target-server-local flow above.

### AWS Deployment

#### 1. EC2 Instance Setup
```bash
# Launch EC2 instance (Ubuntu 22.04 LTS)
# Instance type: t3.xlarge or better (for GPU: p3.2xlarge)
# Storage: 100GB+ SSD

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# For GPU support
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

#### 2. Deploy Application
```bash
# Clone repository
git clone <repository-url>
cd ollama

# Setup environment
cp .env.example .env.prod
nano .env.prod  # Configure production settings

# Build and start
docker-compose -f docker/docker-compose.prod.yml up -d --build

# Check logs
docker-compose -f docker/docker-compose.prod.yml logs -f api
```

#### 3. Setup SSL with Let's Encrypt
```bash
# Install certbot
sudo apt-get update
sudo apt-get install -y certbot

# Generate certificates
sudo certbot certonly --standalone -d api.yourdomain.com

# Configure nginx
sudo cp nginx/nginx.conf.example nginx/nginx.conf
sudo nano nginx/nginx.conf  # Update domain and SSL paths

# Restart services
docker-compose -f docker/docker-compose.prod.yml restart nginx
```

### GCP Deployment

#### 1. Create GCE Instance
```bash
# Create instance with Container-Optimized OS
gcloud compute instances create-with-container ollama-api \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --boot-disk-size=100GB \
  --container-image=gcr.io/<project-id>/ollama-api:latest \
  --container-env-file=.env.prod

# For GPU support
gcloud compute instances create ollama-api-gpu \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --boot-disk-size=100GB \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE
```

#### 2. Setup Load Balancer
```bash
# Create instance group
gcloud compute instance-groups unmanaged create ollama-api-group \
  --zone=us-central1-a

gcloud compute instance-groups unmanaged add-instances ollama-api-group \
  --instances=ollama-api \
  --zone=us-central1-a

# Create health check
gcloud compute health-checks create http ollama-health \
  --port=8000 \
  --request-path=/health

# Create backend service
gcloud compute backend-services create ollama-backend \
  --protocol=HTTP \
  --health-checks=ollama-health \
  --global

# Add instance group to backend
gcloud compute backend-services add-backend ollama-backend \
  --instance-group=ollama-api-group \
  --instance-group-zone=us-central1-a \
  --global

# Create URL map and proxy
gcloud compute url-maps create ollama-lb \
  --default-service=ollama-backend

gcloud compute target-http-proxies create ollama-proxy \
  --url-map=ollama-lb

# Create forwarding rule
gcloud compute forwarding-rules create ollama-forwarding-rule \
  --global \
  --target-http-proxy=ollama-proxy \
  --ports=80
```

## Environment Variables

### Required Variables
```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:password@host:5432/dbname

# Redis
REDIS_URL=redis://host:6379/0

# Qdrant
QDRANT_URL=http://host:6333
QDRANT_API_KEY=your-api-key  # Production only

# Ollama
OLLAMA_URL=http://host:11434

# Security
SECRET_KEY=your-secret-key-min-32-chars
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### Optional Variables
```bash
# Environment
ENVIRONMENT=production  # development, staging, production
DEBUG=false

# Workers (Production)
WORKERS=4

# Monitoring
JAEGER_AGENT_HOST=jaeger
JAEGER_AGENT_PORT=6831

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=60
```

## Database Migrations

### Using Alembic
```bash
# Create new migration
alembic revision --autogenerate -m "Description of changes"

# Apply migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1

# View migration history
alembic history

# Rollback to specific revision
alembic downgrade <revision_id>
```

### Manual Migrations
```bash
# Connect to database
docker-compose exec postgres psql -U ollama -d ollama

# Run SQL commands
\dt  # List tables
\d+ users  # Describe users table
```

## Monitoring

### Prometheus Metrics
- **Endpoint**: http://localhost:9090
- **Targets**: API, PostgreSQL, Redis
- **Retention**: 30 days

### Grafana Dashboards
- **URL**: http://localhost:3000
- **Default credentials**: admin/admin
- **Dashboards**: Pre-configured for API, Database, Cache

### Jaeger Tracing
- **UI**: http://localhost:16686
- **Trace retention**: 24 hours
- **Sampling**: 100% in development, 10% in production

### Health Checks
```bash
# API Health
curl http://localhost:8000/health

# Database Health
docker-compose exec postgres pg_isready -U ollama

# Redis Health
docker-compose exec redis redis-cli ping

# Qdrant Health
curl http://localhost:6333/health

# Ollama Health
curl http://localhost:11434/api/version
```

## Troubleshooting

### Common Issues

#### 1. Database Connection Failed
```bash
# Check if PostgreSQL is running
docker-compose ps postgres

# View logs
docker-compose logs postgres

# Test connection
docker-compose exec postgres psql -U ollama -d ollama -c "SELECT 1;"

# Reset database
docker-compose down -v
docker-compose up -d postgres
alembic upgrade head
```

#### 2. Redis Connection Failed
```bash
# Check if Redis is running
docker-compose ps redis

# Test connection
docker-compose exec redis redis-cli ping

# View keys
docker-compose exec redis redis-cli keys "*"
```

#### 3. API Not Starting
```bash
# View detailed logs
docker-compose logs -f api

# Check environment variables
docker-compose exec api env | grep -E "(DATABASE|REDIS|QDRANT|OLLAMA)_URL"

# Restart service
docker-compose restart api

# Rebuild image
docker-compose build --no-cache api
docker-compose up -d api
```

#### 4. High Memory Usage
```bash
# Check container stats
docker stats

# Limit memory (docker-compose.yml)
services:
  api:
    deploy:
      resources:
        limits:
          memory: 2G
```

#### 5. Slow API Response
```bash
# Check Jaeger traces
# Open http://localhost:16686

# Check database connections
docker-compose exec postgres psql -U ollama -d ollama -c "SELECT count(*) FROM pg_stat_activity;"

# Monitor Redis operations
docker-compose exec redis redis-cli monitor

# Check API metrics
curl http://localhost:8000/metrics
```

### Debug Mode

Enable debug logging:
```bash
# Set in .env
DEBUG=true
LOG_LEVEL=DEBUG

# Restart services
docker-compose restart api
```

### Performance Tuning

#### PostgreSQL
```sql
-- Increase connection pool
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';

-- Apply changes
SELECT pg_reload_conf();
```

#### Redis
```bash
# Edit redis.conf
maxmemory 512mb
maxmemory-policy allkeys-lru
```

#### API Workers
```yaml
# docker-compose.prod.yml
environment:
  WORKERS: 8  # Increase based on CPU cores
```

## Backup and Recovery

### Database Backup
```bash
# Manual backup
docker-compose exec postgres pg_dump -U ollama ollama > backup_$(date +%Y%m%d).sql

# Restore
docker-compose exec -T postgres psql -U ollama ollama < backup_20240101.sql
```

### Automated Backups
```bash
# Add to crontab
0 2 * * * cd /path/to/ollama && docker-compose exec postgres pg_dump -U ollama ollama | gzip > /backups/ollama_$(date +\%Y\%m\%d).sql.gz
```

## Security Checklist

- [ ] Change default passwords in `.env.prod`
- [ ] Use strong SECRET_KEY (32+ characters)
- [ ] Enable HTTPS/SSL certificates
- [ ] Configure firewall rules
- [ ] Enable API key authentication
- [ ] Set up rate limiting
- [ ] Configure CORS origins
- [ ] Enable audit logging
- [ ] Regular security updates
- [ ] Backup encryption

## Support

For issues and questions:
- Documentation: `/docs` in this repository
- Issues: GitHub Issues
- Email: admin@elevatediq.ai
