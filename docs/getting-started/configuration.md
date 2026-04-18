# Configuration Guide

Customize Ollama for your environment.

## Environment Variables

Create `.env` file (copy from `.env.example`):

```bash
# API Configuration
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=info
FASTAPI_WORKERS=4
FASTAPI_PORT=8000

# Security
API_KEY_PREFIX=sk
REQUIRE_API_KEY=true
CORS_ORIGINS=https://elevatediq.ai

# Database
DATABASE_URL=postgresql://user:pass@postgres:5432/ollama
DATABASE_POOL_SIZE=20

# Cache
REDIS_URL=redis://:password@redis:6379/0
CACHE_TTL=3600

# Inference
OLLAMA_BASE_URL=http://ollama:11434
MODEL_CACHE_SIZE=5000000000
MAX_BATCH_SIZE=32

# Monitoring
PROMETHEUS_PORT=9090
JAEGER_ENABLED=true
JAEGER_AGENT_HOST=localhost
JAEGER_AGENT_PORT=6831
```

## Configuration Files

### docker-compose.yml

Override in `docker-compose.override.yml` for local development:

```yaml
version: "3.9"

services:
  api:
    environment:
      DEBUG: "true"
      LOG_LEVEL: "debug"
    ports:
      - "8001:8000" # Custom port
    volumes:
      - ./ollama:/app/ollama:delegated

  postgres:
    environment:
      POSTGRES_PASSWORD: dev_password
```

### config/development.yaml

Development-specific settings:

```yaml
api:
  debug: true
  log_level: debug
  workers: 1

database:
  echo: true # Log SQL
  pool_size: 5

cache:
  ttl: 600

inference:
  timeout: 30
```

## Advanced Configuration

### Model Management

Configure available models in `config/models.yaml`:

```yaml
models:
  - name: llama2
    display_name: Llama 2
    size: 7B
    parameters: 7000000000
    context_length: 4096
    quantization: q4_K_M

  - name: llama2-13b
    display_name: Llama 2 13B
    size: 13B
    parameters: 13000000000
    context_length: 4096
    quantization: q4_0
```

### Resource Limits

In `docker-compose.yml`:

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: "4"
          memory: 8G
        reservations:
          cpus: "2"
          memory: 4G
```

### Database Tuning

Optimize PostgreSQL in `docker/postgres/postgresql.conf`:

```
max_connections = 200
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 10MB
```

## Production Checklist

- [ ] `DEBUG=false` in production
- [ ] `REQUIRE_API_KEY=true`
- [ ] CORS origins restricted to your domain
- [ ] Database credentials in GCP Secret Manager
- [ ] TLS certificates configured
- [ ] Health check endpoint monitoring
- [ ] Alerting configured for critical metrics
- [ ] Logging and tracing enabled
- [ ] Regular backups configured
- [ ] Monitoring dashboards set up
