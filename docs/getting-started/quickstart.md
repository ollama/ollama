# Quickstart Guide

Get Ollama running locally in 5 minutes.

## Prerequisites

- Docker 24+ and Docker Compose 2.20+
- Python 3.11+ (for development)
- 4GB RAM minimum, 8GB+ recommended
- 10GB disk space for models

## 1. Clone Repository

```bash
git clone https://github.com/kushin77/ollama.git
cd ollama
```

## 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

## 3. Start Services

```bash
docker-compose up -d
```

This starts:

- FastAPI server (port 8000)
- PostgreSQL database
- Redis cache
- Ollama inference engine (port 11434)

## 4. Verify Health

```bash
curl http://localhost:8000/api/v1/health

# Expected response:
# {"status": "healthy"}
```

## 5. Generate Text

```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "prompt": "Hello, how are you?",
    "stream": false
  }'
```

## Next Steps

- [Installation Guide](installation.md) - Detailed setup
- [Configuration](configuration.md) - Customize settings
- [API Reference](../api/endpoints.md) - Full API documentation
- [Deployment Guide](../deployment/gcp-deployment.md) - Deploy to production

## Troubleshooting

### Port Already in Use

```bash
docker-compose down
# Change ports in docker-compose.yml
docker-compose up -d
```

### Out of Memory

Increase Docker memory allocation in Docker Desktop settings.

### Model Not Found

```bash
ollama pull llama2
```

See [Troubleshooting Guide](../operations/troubleshooting.md) for more.
