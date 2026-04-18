# Installation Guide

Detailed instructions for installing Ollama.

## System Requirements

| Component      | Requirement                   |
| -------------- | ----------------------------- |
| OS             | Linux, macOS, Windows (WSL2)  |
| Docker         | 24.0+                         |
| Docker Compose | 2.20+                         |
| RAM            | 4GB minimum, 8GB+ recommended |
| Disk           | 10GB minimum (models, cache)  |
| Python         | 3.11+ (development only)      |
| Git            | Latest stable                 |

## Installation

### Option 1: Docker Compose (Recommended)

Fastest way to get started.

```bash
# 1. Clone repository
git clone https://github.com/kushin77/ollama.git
cd ollama

# 2. Start services
docker-compose up -d

# 3. Verify
curl http://localhost:8000/api/v1/health
```

### Option 2: Manual Installation

For development or custom setups.

```bash
# 1. Install Python dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements/base.txt

# 2. Install system dependencies
# macOS
brew install postgresql redis ollama

# Ubuntu/Debian
sudo apt-get install postgresql redis-server

# 3. Start services
postgres -D /usr/local/var/postgres &
redis-server &
ollama serve &

# 4. Run API server
uvicorn ollama.main:app --reload --port 8000
```

### Option 3: Kubernetes

For production deployments.

```bash
# 1. Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# 2. Deploy
helm install ollama ./k8s/helm/ollama \
  --namespace ollama \
  --create-namespace \
  -f values-prod.yaml

# 3. Verify
kubectl get pods -n ollama
```

## Verification

After installation, verify all services:

```bash
# Health check
curl http://localhost:8000/api/v1/health

# List models
curl http://localhost:8000/api/v1/models

# Generate text
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"llama2","prompt":"Hello"}'
```

## Troubleshooting Installation

### Docker not running

```bash
# Start Docker service
sudo systemctl start docker  # Linux
open -a Docker              # macOS
```

### Port conflicts

Edit `docker-compose.yml` to use different ports:

```yaml
services:
  api:
    ports:
      - "8001:8000" # Use 8001 instead of 8000
```

### Low disk space

Install smaller model:

```bash
ollama pull phi  # 3B parameters
# or
ollama pull neural-chat:7b  # 7B parameters
```

### Python version mismatch

Ensure Python 3.11+:

```bash
python3 --version  # Should be 3.11+
python3 -m venv venv
```

## Next Steps

- [Configuration Guide](configuration.md) - Customize settings
- [Quickstart](quickstart.md) - Get started with first request
- [API Reference](../api/endpoints.md) - Learn the API
