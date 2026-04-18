# 🚀 Quick Reference Guide

## One-Line Setup

```bash
git clone https://github.com/kushin77/ollama.git && cd ollama && ./scripts/bootstrap.sh
```

## Development Commands

### Local Development

```bash
# Activate environment
source venv/bin/activate

# Start services
docker-compose -f docker/docker-compose.local.yml up -d

# Run API server
python -m ollama.server

# API available at: http://localhost:8000
```

### Code Quality

```bash
# Format code
black ollama/ tests/
isort ollama/ tests/

# Check for issues
ruff check ollama/ tests/
mypy ollama/ --strict

# Run tests
pytest tests/ -v --cov=ollama

# Security scanning
bandit -r ollama/
pip-audit
```

### Production Deployment

```bash
# Start full stack
docker-compose -f docker/docker-compose.prod.yml up -d

# Check health
curl http://localhost:8000/health
curl http://localhost:9090/api/v1/query?query=up

# View logs
docker-compose -f docker/docker-compose.prod.yml logs -f ollama-api
```

## 🏗 Infrastructure (GCP Landing Zone)

For infrastructure management and onboarding, refer to [Infrastructure Onboarding Guide](ONBOARDING_INFRA.md).

### Core Tools

- **Terraform** (v1.7+)
- **gcloud CLI**
- **kubectl**

### Critical Commands

```bash
# Authenticate
gcloud auth login
gcloud config set project gcp-eiq

# Terraform basic workflow
terraform init
terraform plan
terraform apply
```

## Key Files

| File                      | Purpose                      |
| ------------------------- | ---------------------------- |
| `.copilot-instructions`   | Elite development guidelines |
| `README.md`               | Complete documentation       |
| `CONTRIBUTING.md`         | Contribution workflow        |
| `docker/docker-compose.local.yml` | Local development stack      |
| `docker/docker-compose.prod.yml`  | Production stack             |
| `config/development.yaml` | Dev configuration            |
| `config/production.yaml`  | Prod configuration           |
| `.env.example`            | Environment template         |
| `scripts/bootstrap.sh`    | Automated setup              |

## API Examples

### REST API

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/api/models

# Generate text
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "prompt": "Explain local AI",
    "stream": false
  }'

# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "messages": [
      {"role": "user", "content": "Hello"}
    ]
  }'
```

### Python Client

```python
from ollama import Client

client = Client()

# Generate
response = client.generate(model="llama2", prompt="Hello")

# Chat
response = client.chat(
    model="llama2",
    messages=[{"role": "user", "content": "Hello"}]
)

# Embeddings
response = client.embeddings(
    model="embedding-model",
    input_text="Text to embed"
)

# List models
models = client.list_models()

# Health
health = client.health()
```

## Git Workflow

### Feature Development

```bash
# Create branch
git checkout -b feature/my-feature

# Make changes, commit atomically
git commit -S -m "feat(scope): description"

# Push and create PR
git push origin feature/my-feature
```

### Commit Message Format

```
type(scope): subject

Optional body with details

Fixes #123
```

**Types**: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `infra`, `chore`

## Environment Variables

Key variables in `.env`:

```bash
# Server
OLLAMA_HOST=0.0.0.0:8000
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql://ollama:password@postgres:5432/ollama

# Cache
REDIS_URL=redis://redis:6379/0

# Security
API_KEY_AUTH_ENABLED=true

# GPU
CUDA_VISIBLE_DEVICES=0
```

## Monitoring

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000
- **Jaeger**: http://localhost:16686

Query metrics:

```bash
# Request rate
curl 'http://localhost:9090/api/v1/query?query=rate(ollama_requests_total[5m])'

# GPU memory
curl 'http://localhost:9090/api/v1/query?query=ollama_gpu_memory_used_bytes'

# Inference latency p99
curl 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.99,ollama_inference_duration_seconds)'
```

## Deterministic Rerun Checks

Use [On-Prem Execution Index](operations/ON_PREM_EXECUTION_INDEX.md) for the matrix command and evidence layout. Run `./scripts/host-profile-matrix.sh` to capture first-pass and second-pass artifacts.

The runner writes evidence to an ephemeral directory under `/tmp` by default. Use `--no-open-issue` if you only want local evidence capture.

## Testing

```bash
# Run all tests
pytest

# Specific test
pytest tests/unit/test_client.py -v

# With coverage
pytest --cov=ollama --cov-report=html

# Watch mode (requires pytest-watch)
ptw

# Run specific test
pytest tests/unit/test_client.py::test_client_initialization
```

## Troubleshooting

### GPU Not Detected

```bash
nvidia-smi  # Check CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Connection Issues

```bash
docker-compose ps  # Check services
curl -i http://localhost:8000/health
```

### Database Issues

```bash
docker-compose logs postgres
psql -U ollama -d ollama -h localhost
```

## Performance Profiling

```bash
# Generate request load
python scripts/benchmark.py --duration 60

# Profile CPU
python -m cProfile -s cumulative ollama/server.py

# Memory profile
python -m memory_profiler ollama/inference/engine.py

# GPU monitoring
nvidia-smi dmon -s puc
```

## Docker Commands

```bash
# Build images
docker build -t ollama:latest -f docker/Dockerfile .

# Run container
docker run -it --gpus all -p 8000:8000 ollama:latest

# View logs
docker logs -f ollama-api

# Execute command
docker exec ollama-api ollama list

# Clean up
docker-compose down -v  # Remove volumes too
```

## Documentation

- **API**: See [README.md](README.md#api-reference)
- **Architecture**: See [docs/architecture.md](docs/architecture.md)
- **Development**: See [CONTRIBUTING.md](CONTRIBUTING.md)
- **Project Structure**: See [docs/structure/README.md](docs/structure/README.md)
- **Monitoring**: See [docs/monitoring.md](docs/monitoring.md)

## Support

- 📖 Full docs: [README.md](README.md)
- 💡 Development: [CONTRIBUTING.md](CONTRIBUTING.md)
- 🏗️ Architecture: [docs/architecture.md](docs/architecture.md)
- 🔍 Guidelines: [.copilot-instructions](.copilot-instructions)
- 📊 Summary: [DEVELOPMENT_SUMMARY.md](DEVELOPMENT_SUMMARY.md)

---

**Version**: 1.0.0
**Repository**: https://github.com/kushin77/ollama
**Status**: Production Ready
