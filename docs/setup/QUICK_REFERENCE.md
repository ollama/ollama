# 🚀 Quick Reference - Ollama Commands

## Docker Management

### Start Services
```bash
cd /home/akushnir/ollama
docker-compose up -d
```

### Stop Services
```bash
docker-compose down
```

### View Status
```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

### View Logs
```bash
docker logs ollama-api        # FastAPI server
docker logs ollama-postgres   # Database
docker logs ollama-redis      # Cache
docker logs ollama-qdrant     # Vector DB
```

---

## FastAPI Server

### Start Development Server
```bash
cd /home/akushnir/ollama
source venv/bin/activate
uvicorn ollama.main:app --reload --host 0.0.0.0 --port 8000
```

### Start with Background Process
```bash
cd /home/akushnir/ollama && source venv/bin/activate && \
  nohup uvicorn ollama.main:app --reload --host 0.0.0.0 --port 8000 \
  > /tmp/ollama.log 2>&1 &
```

### Stop Server
```bash
pkill -f uvicorn
```

### View Server Logs
```bash
tail -100 /tmp/ollama.log
```

---

## Testing

### Health Check (No Auth)
```bash
curl http://127.0.0.1:8000/health
```

### Liveness Probe (No Auth)
```bash
curl http://127.0.0.1:8000/health/live
```

### Protected Health Check (OAuth Required)
```bash
curl -H "Authorization: Bearer <token>" \
  http://127.0.0.1:8000/api/v1/health
```

### Run Tests
```bash
cd /home/akushnir/ollama
source venv/bin/activate
pytest tests/ -v --cov=ollama
```

### Type Checking
```bash
mypy ollama/ --strict
```

### Linting
```bash
ruff check ollama/
```

### Security Audit
```bash
pip-audit
```

### Run All Checks
```bash
pytest tests/ -v --cov=ollama && \
mypy ollama/ --strict && \
ruff check ollama/ && \
pip-audit
```

---

## Monitoring & Observability

### Prometheus
**URL**: http://127.0.0.1:9090
**Query**: View metrics, set up alerts

### Grafana
**URL**: http://127.0.0.1:3300
**User**: admin
**Pass**: stronglocaldevpassword

### Jaeger
**URL**: http://127.0.0.1:16686
**Feature**: Trace API requests across services

---

## Configuration

### View Configuration
```bash
cat /home/akushnir/ollama/.env | grep -E "^[A-Z_]+" | head -20
```

### Update Configuration
```bash
# Edit .env
nano /home/akushnir/ollama/.env

# Apply changes
source /home/akushnir/ollama/.env
```

### Common Settings
```bash
# Public endpoint
PUBLIC_API_ENDPOINT=https://elevatediq.ai/ollama

# Database
DATABASE_URL=postgresql+asyncpg://ollama:password@127.0.0.1:5432/ollama

# Cache
REDIS_URL=redis://127.0.0.1:6379/0

# Vector DB
QDRANT_HOST=127.0.0.1
QDRANT_PORT=6333

# OAuth
FIREBASE_ENABLED=true
FIREBASE_CREDENTIALS_PATH=/secrets/firebase-service-account.json
```

---

## Database Management

### Connect to PostgreSQL
```bash
psql postgresql://ollama:password@127.0.0.1:5432/ollama
```

### Run Migrations
```bash
cd /home/akushnir/ollama
source venv/bin/activate
alembic upgrade head
```

### View Migration Status
```bash
alembic current
alembic history
```

### Create New Migration
```bash
alembic revision -m "description_of_change"
# Edit the generated file in alembic/versions/
alembic upgrade head
```

---

## Deployment

### Build Docker Image
```bash
cd /home/akushnir/ollama
docker build -t ollama:1.0.0 -f docker/Dockerfile .
```

### Push to Registry
```bash
docker tag ollama:1.0.0 gcr.io/your-project/ollama:1.0.0
docker push gcr.io/your-project/ollama:1.0.0
```

### Deploy to GCP Cloud Run
```bash
gcloud run deploy ollama-api \
  --image gcr.io/your-project/ollama:1.0.0 \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Deploy to GKE
```bash
kubectl apply -f k8s/base/
```

---

## Troubleshooting

### Check if Port is in Use
```bash
lsof -i :8000
```

### Free Port
```bash
fuser -k 8000/tcp
```

### View Python Imports
```bash
cd /home/akushnir/ollama && source venv/bin/activate && \
  python -c "from ollama.auth import init_firebase; print('✅ Imports OK')"
```

### Test Database Connection
```bash
psql postgresql://ollama:password@127.0.0.1:5432/ollama -c "SELECT 1"
```

### Test Redis Connection
```bash
redis-cli -h 127.0.0.1 -p 6379 ping
```

### Test Qdrant Connection
```bash
curl http://127.0.0.1:6333/health
```

### Check Docker Network
```bash
docker network inspect ollama_default
```

---

## Git & Version Control

### Check Status
```bash
cd /home/akushnir/ollama
git status
```

### View Recent Commits
```bash
git log --oneline -10
```

### Create Feature Branch
```bash
git checkout -b feature/my-feature
```

### Commit Changes
```bash
git add .
git commit -S -m "feat(scope): description"
```

### Push Changes
```bash
git push origin feature/my-feature
```

### Create Pull Request
```bash
# Push branch, then create PR via GitHub UI
```

---

## Quick Diagnostics

### Full System Check
```bash
echo "🔍 OLLAMA SYSTEM CHECK"
echo "====================="
echo ""
echo "📦 Docker Services:"
docker ps --format "table {{.Names}}\t{{.Status}}" | grep ollama
echo ""
echo "🌐 Port Status:"
netstat -tln | grep -E ":8000|:5432|:6379|:6333|:3300" || echo "Ports not in use"
echo ""
echo "🐍 Python Packages:"
source venv/bin/activate && python -c "import fastapi, sqlalchemy, pydantic; print('✅ Core packages OK')"
echo ""
echo "📄 Configuration:"
grep -E "FIREBASE_ENABLED|QDRANT_HOST" /home/akushnir/ollama/.env
echo ""
echo "✅ System Check Complete"
```

---

## Common Issues & Solutions

### Issue: Port 8000 Already in Use
```bash
# Find process
lsof -i :8000
# Kill process
pkill -f uvicorn
# Or specify different port
uvicorn ollama.main:app --port 8001
```

### Issue: Database Connection Failed
```bash
# Check if PostgreSQL is running
docker logs ollama-postgres
# Verify credentials in .env
cat /home/akushnir/ollama/.env | grep DATABASE_URL
# Test connection
psql postgresql://ollama:password@127.0.0.1:5432/ollama -c "SELECT 1"
```

### Issue: Firebase Not Initialized
```bash
# Make sure credentials file exists
ls -la /home/akushnir/ollama/secrets/firebase-service-account.json
# Check if Firebase is enabled
grep FIREBASE_ENABLED /home/akushnir/ollama/.env
# Enable if not
echo "FIREBASE_ENABLED=true" >> /home/akushnir/ollama/.env
```

### Issue: Type Check Failures
```bash
# Run mypy with details
mypy ollama/ --strict --show-error-codes --show-error-context
# Fix specific file
mypy ollama/api/routes/health.py --strict
```

---

## Performance Tuning

### Increase Worker Count
```bash
# In docker-compose.yml or deployment
uvicorn ollama.main:app --workers 8 --host 0.0.0.0 --port 8000
```

### Enable Query Logging
```bash
# In .env
SQLALCHEMY_ECHO=true
```

### Profile Code
```bash
cd /home/akushnir/ollama && source venv/bin/activate
python -m cProfile -s cumtime -o /tmp/profile.prof main.py
# Analyze
python -m pstats /tmp/profile.prof
```

---

## Monitoring Commands

### View Metrics
```bash
# Raw Prometheus metrics
curl http://127.0.0.1:9090/api/v1/query?query=request_count_total

# Formatted JSON
curl -s http://127.0.0.1:9090/api/v1/query?query=request_count_total | \
  python -m json.tool
```

### Watch Logs
```bash
# Follow logs in real-time
tail -f /tmp/ollama.log

# Watch specific service
docker logs -f ollama-api

# Watch with grep
docker logs -f ollama-api | grep ERROR
```

### System Resources
```bash
# Docker container stats
docker stats ollama-api ollama-postgres ollama-redis

# Host system
top
free -h
df -h
```

---

## Development Workflow

### Daily Startup
```bash
cd /home/akushnir/ollama
docker-compose up -d
source venv/bin/activate
uvicorn ollama.main:app --reload --host 0.0.0.0 --port 8000 &
echo "✅ Ollama development environment ready"
```

### Before Commit
```bash
cd /home/akushnir/ollama
source venv/bin/activate

# Format code
black ollama/ tests/ --line-length=100

# Run checks
pytest tests/ -v
mypy ollama/ --strict
ruff check ollama/
pip-audit

# Commit
git add .
git commit -S -m "feat(scope): description"
git push origin feature/branch-name
```

### Code Review
```bash
# View changes
git diff main...feature/branch-name

# Check specific file
git diff main...feature/branch-name -- ollama/api/routes/health.py
```

---

## Documentation

- Full deployment: [DEPLOYMENT_STATUS.md](DEPLOYMENT_STATUS.md)
- OAuth setup: [docs/OAUTH_SETUP.md](docs/OAUTH_SETUP.md)
- Architecture: [docs/architecture.md](docs/architecture.md)
- API reference: [PUBLIC_API.md](PUBLIC_API.md)
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)

---

**Last Updated**: January 13, 2026
**Environment**: Development/Production Ready
