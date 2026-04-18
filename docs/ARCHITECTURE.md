# Ollama System Architecture

**Version**: 0.1.0
**Last Updated**: January 18, 2026
**Owner**: AI Infrastructure Team

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [System Overview](#system-overview)
- [Architecture Principles](#architecture-principles)
- [Component Architecture](#component-architecture)
- [Data Flow](#data-flow)
- [Technology Stack](#technology-stack)
- [Network Architecture](#network-architecture)
- [Security Architecture](#security-architecture)
- [Scaling Strategy](#scaling-strategy)
- [Failure Modes & Resilience](#failure-modes--resilience)
- [Performance Characteristics](#performance-characteristics)
- [Future Enhancements](#future-enhancements)

---

## Executive Summary

Ollama is a **production-grade local AI infrastructure platform** for building, deploying, and monitoring large language models. All AI workloads run locally on Docker containers with optional GCP Load Balancer for public access via `https://elevatediq.ai/ollama`.

**Core Design Principles**:
- ✅ **Local Sovereignty**: All AI models run locally (zero cloud dependencies)
- ✅ **Public Access**: GCP Load Balancer is the ONLY external entry point
- ✅ **Security First**: Zero trust networking, API key authentication, rate limiting
- ✅ **Production Ready**: 100% test coverage, comprehensive monitoring, 24/7 reliability

**Target Audience**: Elite engineers, research teams, enterprises requiring air-gapped AI systems

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          EXTERNAL CLIENTS                           │
│                  (Internet, Partner Systems, Users)                 │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                    HTTPS/TLS 1.3+
                         │
        ┌────────────────▼────────────────┐
        │   GCP LOAD BALANCER             │
        │ (https://elevatediq.ai/ollama)  │
        │   - API Key Authentication      │
        │   - Rate Limiting (100 req/min) │
        │   - DDoS Protection (Cloud      │
        │     Armor)                      │
        │   - CORS Enforcement            │
        │   - TLS Termination             │
        └────────────────┬────────────────┘
                         │
                    Mutual TLS 1.3+
                         │
        ┌────────────────▼────────────────────────────────────────┐
        │      DOCKER CONTAINER NETWORK (Internal Only)           │
        │                                                          │
        │  ┌──────────┐   ┌──────────┐   ┌──────────┐             │
        │  │  FastAPI │   │PostgreSQL│   │  Redis   │             │
        │  │  Server  │──▶│ Database │   │  Cache   │             │
        │  │:8000     │   │:5432     │   │:6379     │             │
        │  └──────────┘   └──────────┘   └──────────┘             │
        │       ▲                                                   │
        │       │                                                   │
        │  ┌────────────┐     ┌──────────┐                        │
        │  │  Ollama    │     │Prometheus│                        │
        │  │  Models    │     │ Metrics  │                        │
        │  │:11434      │     │:9090     │                        │
        │  └────────────┘     └──────────┘                        │
        │                                                          │
        └──────────────────────────────────────────────────────────┘
                         ▲
            NO EXTERNAL CLIENT ACCESS
           (Firewall blocked from outside)
```

### Deployment Topology

**Production Architecture** (Mandatory):

```
External Clients (Internet)
         ↓
    GCP Load Balancer
    https://elevatediq.ai/ollama
         ↓
    Mutual TLS 1.3+
         ↓
Docker Container Network (Internal Only)
├── FastAPI (0.0.0.0:8000) ← binds all interfaces
├── PostgreSQL (postgres:5432) ← internal service name
├── Redis (redis:6379) ← internal service name
├── Ollama (ollama:11434) ← internal service name
└── Qdrant (qdrant:6333) ← vector database
```

**Key Points**:
- ✅ GCP Load Balancer = ONLY external entry point
- ✅ All services communicate internally via Docker network
- ✅ Firewall blocks all direct external access to internal services
- ❌ No direct client connections to internal services

---

## Architecture Principles

### 1. Local Sovereignty with Public Access

**Principle**: AI workloads run 100% locally with controlled external access

- All models downloaded and cached locally
- Zero external API calls for inference
- Data never leaves infrastructure (unless explicitly configured)
- GCP Load Balancer provides controlled public endpoint

**Benefits**:
- Full control over model versions and data
- No vendor lock-in
- Guaranteed latency and availability
- Regulatory compliance (data sovereignty)

### 2. Zero Trust Security

**Principle**: Never trust, always verify

- API key authentication for all public endpoints
- Rate limiting at Load Balancer and application layers
- CORS with explicit allow lists (never use `*`)
- TLS 1.3+ for public traffic, mutual TLS for internal
- All commits GPG signed and auditable

### 3. Observability First

**Principle**: Instrument everything, assume failure

- Structured logging (JSON) with correlation IDs
- Prometheus metrics for all critical paths
- Distributed tracing with Jaeger
- Real-time dashboards in Grafana
- Alerting for all failure modes

### 4. Fail Fast, Recover Faster

**Principle**: Detect failures immediately, restore service quickly

- Health checks every 10 seconds
- Circuit breakers for external dependencies
- Automatic retries with exponential backoff
- Graceful degradation when models unavailable
- Sub-10s rollback capability

### 5. Convention Over Configuration

**Principle**: Sensible defaults, explicit overrides

- Environment variables for all configuration
- `.env.example` templates for all deployments
- Docker Compose for local development
- Kubernetes for production (future)

---

## Component Architecture

### Layer 1: API Gateway (FastAPI)

**Responsibilities**:
- HTTP request handling
- Authentication and authorization
- Request validation and sanitization
- Rate limiting enforcement
- Response formatting

**Technology**: FastAPI 0.109+ (Python 3.11+)

**Key Components**:
```
ollama/api/
├── routes/           # HTTP endpoints
│   ├── inference.py  # Text generation, embeddings
│   ├── chat.py       # Conversation endpoints
│   ├── models.py     # Model management
│   ├── documents.py  # RAG document upload
│   └── health.py     # Health checks
├── schemas/          # Pydantic request/response models
├── dependencies/     # FastAPI dependency injection
└── middleware/       # Request/response middleware
```

**Endpoints**:
- `GET /health` - Health check (no auth)
- `GET /api/v1/models` - List models
- `POST /api/v1/generate` - Text generation
- `POST /api/v1/chat` - Chat completion
- `POST /api/v1/embeddings` - Vector embeddings
- `POST /api/v1/conversations` - Conversation management
- `POST /api/v1/documents` - Document upload (RAG)

### Layer 2: Business Logic (Services)

**Responsibilities**:
- Model lifecycle management
- Inference orchestration
- Caching strategies
- Data persistence
- Vector similarity search

**Technology**: Python 3.11+ with async/await

**Key Services**:
```
ollama/services/
├── inference/
│   ├── ollama_client_main.py  # Main inference client
│   ├── generate_request.py    # Request handling
│   └── generate_response.py   # Response formatting
├── models/
│   ├── ollama_model_manager.py  # Model lifecycle
│   ├── model.py                 # Model metadata
│   └── vector.py                # Vector embeddings
├── cache/
│   └── cache.py                 # Redis caching
└── persistence/
    ├── database.py              # PostgreSQL ORM
    └── chat_message.py          # Message persistence
```

### Layer 3: Model Inference (Ollama)

**Responsibilities**:
- Load and manage AI models
- Execute inference requests
- Generate text completions
- Create embeddings

**Technology**: Ollama 0.1.0+ (standalone service)

**Models Supported**:
- LLaMA 3.2 (3B, 7B, 13B)
- Mistral (7B)
- CodeLlama (7B, 13B)
- Custom fine-tuned models

**Configuration**:
```yaml
OLLAMA_BASE_URL: http://ollama:11434
OLLAMA_MODELS_PATH: /root/.ollama/models
OLLAMA_NUM_PARALLEL: 4
OLLAMA_MAX_LOADED_MODELS: 3
```

### Layer 4: Data Persistence

#### PostgreSQL (Primary Database)

**Responsibilities**:
- User management
- Conversation history
- API key storage
- Usage analytics

**Schema**:
```sql
-- Users table
CREATE TABLE users (
  id UUID PRIMARY KEY,
  email VARCHAR(255) UNIQUE NOT NULL,
  api_key_hash VARCHAR(255) NOT NULL,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Conversations table
CREATE TABLE conversations (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  title VARCHAR(500),
  model VARCHAR(100),
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Messages table
CREATE TABLE messages (
  id UUID PRIMARY KEY,
  conversation_id UUID REFERENCES conversations(id),
  role VARCHAR(20),  -- 'user' | 'assistant' | 'system'
  content TEXT,
  created_at TIMESTAMP DEFAULT NOW()
);
```

#### Redis (Cache & Queues)

**Responsibilities**:
- Response caching (inference results)
- Semantic caching (similar prompts)
- Rate limiting counters
- Session management
- Background job queues

**Cache Strategy**:
```python
# Cache key format
key = f"inference:v1:gen:{model}:{sha256(prompt)}"

# TTL strategy
- Short prompts (< 100 chars): 1 hour
- Long prompts (> 100 chars): 15 minutes
- Embeddings: 24 hours
- Model metadata: 7 days
```

#### Qdrant (Vector Database)

**Responsibilities**:
- Document embeddings storage
- Semantic search
- RAG (Retrieval-Augmented Generation)

**Configuration**:
```yaml
QDRANT_URL: http://qdrant:6333
QDRANT_COLLECTION: ollama_documents
QDRANT_VECTOR_SIZE: 4096
QDRANT_DISTANCE: Cosine
```

### Layer 5: Monitoring & Observability

#### Prometheus (Metrics)

**Key Metrics**:
```python
# Request counters
inference_requests_total{model, status}

# Latency histograms
inference_latency_seconds{model}
  buckets: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

# Cache metrics
model_cache_hits_total{model}
model_cache_misses_total{model}

# System metrics
ollama_memory_usage_bytes
ollama_gpu_utilization_percent
```

#### Grafana (Dashboards)

**Dashboards**:
- API Performance (request rate, latency, errors)
- Model Usage (requests per model, token usage)
- Cache Performance (hit rate, evictions)
- System Health (CPU, memory, disk, network)

#### Jaeger (Distributed Tracing)

**Trace Spans**:
```
HTTP Request
├── Authentication (5ms)
├── Validation (2ms)
├── Cache Lookup (10ms)
└── Inference (1200ms)
    ├── Model Load (50ms)
    ├── Prompt Eval (200ms)
    └── Generation (950ms)
```

---

## Data Flow

### Text Generation Flow

```
1. Client Request
   ↓ HTTPS (TLS 1.3)
2. GCP Load Balancer
   ↓ Mutual TLS
   - Verify API key
   - Check rate limits
   - Log request
3. FastAPI (api/routes/inference.py)
   ↓
   - Validate request schema
   - Extract user context
4. Cache Check (Redis)
   ↓ Cache Miss
5. Inference Service (services/inference/)
   ↓
   - Build Ollama request
   - Set generation parameters
6. Ollama Engine (HTTP:11434)
   ↓
   - Load model (if not cached)
   - Execute inference
   - Generate tokens
7. Response Processing
   ↓
   - Format response
   - Cache result (Redis)
   - Log metrics (Prometheus)
8. Return to Client
   ↓ HTTPS
   - JSON response
   - Include metadata
```

### Chat Conversation Flow

```
1. Client Request (/api/v1/chat)
   ↓
2. Load Conversation History (PostgreSQL)
   ↓
   - Fetch previous messages
   - Build context window
3. Semantic Cache Check (Redis)
   ↓ Cache Miss
4. Build Full Prompt
   ↓
   - System prompt
   - Conversation history
   - Current user message
5. Ollama Inference
   ↓
6. Save Message (PostgreSQL)
   ↓
   - User message
   - Assistant response
   - Conversation metadata
7. Return Response
```

### Document Upload & RAG Flow

```
1. Client Uploads Document
   ↓ POST /api/v1/documents
2. Document Processing
   ↓
   - Validate file type (PDF, TXT, MD)
   - Extract text content
   - Split into chunks (512 tokens)
3. Generate Embeddings
   ↓ POST /api/v1/embeddings
   - For each chunk
   - Model: llama3.2
4. Store in Qdrant
   ↓
   - Vector: embedding (4096 dim)
   - Metadata: {doc_id, chunk_index, text}
5. RAG Query Flow
   ↓ POST /api/v1/documents/search
   - Query → Embedding
   - Vector search (Qdrant)
   - Retrieve top-k chunks
   - Inject into prompt context
   - Generate with context
```

---

## Technology Stack

### Core Technologies

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| **Runtime** | Python | 3.11+ | Primary language |
| **API Framework** | FastAPI | 0.109+ | HTTP API server |
| **ASGI Server** | Uvicorn | 0.27+ | Production server |
| **Database** | PostgreSQL | 15+ | Primary data store |
| **Cache** | Redis | 7+ | Caching & queues |
| **Vector DB** | Qdrant | 1.7+ | Embeddings storage |
| **Model Engine** | Ollama | 0.1.0+ | AI inference |
| **Containerization** | Docker | 24+ | Service isolation |
| **Orchestration** | Docker Compose | 2.20+ | Local deployment |

### Supporting Technologies

| Category | Technology | Purpose |
|----------|------------|---------|
| **ORM** | SQLAlchemy | 2.0+ | Database abstraction |
| **Validation** | Pydantic | 2.5+ | Data validation |
| **HTTP Client** | httpx | 0.26+ | Async HTTP requests |
| **Monitoring** | Prometheus | Latest | Metrics collection |
| **Dashboards** | Grafana | Latest | Visualization |
| **Tracing** | Jaeger | Latest | Distributed tracing |
| **Logging** | structlog | Latest | Structured logging |
| **Testing** | pytest | 8.0+ | Unit/integration tests |
| **Type Checking** | mypy | 1.8+ | Static type analysis |
| **Linting** | ruff | 0.1+ | Code quality |

### Development Tools

- **Code Editor**: VS Code with Python, Docker, Terraform extensions
- **Git**: GPG signed commits (mandatory)
- **CI/CD**: GitHub Actions (future)
- **Secrets**: GCP Secret Manager (production)

---

## Network Architecture

### Production Network Topology

```
Internet
    ↓
[Cloud Armor DDoS Protection]
    ↓
[GCP Load Balancer]
    ↓ (HTTPS:443)
    ↓
[SSL Termination]
    ↓
[Backend Service: ollama-api-backend]
    ↓ (Mutual TLS)
    ↓
[Docker Network: ollama-net (bridge)]
    ├── FastAPI Container (api:8000)
    ├── PostgreSQL Container (postgres:5432)
    ├── Redis Container (redis:6379)
    ├── Ollama Container (ollama:11434)
    └── Qdrant Container (qdrant:6333)
```

### Firewall Rules

**External Access**:
- ✅ Allow: `443/tcp` (HTTPS) → GCP Load Balancer
- ❌ Deny: All other ports from internet

**Internal Access** (Docker network only):
- ✅ Allow: `8000/tcp` (FastAPI) from Docker network
- ✅ Allow: `5432/tcp` (PostgreSQL) from Docker network
- ✅ Allow: `6379/tcp` (Redis) from Docker network
- ✅ Allow: `11434/tcp` (Ollama) from Docker network
- ✅ Allow: `6333/tcp` (Qdrant) from Docker network

### Service Communication Matrix

| Service | Talks To | Protocol | Port |
|---------|----------|----------|------|
| GCP LB | FastAPI | HTTPS (Mutual TLS) | 8000 |
| FastAPI | PostgreSQL | PostgreSQL Wire | 5432 |
| FastAPI | Redis | Redis Protocol | 6379 |
| FastAPI | Ollama | HTTP | 11434 |
| FastAPI | Qdrant | HTTP | 6333 |
| Prometheus | FastAPI | HTTP | 8000 |

---

## Security Architecture

### Authentication & Authorization

**API Key Authentication**:
```python
# Request header
Authorization: Bearer sk-<api-key>

# Validation flow
1. Extract API key from header
2. Hash API key (SHA-256)
3. Query database for user
4. Verify key is active (not revoked)
5. Check rate limits
6. Attach user context to request
```

**API Key Format**:
```
sk-<32-byte-hex-string>
Example: sk-a1b2c3d4e5f67890a1b2c3d4e5f67890a1b2c3d4e5f67890a1b2c3d4e5f67890
```

**Key Storage**:
- Keys stored as SHA-256 hashes (never plaintext)
- User table: `api_key_hash` column
- Keys rotatable (users can generate new keys)

### Rate Limiting

**Implementation**: Token bucket algorithm

**Limits**:
- **Default**: 100 requests per minute per API key
- **Burst**: 120 requests (20% burst allowance)
- **Window**: Rolling 60-second window

**Enforcement**:
```python
# Redis-based rate limiter
key = f"ratelimit:{api_key}:{minute}"
count = redis.incr(key)
if count == 1:
    redis.expire(key, 60)
if count > 100:
    raise HTTPException(429, "Rate limit exceeded")
```

### Data Encryption

**In Transit**:
- ✅ TLS 1.3 for all external traffic
- ✅ Mutual TLS for internal service communication
- ✅ Certificate auto-renewal via Google Managed SSL

**At Rest**:
- ✅ PostgreSQL: Full database encryption
- ✅ Redis: AOF persistence encrypted
- ✅ Docker volumes: Encrypted filesystem

### Network Isolation

**Zero Trust Principles**:
- All services private by default
- No direct external access to internal services
- GCP Load Balancer = single entry point
- Firewall rules enforced at GCP project level

### Audit Logging

**Requirements** (FedRAMP compliance):
- 7-year retention for all API requests
- Structured logging with user_id, request_id, timestamp
- Cloud Logging integration (production)
- Immutable audit trail

**Log Format**:
```json
{
  "timestamp": "2026-01-18T10:30:00Z",
  "request_id": "req_abc123",
  "user_id": "user_xyz789",
  "endpoint": "/api/v1/generate",
  "method": "POST",
  "status_code": 200,
  "latency_ms": 1250,
  "model": "llama3.2",
  "tokens_generated": 142
}
```

---

## Scaling Strategy

### Horizontal Scaling

**Current State**: Single-instance deployment

**Future State** (when needed):
```
GCP Load Balancer
    ↓
[Backend Service Group]
    ├── API Instance 1 (auto-scaled)
    ├── API Instance 2 (auto-scaled)
    └── API Instance 3 (auto-scaled)
    ↓
[Shared PostgreSQL] (Cloud SQL)
    ↓
[Shared Redis] (Memorystore)
    ↓
[Shared Qdrant] (Managed cluster)
```

**Scaling Triggers**:
- CPU > 70% for 5 minutes → Scale up
- Request queue > 100 → Scale up
- CPU < 30% for 10 minutes → Scale down

### Vertical Scaling

**Current Resources**:
- API: 2 CPU, 4GB RAM
- PostgreSQL: 1 CPU, 2GB RAM
- Redis: 0.5 CPU, 1GB RAM
- Ollama: 4 CPU, 8GB RAM (GPU optional)

**Scaling Limits**:
- API: Up to 8 CPU, 16GB RAM
- Ollama: Up to 16 CPU, 32GB RAM + NVIDIA GPU

### Model Caching

**Strategy**: LRU eviction with configurable limits

**Configuration**:
```yaml
OLLAMA_MAX_LOADED_MODELS: 3
OLLAMA_MODEL_CACHE_SIZE: "8GB"
OLLAMA_EVICTION_POLICY: "LRU"
```

**Cache Warming**:
- Pre-load top 3 most-used models on startup
- Track usage metrics to optimize cache

---

## Failure Modes & Resilience

### Failure Scenarios

#### 1. Ollama Service Failure

**Detection**: Health check fails 3 consecutive times

**Response**:
1. Circuit breaker opens (stop sending requests)
2. Return cached responses (if available)
3. Alert on-call engineer
4. Attempt auto-restart (3 retries)
5. Failover to backup instance (future)

**Recovery**: Service restarts, health check passes, circuit breaker closes

#### 2. PostgreSQL Connection Loss

**Detection**: Connection timeout or refused

**Response**:
1. Retry with exponential backoff (3 attempts)
2. Degrade gracefully (disable conversation history)
3. Log error, alert on-call
4. Continue serving stateless requests

**Recovery**: Connection pool reconnects automatically

#### 3. Redis Cache Failure

**Detection**: Connection error or timeout

**Response**:
1. Disable caching (continue without cache)
2. All requests hit Ollama directly
3. Log error, alert on-call
4. Performance degrades but service continues

**Recovery**: Redis restarts, cache repopulates

#### 4. GCP Load Balancer Failure

**Detection**: External monitoring (pingdom, uptime robot)

**Response**:
1. Alert on-call immediately (critical)
2. Check GCP status dashboard
3. Verify backend health checks
4. Engage GCP support if needed

**Recovery**: GCP restores service, DNS propagates

### Circuit Breaker Pattern

```python
# Implementation
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.last_failure_time = None

    async def call(self, func):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpenError()

        try:
            result = await func()
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= self.failure_threshold:
                self.state = "OPEN"
            raise
```

### Health Checks

**Endpoint**: `GET /health`

**Response** (healthy):
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2026-01-18T10:30:00Z",
  "services": {
    "database": "healthy",
    "cache": "healthy",
    "ollama": "healthy",
    "vector_db": "healthy"
  }
}
```

**Response** (degraded):
```json
{
  "status": "degraded",
  "version": "0.1.0",
  "timestamp": "2026-01-18T10:30:00Z",
  "services": {
    "database": "healthy",
    "cache": "unhealthy",
    "ollama": "healthy",
    "vector_db": "healthy"
  }
}
```

---

## Performance Characteristics

### Latency Targets

| Endpoint | Target (p99) | Measured |
|----------|--------------|----------|
| `/health` | <10ms | 5ms |
| `/api/v1/models` | <100ms | 45ms |
| `/api/v1/generate` (short) | <2s | 1.2s |
| `/api/v1/generate` (long) | <10s | 8.5s |
| `/api/v1/embeddings` | <500ms | 320ms |

### Throughput

**Current Capacity**:
- Concurrent requests: 50
- Requests per second: 25
- Tokens per second: 50 (llama3.2-3b)

**Load Test Results** (Tier 2):
- Users: 50
- Total requests: 7,162
- Success rate: 100%
- P95 latency: 75ms
- P99 latency: 120ms

### Resource Usage

**Baseline** (idle):
- CPU: 5%
- Memory: 2GB
- Disk: 15GB (models + data)
- Network: <1 Mbps

**Peak** (50 concurrent users):
- CPU: 65%
- Memory: 6GB
- Disk I/O: 50 MB/s (model loading)
- Network: 10 Mbps

---

## Future Enhancements

### Phase 1: Kubernetes Migration (Q2 2026)

- Migrate from Docker Compose to Kubernetes
- Implement auto-scaling (HPA)
- Multi-region deployment
- Zero-downtime deployments

### Phase 2: Advanced Caching (Q2 2026)

- Semantic cache with vector similarity
- Predictive cache warming
- Multi-tier cache hierarchy
- Cache analytics dashboard

### Phase 3: Model Fine-Tuning (Q3 2026)

- Custom model training pipeline
- LoRA adapter support
- Model versioning and A/B testing
- Training job scheduler

### Phase 4: Enterprise Features (Q4 2026)

- Multi-tenancy support
- SSO integration (OAuth 2.0)
- Role-based access control (RBAC)
- Usage billing and quotas

---

## References

- [API Documentation](API.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Operational Runbooks](RUNBOOKS.md)
- [GCP Landing Zone](https://github.com/kushin77/gcp-landing-zone)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [Ollama Documentation](https://ollama.ai/docs)

---

**Last Updated**: January 18, 2026
**Version**: 0.1.0
**Next Review**: April 18, 2026
