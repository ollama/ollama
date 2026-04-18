# Ollama Agentic API Documentation

## Overview

The Ollama Agentic API enables autonomous AI agents to execute complex, multi-step tasks using the Ollama model server. Agents can reason, plan, use tools, and persist state across executions.

**Status**: ✅ Production-Ready (Landing Zone Compliant)  
**Version**: 0.1.0  
**Environment**: GCP Cloud Run (via GCP Load Balancer)  
**Endpoint**: `https://elevatediq.ai/ollama/api/v1/agents`

---

## Architecture

### Services

1. **Agent Service** (`prod-ollama-agents-service`)
   - Cloud Run service for individual agent execution
   - Auto-scales from 0-10 instances
   - Processes inference requests through Ollama
   - Publishes results to Pub/Sub

2. **Orchestrator Service** (`prod-ollama-orchestrator-service`)
   - Cloud Run service for task coordination
   - Manages task distribution via Cloud Tasks
   - Tracks agent state in Firestore
   - Provides orchestration APIs

3. **Task Queue** (`prod-ollama-agent-tasks`)
   - Cloud Tasks queue for reliable task delivery
   - Retry logic with exponential backoff
   - Max 100 concurrent dispatches

4. **State Storage** (Firestore)
   - Persistent agent state and conversation history
   - Document-based NoSQL for flexibility
   - Point-in-time recovery enabled

5. **Results Streaming** (Pub/Sub)
   - Real-time agent execution results
   - Dead-letter queue for failed messages
   - 24-hour message retention

6. **Analytics** (BigQuery)
   - Execution logs and metrics
   - Performance analysis and cost tracking
   - CMEK encryption enabled

---

## API Endpoints

### Execute Agent Task

```
POST /api/v1/agents/{agent_id}/execute
Authorization: Bearer {api_key}
Content-Type: application/json

Request:
{
  "task_id": "task-abc123",
  "prompt": "Summarize the key findings from the research paper...",
  "context": {
    "topic": "machine-learning",
    "max_tokens": 2048,
    "temperature": 0.7
  }
}

Response (202 Accepted):
{
  "success": true,
  "data": {
    "task_id": "task-abc123",
    "agent_id": "agent-reasoning-v1",
    "status": "running",
    "estimated_duration_ms": 5000,
    "result_topic": "projects/my-project/topics/prod-ollama-agent-results"
  },
  "metadata": {
    "request_id": "req-xyz789",
    "timestamp": "2026-01-26T10:30:00Z"
  }
}
```

### Get Task Status

```
GET /api/v1/agents/tasks/{task_id}
Authorization: Bearer {api_key}

Response:
{
  "success": true,
  "data": {
    "task_id": "task-abc123",
    "agent_id": "agent-reasoning-v1",
    "status": "success",
    "output": {
      "summary": "The paper presents...",
      "key_findings": [...]
    },
    "metrics": {
      "duration_ms": 4250,
      "tokens_used": 1542,
      "cost_usd": 0.045
    }
  },
  "metadata": {
    "request_id": "req-xyz789",
    "timestamp": "2026-01-26T10:30:05Z"
  }
}
```

### List Available Agents

```
GET /api/v1/agents
Authorization: Bearer {api_key}

Response:
{
  "success": true,
  "data": [
    {
      "agent_id": "agent-reasoning-v1",
      "model": "llama3.2",
      "capabilities": ["reasoning", "planning", "tool_use"],
      "max_tokens": 2048,
      "status": "ready"
    },
    {
      "agent_id": "agent-research-v1",
      "model": "neural-chat",
      "capabilities": ["memory", "multi_step", "streaming"],
      "max_tokens": 4096,
      "status": "ready"
    }
  ],
  "metadata": {
    "request_id": "req-xyz789",
    "timestamp": "2026-01-26T10:30:00Z"
  }
}
```

### Stream Agent Results

```
GET /api/v1/agents/tasks/{task_id}/stream
Authorization: Bearer {api_key}

Response (Server-Sent Events):
data: {"status":"running","progress":25}
data: {"status":"running","progress":50}
data: {"status":"running","progress":75}
data: {"status":"success","output":"...","metrics":{...}}
```

---

## Authentication

All agentic API endpoints require authentication via:

1. **API Key** (for public clients)
   ```
   Authorization: Bearer sk-1234567890abcdef
   ```

2. **Workload Identity** (for GCP services)
   - Service account: `prod-ollama-agents@project.iam.gserviceaccount.com`
   - Workload Identity binding enabled
   - Mutual TLS for internal communication

---

## Error Handling

All errors follow the standard Ollama error format:

```json
{
  "success": false,
  "error": {
    "code": "AGENT_TIMEOUT",
    "message": "Agent execution timed out after 300s",
    "details": {
      "task_id": "task-abc123",
      "timeout_seconds": 300,
      "retry_after": 60
    }
  },
  "metadata": {
    "request_id": "req-xyz789",
    "timestamp": "2026-01-26T10:30:00Z"
  }
}
```

### Error Codes

| Code | Status | Description |
|------|--------|-------------|
| AGENT_NOT_FOUND | 404 | Requested agent does not exist |
| TASK_NOT_FOUND | 404 | Task not found in state store |
| AGENT_TIMEOUT | 504 | Agent execution exceeded timeout |
| INVALID_INPUT | 400 | Input validation failed |
| RATE_LIMIT_EXCEEDED | 429 | Rate limit exceeded (100 req/min) |
| INTERNAL_ERROR | 500 | Internal server error |

---

## Rate Limiting

**Production Limits**:
- 100 requests per minute per API key
- 10 concurrent executions per agent
- 300 second timeout per execution

**Retry Strategy**:
- Automatic retry with exponential backoff
- Max 3 attempts per task
- Backoff: 1s, 2s, 4s

---

## Monitoring & Observability

### Metrics

All metrics published to Cloud Monitoring with 60-second granularity:

```
ollama_agent_executions_total{agent_id, status}
ollama_agent_execution_duration_ms{agent_id, quantile}
ollama_agent_tokens_used{agent_id}
ollama_agent_cost_usd{agent_id}
```

### Logs

Structured logging to Cloud Logging with JSON format:

```json
{
  "timestamp": "2026-01-26T10:30:00Z",
  "severity": "INFO",
  "message": "task_execution_completed",
  "task_id": "task-abc123",
  "agent_id": "agent-reasoning-v1",
  "status": "success",
  "duration_ms": 4250,
  "tokens_used": 1542,
  "cost_usd": 0.045,
  "request_id": "req-xyz789"
}
```

### Alerting

Automatic alerts trigger on:
- Error rate > 1%
- P99 latency > 10 seconds
- Agent unavailability
- Cost anomaly (> 2 std dev)

---

## Landing Zone Compliance

### 8-Point Mandate ✅

1. **Infrastructure Alignment** ✅
   - GCP Load Balancer as single entry point
   - Cloud Run auto-scaling for elasticity
   - Multi-region failover ready

2. **Mandatory Labeling** ✅
   - All resources tagged with 24 labels
   - Cost attribution: `component=agents`
   - Lifecycle: `lifecycle_state=active`

3. **Naming Conventions** ✅
   - `prod-ollama-agents-service` (service)
   - `prod-ollama-orchestrator-service` (orchestrator)
   - `prod-ollama-agent-tasks` (task queue)

4. **Zero Trust Auth** ✅
   - Workload Identity Federation enabled
   - No service account keys
   - Mutual TLS for internal services

5. **No Root Chaos** ✅
   - Code organized in Level 3: `ollama/agents/`
   - Terraform in Level 2: `docker/terraform/04-agentic/`
   - Docs in Level 2: `docs/agents/`

6. **GPG Signed Commits** ✅
   - All commits signed with `git commit -S`
   - Pre-commit hooks enforced

7. **PMO Metadata** ✅
   - 24 labels defined in `pmo.yaml`
   - Cost center: `ai-infrastructure`
   - Team: `ai-platform`

8. **Automated Compliance** ✅
   - `scripts/validate_landing_zone_compliance.py` passing
   - Terraform linting: `tflint` ✅
   - Type checking: `mypy --strict` ✅

---

## Deployment

### Prerequisites

```bash
# GCP setup
gcloud auth login
gcloud config set project gcp-eiq

# Terraform
terraform init
terraform validate
tflint
```

### Deploy Agentic Infrastructure

```bash
# Create terraform variables
cat > terraform.tfvars <<EOF
project_id = "gcp-eiq"
gcp_project = "gcp-eiq"
region = "us-central1"
environment = "production"
application = "ollama"
component = "agents"

agent_image_uri = "us-central1-docker.pkg.dev/gcp-eiq/prod-ollama-agents/agent:latest"
orchestrator_image_uri = "us-central1-docker.pkg.dev/gcp-eiq/prod-ollama-agents/orchestrator:latest"

resource_labels = {
  environment = "production"
  cost_center = "ai-infrastructure"
  team = "ai-platform"
  managed_by = "terraform"
  created_by = "infrastructure-team@elevatediq.ai"
  created_date = "2026-01-26"
  lifecycle_state = "active"
  teardown_date = "none"
  retention_days = "3650"
  product = "ollama"
  component = "agents"
  tier = "critical"
  compliance = "fedramp"
  version = "0.1.0"
  stack = "python-3.11-fastapi-gcp"
  backup_strategy = "continuous"
  monitoring_enabled = "true"
  budget_owner = "infrastructure-team@elevatediq.ai"
  project_code = "OLLAMA-2026-001"
  monthly_budget_usd = "5000"
  chargeback_unit = "ai-infrastructure"
  git_repository = "github.com/kushin77/ollama"
  git_branch = "main"
  auto_delete = "false"
}
EOF

# Deploy
cd docker/terraform/04-agentic
terraform plan
terraform apply
```

---

## Testing

### Unit Tests

```bash
pytest tests/unit/agents/ -v --cov=ollama.agents
```

### Integration Tests

```bash
pytest tests/integration/agents/ -v --cov=ollama.agents
```

### Load Testing

```bash
# Using locust (100 concurrent users, 1000 requests)
locust -f load-tests/agents/locustfile.py --headless -u 100 -r 10 -t 60s
```

---

## Troubleshooting

### Agent Service Not Responding

```bash
# Check Cloud Run service status
gcloud run services describe prod-ollama-agents-service --region us-central1

# View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=prod-ollama-agents-service" --limit 50
```

### Task Stuck in Queue

```bash
# Inspect Cloud Tasks queue
gcloud tasks queues describe prod-ollama-agent-tasks --location us-central1

# Purge queue (production: use with caution)
gcloud tasks queues purge prod-ollama-agent-tasks --location us-central1
```

### High Latency

```bash
# Check agent metrics
gcloud monitoring time-series list --filter 'metric.type=ollama_agent_execution_duration_ms'

# View profiler data
gcloud profiler datasets list --deployment-target=cloud_run
```

---

## References

- [GCP Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Cloud Tasks Documentation](https://cloud.google.com/tasks/docs)
- [Firestore Documentation](https://cloud.google.com/firestore/docs)
- [Pub/Sub Documentation](https://cloud.google.com/pubsub/docs)
- [Landing Zone Standards](./docs/LANDING_ZONE_QUICK_REFERENCE.md)
- [Ollama Model Server](https://github.com/ollama/ollama)

---

**Last Updated**: January 26, 2026  
**Version**: 0.1.0  
**Status**: ✅ Production-Ready (Landing Zone Compliant)
