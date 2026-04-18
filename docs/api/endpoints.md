# API Endpoints

Complete reference for Ollama API endpoints.

## Base URL

```
https://elevatediq.ai/ollama
```

## Health Check

**Endpoint**: `GET /api/v1/health`

No authentication required. Used by load balancer for health checks.

=== "Request"
`bash
    curl https://elevatediq.ai/ollama/api/v1/health
    `

=== "Response"
`json
    {
      "status": "healthy",
      "timestamp": "2026-01-18T10:30:00Z",
      "version": "1.0.0"
    }
    `

## Text Generation

**Endpoint**: `POST /api/v1/generate`

Generate text completion for a prompt.

=== "Request"
`bash
    curl -X POST https://elevatediq.ai/ollama/api/v1/generate \
      -H "Authorization: Bearer {api_key}" \
      -H "Content-Type: application/json" \
      -d '{
        "model": "llama2",
        "prompt": "Explain quantum computing",
        "stream": false,
        "temperature": 0.7,
        "max_tokens": 500
      }'
    `

=== "Response"
`json
    {
      "success": true,
      "data": {
        "text": "Quantum computing is...",
        "model": "llama2",
        "tokens": 124,
        "inference_time_ms": 2150
      },
      "metadata": {
        "request_id": "req_abc123",
        "timestamp": "2026-01-18T10:30:00Z"
      }
    }
    `

## Chat Completion

**Endpoint**: `POST /api/v1/chat`

Generate chat responses with conversation history.

=== "Request"
`bash
    curl -X POST https://elevatediq.ai/ollama/api/v1/chat \
      -H "Authorization: Bearer {api_key}" \
      -H "Content-Type: application/json" \
      -d '{
        "model": "llama2",
        "messages": [
          {"role": "user", "content": "Hello"},
          {"role": "assistant", "content": "Hi there!"},
          {"role": "user", "content": "How are you?"}
        ],
        "temperature": 0.7,
        "max_tokens": 200
      }'
    `

=== "Response"
`json
    {
      "success": true,
      "data": {
        "message": "I'm doing well, thank you for asking!",
        "model": "llama2",
        "tokens": 18
      },
      "metadata": {
        "request_id": "req_def456",
        "timestamp": "2026-01-18T10:30:00Z"
      }
    }
    `

## List Models

**Endpoint**: `GET /api/v1/models`

List available models.

=== "Response"
`json
    {
      "success": true,
      "data": {
        "models": [
          {
            "name": "llama2",
            "size": "7B",
            "parameters": 7000000000,
            "context_length": 4096
          },
          {
            "name": "llama2-13b",
            "size": "13B",
            "parameters": 13000000000,
            "context_length": 4096
          }
        ]
      }
    }
    `

## Error Responses

All endpoints return consistent error format:

```json
{
  "success": false,
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Missing required field: model",
    "details": {
      "field": "model",
      "reason": "required"
    }
  },
  "metadata": {
    "request_id": "req_xyz789",
    "timestamp": "2026-01-18T10:30:00Z"
  }
}
```

### Common Error Codes

| Code                  | Status | Description              |
| --------------------- | ------ | ------------------------ |
| `INVALID_REQUEST`     | 400    | Malformed request        |
| `UNAUTHORIZED`        | 401    | Missing/invalid API key  |
| `RATE_LIMIT_EXCEEDED` | 429    | Rate limit exceeded      |
| `MODEL_NOT_FOUND`     | 404    | Model not available      |
| `INFERENCE_TIMEOUT`   | 504    | Generation took too long |
| `INTERNAL_ERROR`      | 500    | Server error             |

## Rate Limiting

All endpoints are rate limited: **100 requests per 60 seconds**.

Rate limit headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1705586400
```

When limit exceeded:

```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Try again after 60 seconds.",
    "details": {
      "limit": 100,
      "window": "60s",
      "retry_after": 60
    }
  }
}
```
