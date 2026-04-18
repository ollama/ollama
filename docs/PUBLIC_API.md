# Public API Quick Reference: elevatediq.ai/ollama

**Endpoint**: `https://elevatediq.ai/ollama`  
**Authentication**: X-API-Key header or Bearer token  
**Rate Limit**: 100 requests/minute (burst to 150)

---

## Quick Start

### Get API Key

Contact the elevatediq.ai team to request an API key.

### Test Endpoint

```bash
API_KEY="your-api-key"

# Health check
curl -H "X-API-Key: $API_KEY" \
  https://elevatediq.ai/ollama/health
```

---

## API Methods

### Health Check

```bash
curl -H "X-API-Key: $API_KEY" \
  https://elevatediq.ai/ollama/health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "production",
  "public_url": "https://elevatediq.ai/ollama"
}
```

### List Models

```bash
curl -H "X-API-Key: $API_KEY" \
  https://elevatediq.ai/ollama/api/models
```

**Response**:
```json
{
  "models": [
    {"name": "llama2", "size": "7b", "quantization": "q4_K_M"},
    {"name": "mistral", "size": "7b", "quantization": "q5_K_M"}
  ]
}
```

### Generate Text

```bash
curl -X POST https://elevatediq.ai/ollama/api/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "model": "llama2",
    "prompt": "What is artificial intelligence?",
    "stream": false,
    "temperature": 0.7
  }'
```

**Parameters**:
- `model` (required): Model identifier (string)
- `prompt` (required): Input text (string)
- `stream` (optional): Stream response (boolean, default: false)
- `temperature` (optional): Sampling temperature 0-2 (float, default: 0.7)
- `top_p` (optional): Nucleus sampling (float, default: 0.95)
- `top_k` (optional): Top-k sampling (int, default: 40)

**Response**:
```json
{
  "model": "llama2",
  "response": "Artificial intelligence is...",
  "done": true,
  "context": []
}
```

### Chat Completions (OpenAI-compatible)

```bash
curl -X POST https://elevatediq.ai/ollama/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "model": "llama2",
    "messages": [
      {"role": "system", "content": "You are a helpful AI assistant"},
      {"role": "user", "content": "Explain machine learning"}
    ],
    "temperature": 0.7,
    "stream": false
  }'
```

**Response**:
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1673564400,
  "model": "llama2",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Machine learning is a subset of artificial intelligence..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 32,
    "completion_tokens": 128,
    "total_tokens": 160
  }
}
```

### Embeddings (OpenAI-compatible)

```bash
curl -X POST https://elevatediq.ai/ollama/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "model": "embedding-model",
    "input": "Generate embeddings for this text"
  }'
```

**Response**:
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.123, -0.456, 0.789, ...],
      "index": 0
    }
  ],
  "model": "embedding-model",
  "usage": {
    "prompt_tokens": 8,
    "total_tokens": 8
  }
}
```

### Admin Stats

```bash
curl -H "X-API-Key: $API_KEY" \
  https://elevatediq.ai/ollama/admin/stats
```

**Response**:
```json
{
  "uptime_seconds": 3600,
  "gpu_memory_used": 5000,
  "gpu_memory_total": 24000,
  "requests_total": 1500,
  "errors_total": 5,
  "average_latency_ms": 850
}
```

---

## Authentication

### API Key Header

```bash
curl -H "X-API-Key: YOUR-API-KEY" https://elevatediq.ai/ollama/health
```

### Bearer Token

```bash
curl -H "Authorization: Bearer YOUR-API-KEY" \
  https://elevatediq.ai/ollama/health
```

---

## Error Handling

### Common Error Responses

**401 Unauthorized**
```json
{"detail": "API key required"}
```

**429 Too Many Requests**
```json
{"detail": "Rate limit exceeded: 100 requests per minute"}
```

**500 Internal Server Error**
```json
{"detail": "Internal server error", "request_id": "abc-123"}
```

---

## Python SDK

### Installation

```bash
pip install ollama
```

### Usage

```python
from ollama import Client

# Connect to public endpoint
client = Client(
    base_url="https://elevatediq.ai/ollama",
    api_key="your-api-key"
)

# Generate text
response = client.generate(
    model="llama2",
    prompt="Explain quantum computing",
    stream=False
)
print(response)

# Chat
response = client.chat(
    model="llama2",
    messages=[
        {"role": "system", "content": "You are an expert"},
        {"role": "user", "content": "What is RAG?"}
    ]
)
print(response)

# Embeddings
embeddings = client.embeddings(
    model="embedding-model",
    input_text="Generate vector representation"
)
print(embeddings)

# List models
models = client.list_models()
print(models)

# Health check
health = client.health()
print(health)
```

---

## Rate Limiting

- **Base limit**: 100 requests per minute per API key
- **Burst capacity**: 150 requests (short bursts allowed)
- **Headers returned**:
  - `X-RateLimit-Limit`: 100
  - `X-RateLimit-Remaining`: Requests remaining
  - `X-RateLimit-Reset`: Unix timestamp when limit resets

### Handling Rate Limits

```python
import time
from ollama import Client
import httpx

client = Client(
    base_url="https://elevatediq.ai/ollama",
    api_key="your-api-key"
)

try:
    response = client.generate(
        model="llama2",
        prompt="Hello"
    )
except httpx.HTTPStatusError as e:
    if e.response.status_code == 429:
        reset_time = int(e.response.headers.get("X-RateLimit-Reset"))
        wait_time = reset_time - time.time()
        print(f"Rate limited. Waiting {wait_time} seconds...")
        time.sleep(wait_time + 1)
        # Retry
```

---

## Response Headers

All responses include security headers:

```
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
X-Request-ID: unique-request-identifier
```

---

## Streaming Responses

Enable streaming for long-running requests:

```bash
curl -X POST https://elevatediq.ai/ollama/api/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "model": "llama2",
    "prompt": "Write a story",
    "stream": true
  }' \
  --stream
```

Each line is a JSON object:
```json
{"response":"word","done":false}
{"response":" by","done":false}
{"response":" word","done":true}
```

### Python Streaming

```python
response = client.generate(
    model="llama2",
    prompt="Write a story",
    stream=True
)

for chunk in response:
    print(chunk.response, end="", flush=True)
```

---

## CORS Support

Public endpoint supports CORS requests from:
- `https://elevatediq.ai`
- `https://*.elevatediq.ai`
- `http://localhost:3000` (development)
- `http://localhost:8080` (development)

### Browser Request Example

```javascript
const API_KEY = "your-api-key";

fetch("https://elevatediq.ai/ollama/api/models", {
    headers: {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
})
.then(r => r.json())
.then(data => console.log(data))
.catch(err => console.error(err));
```

---

## Best Practices

1. **Never hardcode API keys** - Use environment variables
2. **Cache responses** where possible to reduce rate limit usage
3. **Use appropriate models** for your use case
4. **Handle errors gracefully** with exponential backoff
5. **Monitor your usage** with admin stats endpoint
6. **Use streaming** for long responses
7. **Set reasonable timeouts** (300s default)

---

## Support

- **Documentation**: https://github.com/kushin77/ollama
- **Issues**: https://github.com/kushin77/ollama/issues
- **Contact**: support@elevatediq.ai

---

**Version**: 1.0.0  
**Last Updated**: January 12, 2026  
**Status**: Production Ready
