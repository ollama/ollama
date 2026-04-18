# Ollama API Documentation

**Version**: 0.1.0
**Base URL**: `https://elevatediq.ai/ollama`
**Protocol**: HTTPS only (TLS 1.3+)
**Authentication**: API Key (Bearer token)

---

## Table of Contents

- [Authentication](#authentication)
- [Rate Limiting](#rate-limiting)
- [Error Handling](#error-handling)
- [Endpoints](#endpoints)
  - [Health Check](#health-check)
  - [Models](#models)
  - [Text Generation](#text-generation)
  - [Chat Completion](#chat-completion)
  - [Embeddings](#embeddings)
  - [Conversations](#conversations)
  - [Documents](#documents)
  - [Usage Analytics](#usage-analytics)
- [Request/Response Schemas](#requestresponse-schemas)
- [Versioning](#versioning)
- [Examples](#examples)

---

## Authentication

All API endpoints (except `/health`) require API key authentication.

### Obtaining an API Key

Contact your system administrator or use the authentication portal at `https://elevatediq.ai/ollama/auth`.

### Using the API Key

Include your API key in the `Authorization` header:

```http
Authorization: Bearer sk-your-api-key-here
```

### Example Request

```bash
curl -H "Authorization: Bearer sk-abc123..." \
     https://elevatediq.ai/ollama/api/v1/models
```

### Authentication Errors

| Status Code | Error | Description |
|-------------|-------|-------------|
| `401` | `UNAUTHORIZED` | Missing or invalid API key |
| `403` | `FORBIDDEN` | API key is valid but lacks required permissions |
| `429` | `RATE_LIMIT_EXCEEDED` | Too many requests (see Rate Limiting) |

---

## Rate Limiting

**Default Limits**: 100 requests per minute per API key

### Rate Limit Headers

All responses include rate limit information:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1642521600
```

### Handling Rate Limits

When rate limited, API returns:

```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Retry after 60 seconds.",
    "details": {
      "limit": 100,
      "window": "1m",
      "retry_after": 60
    }
  }
}
```

**Best Practice**: Implement exponential backoff when receiving 429 responses.

---

## Error Handling

### Standard Error Response

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {
      "field": "additional context"
    }
  },
  "metadata": {
    "request_id": "req_abc123",
    "timestamp": "2026-01-18T10:30:00Z"
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_REQUEST` | 400 | Malformed request body or parameters |
| `UNAUTHORIZED` | 401 | Missing or invalid authentication |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `MODEL_NOT_FOUND` | 404 | Requested model does not exist |
| `INFERENCE_TIMEOUT` | 504 | Inference took too long (>30s) |
| `INTERNAL_ERROR` | 500 | Unexpected server error |

---

## Endpoints

### Health Check

**Endpoint**: `GET /health`

**Authentication**: Not required

**Description**: Check API health and readiness status

**Response**:

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2026-01-18T10:30:00Z"
}
```

**Example**:

```bash
curl https://elevatediq.ai/ollama/health
```

---

### Models

#### List Available Models

**Endpoint**: `GET /api/v1/models`

**Authentication**: Required

**Description**: List all available AI models

**Response**:

```json
{
  "success": true,
  "data": {
    "models": [
      {
        "name": "llama3.2",
        "size": "3B",
        "format": "gguf",
        "family": "llama",
        "modified_at": "2026-01-15T00:00:00Z"
      },
      {
        "name": "mistral",
        "size": "7B",
        "format": "gguf",
        "family": "mistral",
        "modified_at": "2026-01-10T00:00:00Z"
      }
    ],
    "total": 2
  }
}
```

**Example**:

```bash
curl -H "Authorization: Bearer sk-abc123..." \
     https://elevatediq.ai/ollama/api/v1/models
```

#### Get Model Details

**Endpoint**: `GET /api/v1/models/{model_name}`

**Authentication**: Required

**Parameters**:
- `model_name` (path): Name of the model (e.g., `llama3.2`)

**Response**:

```json
{
  "success": true,
  "data": {
    "name": "llama3.2",
    "size": "3B",
    "format": "gguf",
    "family": "llama",
    "parameter_size": "3B",
    "quantization_level": "Q4_0",
    "modified_at": "2026-01-15T00:00:00Z",
    "details": {
      "parent_model": "llama3",
      "format_version": "v2",
      "model_file": "llama3-2-3b-q4_0.gguf"
    }
  }
}
```

**Example**:

```bash
curl -H "Authorization: Bearer sk-abc123..." \
     https://elevatediq.ai/ollama/api/v1/models/llama3.2
```

#### Pull Model

**Endpoint**: `POST /api/v1/models/pull`

**Authentication**: Required

**Request Body**:

```json
{
  "name": "llama3.2",
  "insecure": false,
  "stream": true
}
```

**Response** (if stream=false):

```json
{
  "success": true,
  "data": {
    "status": "success",
    "model": "llama3.2"
  }
}
```

**Response** (if stream=true): Server-Sent Events (SSE)

```
data: {"status": "downloading", "progress": 0.25}
data: {"status": "downloading", "progress": 0.50}
data: {"status": "downloading", "progress": 1.0}
data: {"status": "success", "model": "llama3.2"}
```

---

### Text Generation

**Endpoint**: `POST /api/v1/generate`

**Authentication**: Required

**Description**: Generate text completion from a prompt

**Request Body**:

```json
{
  "model": "llama3.2",
  "prompt": "Explain quantum computing in simple terms:",
  "system": "You are a helpful science teacher.",
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "num_predict": 256,
  "stream": false
}
```

**Parameters**:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | Yes | - | Model name (e.g., `llama3.2`) |
| `prompt` | string | Yes | - | Input prompt for generation |
| `system` | string | No | `null` | System prompt (instructions) |
| `temperature` | float | No | `0.7` | Randomness (0.0-2.0) |
| `top_p` | float | No | `0.9` | Nucleus sampling (0.0-1.0) |
| `top_k` | int | No | `40` | Top-K sampling |
| `num_predict` | int | No | `128` | Max tokens to generate |
| `stream` | bool | No | `false` | Enable streaming response |

**Response** (non-streaming):

```json
{
  "success": true,
  "data": {
    "model": "llama3.2",
    "response": "Quantum computing is a revolutionary approach to computation...",
    "done": true,
    "total_duration": 1250000000,
    "load_duration": 50000000,
    "prompt_eval_count": 12,
    "prompt_eval_duration": 200000000,
    "eval_count": 142,
    "eval_duration": 1000000000
  },
  "metadata": {
    "request_id": "req_abc123",
    "timestamp": "2026-01-18T10:30:00Z"
  }
}
```

**Response** (streaming): Server-Sent Events

```
data: {"response": "Quantum", "done": false}
data: {"response": " computing", "done": false}
data: {"response": " is", "done": false}
data: {"response": " a", "done": false}
...
data: {"response": ".", "done": true, "total_duration": 1250000000}
```

**Example**:

```bash
curl -X POST \
     -H "Authorization: Bearer sk-abc123..." \
     -H "Content-Type: application/json" \
     -d '{
       "model": "llama3.2",
       "prompt": "What is AI?",
       "temperature": 0.7,
       "num_predict": 100
     }' \
     https://elevatediq.ai/ollama/api/v1/generate
```

---

### Chat Completion

**Endpoint**: `POST /api/v1/chat`

**Authentication**: Required

**Description**: Multi-turn conversation with context

**Request Body**:

```json
{
  "model": "llama3.2",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What is the capital of France?"
    },
    {
      "role": "assistant",
      "content": "The capital of France is Paris."
    },
    {
      "role": "user",
      "content": "What is its population?"
    }
  ],
  "temperature": 0.7,
  "stream": false
}
```

**Parameters**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Model name |
| `messages` | array | Yes | Conversation history |
| `temperature` | float | No | Randomness (0.0-2.0) |
| `top_p` | float | No | Nucleus sampling |
| `stream` | bool | No | Enable streaming |

**Message Object**:

```json
{
  "role": "user|assistant|system",
  "content": "Message text"
}
```

**Response**:

```json
{
  "success": true,
  "data": {
    "model": "llama3.2",
    "message": {
      "role": "assistant",
      "content": "The population of Paris is approximately 2.2 million people..."
    },
    "done": true,
    "total_duration": 1800000000
  }
}
```

**Example**:

```bash
curl -X POST \
     -H "Authorization: Bearer sk-abc123..." \
     -H "Content-Type: application/json" \
     -d '{
       "model": "llama3.2",
       "messages": [
         {"role": "user", "content": "Hello!"}
       ]
     }' \
     https://elevatediq.ai/ollama/api/v1/chat
```

---

### Embeddings

**Endpoint**: `POST /api/v1/embeddings`

**Authentication**: Required

**Description**: Generate vector embeddings for text

**Request Body**:

```json
{
  "model": "llama3.2",
  "prompt": "The quick brown fox jumps over the lazy dog"
}
```

**Response**:

```json
{
  "success": true,
  "data": {
    "embedding": [0.123, -0.456, 0.789, ...],
    "dimension": 4096
  }
}
```

**Example**:

```bash
curl -X POST \
     -H "Authorization: Bearer sk-abc123..." \
     -H "Content-Type: application/json" \
     -d '{
       "model": "llama3.2",
       "prompt": "Hello world"
     }' \
     https://elevatediq.ai/ollama/api/v1/embeddings
```

---

### Conversations

#### Create Conversation

**Endpoint**: `POST /api/v1/conversations`

**Authentication**: Required

**Request Body**:

```json
{
  "title": "My conversation",
  "model": "llama3.2"
}
```

**Response**:

```json
{
  "success": true,
  "data": {
    "id": "conv_abc123",
    "title": "My conversation",
    "model": "llama3.2",
    "created_at": "2026-01-18T10:30:00Z",
    "updated_at": "2026-01-18T10:30:00Z"
  }
}
```

#### Get Conversation History

**Endpoint**: `GET /api/v1/conversations/{conversation_id}`

**Authentication**: Required

**Response**:

```json
{
  "success": true,
  "data": {
    "id": "conv_abc123",
    "title": "My conversation",
    "messages": [
      {
        "role": "user",
        "content": "Hello",
        "timestamp": "2026-01-18T10:30:00Z"
      },
      {
        "role": "assistant",
        "content": "Hi! How can I help?",
        "timestamp": "2026-01-18T10:30:01Z"
      }
    ]
  }
}
```

---

### Documents

#### Upload Document

**Endpoint**: `POST /api/v1/documents`

**Authentication**: Required

**Description**: Upload document for RAG (Retrieval-Augmented Generation)

**Request**: Multipart form-data

```
POST /api/v1/documents
Content-Type: multipart/form-data

file: [binary content]
metadata: {"title": "My Document", "tags": ["research"]}
```

**Response**:

```json
{
  "success": true,
  "data": {
    "document_id": "doc_abc123",
    "filename": "document.pdf",
    "size_bytes": 102400,
    "uploaded_at": "2026-01-18T10:30:00Z"
  }
}
```

#### Search Documents

**Endpoint**: `POST /api/v1/documents/search`

**Authentication**: Required

**Request Body**:

```json
{
  "query": "quantum computing applications",
  "limit": 10
}
```

**Response**:

```json
{
  "success": true,
  "data": {
    "results": [
      {
        "document_id": "doc_abc123",
        "chunk": "Quantum computing has applications in...",
        "score": 0.95
      }
    ],
    "total": 5
  }
}
```

---

### Usage Analytics

**Endpoint**: `GET /api/v1/usage`

**Authentication**: Required

**Description**: Get API usage statistics

**Query Parameters**:
- `start_date` (optional): ISO 8601 date (e.g., `2026-01-01`)
- `end_date` (optional): ISO 8601 date
- `group_by` (optional): `day|week|month`

**Response**:

```json
{
  "success": true,
  "data": {
    "period": {
      "start": "2026-01-01T00:00:00Z",
      "end": "2026-01-18T23:59:59Z"
    },
    "total_requests": 15234,
    "total_tokens": 3456789,
    "breakdown": [
      {
        "date": "2026-01-15",
        "requests": 523,
        "tokens": 123456
      }
    ]
  }
}
```

---

## Request/Response Schemas

### GenerateRequest

```typescript
{
  model: string;           // Required: Model name
  prompt: string;          // Required: Input prompt
  system?: string;         // Optional: System instructions
  temperature?: number;    // Optional: 0.0-2.0 (default: 0.7)
  top_p?: number;          // Optional: 0.0-1.0 (default: 0.9)
  top_k?: number;          // Optional: (default: 40)
  num_predict?: number;    // Optional: (default: 128)
  stream?: boolean;        // Optional: (default: false)
}
```

### GenerateResponse

```typescript
{
  success: boolean;
  data: {
    model: string;
    response: string;
    done: boolean;
    total_duration: number;    // nanoseconds
    load_duration: number;     // nanoseconds
    prompt_eval_count: number;
    prompt_eval_duration: number;
    eval_count: number;
    eval_duration: number;
  };
  metadata: {
    request_id: string;
    timestamp: string;        // ISO 8601
  };
}
```

---

## Versioning

**Current Version**: v1

**API Version Format**: `/api/v{major}`

**Versioning Policy**:
- **Major version** (`v1`, `v2`): Breaking changes
- **Minor updates**: Backward-compatible additions
- **Patch updates**: Bug fixes

**Deprecation Policy**:
- Old versions supported for minimum 6 months
- Deprecation notices sent 3 months in advance
- Header: `X-API-Deprecation: 2026-06-01`

---

## Examples

### Python Example

```python
import requests

API_KEY = "sk-your-api-key"
BASE_URL = "https://elevatediq.ai/ollama"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Generate text
response = requests.post(
    f"{BASE_URL}/api/v1/generate",
    headers=headers,
    json={
        "model": "llama3.2",
        "prompt": "Explain AI in simple terms",
        "temperature": 0.7,
        "num_predict": 100
    }
)

data = response.json()
print(data["data"]["response"])
```

### JavaScript Example

```javascript
const API_KEY = "sk-your-api-key";
const BASE_URL = "https://elevatediq.ai/ollama";

async function generate(prompt) {
  const response = await fetch(`${BASE_URL}/api/v1/generate`, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${API_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: "llama3.2",
      prompt: prompt,
      temperature: 0.7,
      num_predict: 100
    })
  });

  const data = await response.json();
  return data.data.response;
}

generate("What is machine learning?").then(console.log);
```

### Streaming Example (Python)

```python
import requests
import json

API_KEY = "sk-your-api-key"
BASE_URL = "https://elevatediq.ai/ollama"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

response = requests.post(
    f"{BASE_URL}/api/v1/generate",
    headers=headers,
    json={
        "model": "llama3.2",
        "prompt": "Write a short story about AI",
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        data = json.loads(line.decode().replace("data: ", ""))
        if not data.get("done"):
            print(data["response"], end="", flush=True)
```

---

## Support

- **Documentation**: https://github.com/kushin77/ollama/docs
- **Issues**: https://github.com/kushin77/ollama/issues
- **Email**: ai-infrastructure@elevatediq.ai

---

**Last Updated**: January 18, 2026
**Version**: 0.1.0
