# Gov-AI-Scout Integration Guide

**Integration Partner**: Gov-AI-Scout
**Ollama Endpoint**: https://elevatediq.ai/ollama
**Integration Date**: January 13, 2026
**Status**: Production Ready

---

## Overview

Gov-AI-Scout integrates with Ollama to enable secure, OAuth-authenticated AI inference for government applications. This guide covers authentication, API usage, and integration patterns.

---

## Authentication Setup

### 1. Get OAuth Credentials

**Contact**: akushnir@bioenergystrategies.com
**GCP Project**: project-131055855980

Gov-AI-Scout receives:
- **OAuth Client ID**: 131055855980-6e0t5fjcnq4akk9rfgjsea801ceu8lcd.apps.googleusercontent.com
- **API Base URL**: https://elevatediq.ai/ollama
- **Auth Method**: Firebase JWT Bearer Token

### 2. Obtain Access Token

#### Option A: Service Account (Recommended for Backend)

```bash
# Using gcloud CLI
gcloud auth application-default login

# Get access token
ACCESS_TOKEN=$(gcloud auth print-access-token)

# Use in requests
curl -H "Authorization: Bearer $ACCESS_TOKEN" \
  https://elevatediq.ai/ollama/api/v1/health
```

#### Option B: Firebase Authentication (For Client Apps)

```javascript
// JavaScript/React integration
import { initializeApp } from "firebase/app";
import { getAuth, signInWithCustomToken } from "firebase/auth";

const firebaseConfig = {
  projectId: "project-131055855980",
  apiKey: "YOUR_FIREBASE_API_KEY",
  authDomain: "project-131055855980.firebaseapp.com",
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

async function authenticateWithOllama() {
  const response = await fetch("https://your-backend.com/get-firebase-token", {
    credentials: "include"
  });

  const { customToken } = await response.json();

  const userCredential = await signInWithCustomToken(auth, customToken);
  const idToken = await userCredential.user.getIdToken();

  return idToken;
}

// Use token in Ollama requests
const token = await authenticateWithOllama();
const response = await fetch("https://elevatediq.ai/ollama/api/v1/generate", {
  method: "POST",
  headers: {
    "Authorization": `Bearer ${token}`,
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    prompt: "Analyze this policy document...",
    model: "llama3.2"
  })
});
```

#### Option C: OAuth 2.0 Authorization Code Flow

```python
# Python integration
from google.oauth2 import service_account
import google.auth.transport.requests

# Create credentials from service account JSON
credentials = service_account.Credentials.from_service_account_file(
    '/path/to/service-account.json',
    scopes=['https://www.googleapis.com/auth/firebase']
)

# Refresh to get access token
request = google.auth.transport.requests.Request()
credentials.refresh(request)

access_token = credentials.token

# Use in Ollama API call
import requests

response = requests.post(
    "https://elevatediq.ai/ollama/api/v1/generate",
    headers={"Authorization": f"Bearer {access_token}"},
    json={
        "prompt": "Policy analysis task",
        "model": "llama3.2"
    }
)
```

---

## API Endpoints

### Health Check (Public - No Auth)

```http
GET https://elevatediq.ai/ollama/health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "qdrant": "healthy"
  }
}
```

### Protected Health Check (OAuth Required)

```http
GET https://elevatediq.ai/ollama/api/v1/health
Authorization: Bearer <FIREBASE_JWT_TOKEN>
```

**Response**:
```json
{
  "status": "healthy",
  "authenticated_user": "gov-ai-scout-service-account",
  "role": "editor",
  "request_id": "req_abc123"
}
```

### Text Generation (OAuth Required)

```http
POST https://elevatediq.ai/ollama/api/v1/generate
Authorization: Bearer <FIREBASE_JWT_TOKEN>
Content-Type: application/json

{
  "model": "llama3.2",
  "prompt": "Summarize the policy implications of climate change",
  "stream": false,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "max_tokens": 500
}
```

**Response**:
```json
{
  "model": "llama3.2",
  "text": "Policy implications of climate change include...",
  "tokens_generated": 142,
  "inference_time_ms": 1250,
  "stop_reason": "length",
  "metadata": {
    "request_id": "req_xyz789",
    "timestamp": "2026-01-13T10:30:00Z"
  }
}
```

### Streaming Generation (OAuth Required)

```http
POST https://elevatediq.ai/ollama/api/v1/generate
Authorization: Bearer <FIREBASE_JWT_TOKEN>
Content-Type: application/json

{
  "model": "llama3.2",
  "prompt": "Generate a policy brief on renewable energy",
  "stream": true
}
```

**Response** (Server-Sent Events):
```
data: {"token": "Policy", "metadata": {"tokens_generated": 1}}
data: {"token": " ", "metadata": {"tokens_generated": 2}}
data: {"token": "changes", "metadata": {"tokens_generated": 3}}
...
data: [DONE]
```

### Embeddings Generation (OAuth Required)

```http
POST https://elevatediq.ai/ollama/api/v1/embeddings
Authorization: Bearer <FIREBASE_JWT_TOKEN>
Content-Type: application/json

{
  "model": "nomic-embed-text",
  "text": "Environmental policy in the United States"
}
```

**Response**:
```json
{
  "embedding": [0.123, -0.456, 0.789, ...],
  "model": "nomic-embed-text",
  "dimensions": 768,
  "metadata": {
    "request_id": "req_emb456",
    "timestamp": "2026-01-13T10:31:00Z"
  }
}
```

---

## Integration Examples

### Example 1: Policy Document Analysis

```python
# govai_scout_integration.py
import requests
import json
from datetime import datetime

class OllamaClient:
    def __init__(self, token: str, base_url: str = "https://elevatediq.ai/ollama"):
        self.token = token
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

    def analyze_policy(self, policy_text: str) -> dict:
        """Analyze a policy document using Ollama."""
        prompt = f"""Analyze the following policy document and provide:
1. Main objectives
2. Key stakeholders affected
3. Potential challenges
4. Implementation timeline
5. Success metrics

Policy:
{policy_text}

Analysis:"""

        response = requests.post(
            f"{self.base_url}/api/v1/generate",
            headers=self.headers,
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "temperature": 0.3,
                "max_tokens": 1000
            }
        )

        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.status_code} - {response.text}")

        return {
            "timestamp": datetime.now().isoformat(),
            "analysis": response.json()["text"],
            "request_id": response.json()["metadata"]["request_id"]
        }

    def compare_policies(self, policy1: str, policy2: str) -> dict:
        """Compare two policy documents."""
        prompt = f"""Compare the following two policies:

Policy 1:
{policy1}

Policy 2:
{policy2}

Comparison (similarities, differences, implications):"""

        response = requests.post(
            f"{self.base_url}/api/v1/generate",
            headers=self.headers,
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "temperature": 0.5,
                "max_tokens": 1500
            }
        )

        return response.json()

    def generate_embeddings(self, texts: list) -> dict:
        """Generate embeddings for policy documents."""
        embeddings = []
        for text in texts:
            response = requests.post(
                f"{self.base_url}/api/v1/embeddings",
                headers=self.headers,
                json={
                    "model": "nomic-embed-text",
                    "text": text
                }
            )
            embeddings.append({
                "text": text[:50] + "...",
                "embedding": response.json()["embedding"]
            })

        return {
            "count": len(embeddings),
            "embeddings": embeddings
        }

# Usage
client = OllamaClient(token="YOUR_FIREBASE_TOKEN")

# Analyze a policy
policy_text = "The proposed Energy Efficiency Act aims to..."
analysis = client.analyze_policy(policy_text)
print(f"Analysis: {analysis['analysis']}")
print(f"Request ID: {analysis['request_id']}")
```

### Example 2: Streaming Response Handler

```python
# streaming_example.py
import requests
import json

def stream_policy_generation(token: str, prompt: str) -> None:
    """Stream policy generation responses."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        "https://elevatediq.ai/ollama/api/v1/generate",
        headers=headers,
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": True
        },
        stream=True
    )

    for line in response.iter_lines():
        if line:
            data = json.loads(line[6:])  # Remove "data: " prefix
            if data.get("token"):
                print(data["token"], end="", flush=True)

# Usage
stream_policy_generation(
    token="YOUR_FIREBASE_TOKEN",
    prompt="Generate a comprehensive policy brief on climate change"
)
```

### Example 3: Batch Processing

```python
# batch_processing.py
import asyncio
import aiohttp
import json

async def batch_analyze_documents(token: str, documents: list) -> list:
    """Analyze multiple documents in parallel."""

    async def analyze_one(session, document):
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        async with session.post(
            "https://elevatediq.ai/ollama/api/v1/generate",
            headers=headers,
            json={
                "model": "llama3.2",
                "prompt": f"Summarize: {document['text']}",
                "max_tokens": 200
            }
        ) as resp:
            return await resp.json()

    async with aiohttp.ClientSession() as session:
        tasks = [analyze_one(session, doc) for doc in documents]
        results = await asyncio.gather(*tasks)

    return results

# Usage
documents = [
    {"id": 1, "text": "Policy text 1..."},
    {"id": 2, "text": "Policy text 2..."},
    {"id": 3, "text": "Policy text 3..."}
]

results = asyncio.run(batch_analyze_documents(
    token="YOUR_FIREBASE_TOKEN",
    documents=documents
))

for result in results:
    print(f"Analysis: {result['text'][:100]}...")
```

---

## Rate Limiting

**Default Limits**:
- 100 requests per 60 seconds per OAuth client
- 1000 concurrent requests per service account

**Headers in Response**:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1673606460
```

**Rate Limit Exceeded Response**:
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "retry_after": 60
  }
}
```

---

## Error Handling

### Authentication Errors

```json
{
  "error": {
    "code": "INVALID_TOKEN",
    "message": "Invalid or expired authentication token",
    "status_code": 401
  }
}
```

### Authorization Errors

```json
{
  "error": {
    "code": "FORBIDDEN",
    "message": "User does not have permission to access this resource",
    "status_code": 403
  }
}
```

### Model Errors

```json
{
  "error": {
    "code": "MODEL_NOT_FOUND",
    "message": "Model 'llama3.2' not found",
    "available_models": ["nomic-embed-text", "mistral"],
    "status_code": 404
  }
}
```

### Server Errors

```json
{
  "error": {
    "code": "INFERENCE_TIMEOUT",
    "message": "Inference timed out after 300 seconds",
    "status_code": 504,
    "request_id": "req_timeout123"
  }
}
```

---

## Monitoring Integration

### Logs

All requests are logged with:
- User ID
- Request timestamp
- Model used
- Tokens generated
- Response time
- Error details (if any)

**Access logs**:
```bash
# View latest requests
curl -H "Authorization: Bearer $ADMIN_TOKEN" \
  https://elevatediq.ai/ollama/api/v1/logs?limit=100

# Filter by user
curl -H "Authorization: Bearer $ADMIN_TOKEN" \
  https://elevatediq.ai/ollama/api/v1/logs?user_id=gov-ai-scout
```

### Metrics

Monitor in GCP Cloud Monitoring:
- Request latency (p50, p99)
- Model cache hit rate
- Error rates by type
- Token generation rate

---

## Testing

### Health Check Test

```bash
# Public endpoint (should succeed)
curl https://elevatediq.ai/ollama/health

# Protected endpoint (should fail without token)
curl https://elevatediq.ai/ollama/api/v1/health
# Returns: 401 Unauthorized

# Protected endpoint (should succeed with token)
TOKEN=$(gcloud auth print-identity-token)
curl -H "Authorization: Bearer $TOKEN" \
  https://elevatediq.ai/ollama/api/v1/health
# Returns: 200 OK
```

### Load Testing

```bash
# Install hey
go install github.com/rakyll/hey@latest

# Run load test
TOKEN=$(gcloud auth print-identity-token)
hey -n 1000 -c 50 \
  -m POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3.2","prompt":"Test"}' \
  https://elevatediq.ai/ollama/api/v1/generate
```

---

## Troubleshooting

### Connection Refused

```
Error: Connection refused
```

**Solution**: Verify endpoint is reachable:
```bash
curl -I https://elevatediq.ai/ollama/health
# Should return HTTP 200
```

### Authentication Failed

```
Error: Invalid or expired authentication token
```

**Solution**: Refresh token:
```bash
# Get new token
ACCESS_TOKEN=$(gcloud auth print-identity-token)

# Retry request with new token
curl -H "Authorization: Bearer $ACCESS_TOKEN" \
  https://elevatediq.ai/ollama/api/v1/health
```

### Rate Limit Exceeded

```
Error: Rate limit exceeded. Retry after 60 seconds.
```

**Solution**: Implement exponential backoff:
```python
import time
import requests

def retry_with_backoff(url, headers, data, max_retries=3):
    for attempt in range(max_retries):
        response = requests.post(url, headers=headers, json=data)

        if response.status_code != 429:
            return response

        retry_after = int(response.headers.get('X-RateLimit-Reset', 60))
        wait_time = min(2 ** attempt * retry_after, 300)
        time.sleep(wait_time)

    raise Exception("Max retries exceeded")
```

### Model Not Found

```
Error: Model 'llama3.2' not found
```

**Solution**: List available models:
```bash
TOKEN=$(gcloud auth print-identity-token)
curl -H "Authorization: Bearer $TOKEN" \
  https://elevatediq.ai/ollama/api/v1/models

# Use one of the available models
```

---

## Support

**Contact**: akushnir@bioenergystrategies.com
**Documentation**: https://elevatediq.ai/ollama/docs
**Issues**: Submit via GCP Support Console
**SLA**: 99.9% uptime, <500ms p99 latency

---

**Status**: ✅ Production Ready
**Last Updated**: January 13, 2026
**Next Review**: Q1 2026
