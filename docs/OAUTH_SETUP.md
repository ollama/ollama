# Firebase OAuth Setup Guide

This document explains how to set up and use Firebase OAuth authentication, mirrored from the **Gov-AI-Scout** client for consistency.

## Overview

Ollama uses **Firebase Authentication** to protect sensitive endpoints via OAuth 2.0 JWT tokens. This mirrors the Gov-AI-Scout implementation exactly for seamless client compatibility.

## Architecture

```
┌─────────────────────┐
│   Client            │
│ (Gov-AI-Scout)      │
└─────────┬───────────┘
          │ Authorization: Bearer <JWT>
          │
    ┌─────▼─────────────────────┐
    │  FastAPI + OAuth          │
    │ ├─ GET /health (optional) │
    │ └─ GET /api/v1/health (required)
    │                           │
    └─────┬───────────────────┬─┘
          │                   │
    ┌─────▼──────┐    ┌───────▼────────┐
    │ Firebase   │    │ Internal       │
    │ JWT        │    │ Services       │
    │ Verify     │    │ (DB, Redis)    │
    └────────────┘    └────────────────┘
```

## Setup Steps

### 1. Create Firebase Project

1. Go to [Firebase Console](https://console.firebase.google.com)
2. Click **Create Project**
3. Name: `ollama-elite-platform`
4. Enable Google Analytics (optional)
5. Click **Create Project**

### 2. Enable Authentication

1. In Firebase Console, go to **Authentication** (left panel)
2. Click **Get Started**
3. Enable **Google** provider:
   - Click Google provider
   - Toggle **Enable**
   - Add support email
   - Save
4. (Optional) Enable other providers: Email/Password, GitHub, etc.

### 3. Create Service Account

1. Go to **Project Settings** (gear icon)
2. Click **Service Accounts** tab
3. Click **Generate New Private Key**
4. Save the JSON file securely:
   ```bash
   mkdir -p /home/akushnir/ollama/secrets
   # Place file at: /home/akushnir/ollama/secrets/firebase-service-account.json
   ```

### 4. Configure Environment

**Development (.env):**
```bash
FIREBASE_ENABLED=true
FIREBASE_CREDENTIALS_PATH=/path/to/firebase-service-account.json
FIREBASE_PROJECT_ID=ollama-elite-platform
ROOT_ADMIN_EMAIL=your-admin-email@gmail.com
```

**Production (Cloud Secret Manager):**
```bash
# Store credentials in GCP Secret Manager
gcloud secrets create firebase-service-account --data-file=firebase-service-account.json

# Reference in deployment
FIREBASE_CREDENTIALS_PATH=/run/secrets/firebase-service-account
```

### 5. Install Firebase SDK

```bash
pip install firebase-admin
```

## API Endpoints

### Public Health Check (OAuth Optional)
```http
GET /health HTTP/1.1
Host: api.elevatediq.ai
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-13T18:15:00Z",
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
GET /api/v1/health HTTP/1.1
Host: api.elevatediq.ai
Authorization: Bearer <firebase-jwt-token>
```

**Request Headers Required:**
- `Authorization: Bearer <jwt-token>` - Valid Firebase ID token

**Response (200 OK):**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-13T18:15:00Z",
  "version": "1.0.0",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "qdrant": "healthy"
  }
}
```

**Error Responses:**
- `401 Unauthorized` - Missing, expired, or invalid token
- `403 Forbidden` - Insufficient permissions

## Client Integration (Gov-AI-Scout Pattern)

### JavaScript/TypeScript Example

```typescript
// Initialize Firebase
import { initializeApp } from "firebase/app";
import { getAuth, signInWithPopup, GoogleAuthProvider } from "firebase/auth";

const firebaseConfig = {
  apiKey: "YOUR_API_KEY",
  authDomain: "ollama-elite-platform.firebaseapp.com",
  projectId: "ollama-elite-platform",
  // ... other config
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

// Sign in with Google
const provider = new GoogleAuthProvider();
const result = await signInWithPopup(auth, provider);

// Get JWT token
const token = await result.user.getIdToken();

// Make authenticated request to Ollama
const response = await fetch("https://elevatediq.ai/ollama/api/v1/health", {
  method: "GET",
  headers: {
    "Authorization": `Bearer ${token}`,
    "Content-Type": "application/json",
  },
});
```

### Python Example

```python
import firebase_admin
from firebase_admin import credentials, auth
import requests

# Initialize Firebase
cred = credentials.Certificate("/path/to/firebase-service-account.json")
firebase_admin.initialize_app(cred)

# Create custom token for user
uid = "user-123"
custom_token = auth.create_custom_token(uid)

# Use in API request
headers = {
    "Authorization": f"Bearer {custom_token}",
    "Content-Type": "application/json",
}

response = requests.get(
    "https://elevatediq.ai/ollama/api/v1/health",
    headers=headers,
)
print(response.json())
```

## Mirrored from Gov-AI-Scout

This implementation mirrors the **Gov-AI-Scout** authentication system exactly:

### Similarities
- ✅ Firebase Admin SDK for server-side verification
- ✅ JWT token extraction from `Authorization: Bearer` header
- ✅ Token expiration and revocation handling
- ✅ Role-based access control (RBAC)
- ✅ Root admin email enforcement
- ✅ Security headers (CSP, HSTS, X-Frame-Options)

### Gov-AI-Scout Reference
See the Gov-AI-Scout repository for the original implementation:
- Repo: https://github.com/kushin77/Gov-AI-Scout
- Auth module: `auth.py`
- Middleware: `security.py`

## Protecting Endpoints

### Require Authentication

```python
from fastapi import Depends
from ollama.auth import get_current_user

@app.get("/api/v1/protected")
async def protected_route(user: dict = Depends(get_current_user)):
    """This endpoint requires valid Firebase JWT."""
    return {"message": f"Hello {user['email']}"}
```

### Require Specific Role

```python
from ollama.auth import require_role

@app.get("/api/v1/admin")
async def admin_only(user: dict = Depends(require_role(["admin"]))):
    """This endpoint requires admin role."""
    return {"message": "Admin only"}
```

### Require Root Admin

```python
from ollama.config import get_settings
from ollama.auth import require_root_admin

settings = get_settings()

@app.delete("/api/v1/system/purge")
async def system_operation(user: dict = Depends(require_root_admin(settings.root_admin_email))):
    """This endpoint requires root admin."""
    return {"message": "System purge initiated"}
```

## Token Revocation

Force user re-authentication:

```python
from ollama.auth import revoke_user_tokens

# Revoke all tokens for a user
revoke_user_tokens(uid="user-123")

# User must sign in again to get new tokens
```

## Troubleshooting

### Issue: "Firebase not initialized"
**Solution:** Ensure `FIREBASE_ENABLED=true` and credentials path is correct.

```bash
# Check credentials
ls -la /path/to/firebase-service-account.json

# Verify environment
echo $FIREBASE_CREDENTIALS_PATH
```

### Issue: "Invalid token" / 401 Unauthorized
**Causes:**
- Token expired (valid for ~1 hour)
- Token from different Firebase project
- Incorrect Authorization header format

**Solution:**
```bash
# Correct format
Authorization: Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6IjEyMzQ...

# Get new token
# For clients: Re-authenticate
# For server: Use service account to create custom token
```

### Issue: Token not in database claims
**Cause:** Custom claims not set on Firebase user

**Solution:**
```python
# Set custom claims on user
firebase_auth.set_custom_user_claims(uid, {"roles": ["admin", "user"]})
```

## Security Best Practices

✅ **DO:**
- Use HTTPS/TLS for all API calls
- Store service account JSON securely (GCP Secret Manager)
- Rotate service account keys quarterly
- Use short token expiration (default: 1 hour)
- Revoke tokens for sensitive operations
- Log all authentication attempts
- Use CORS allowlist (never `["*"]`)

❌ **DON'T:**
- Commit service account JSON to git
- Use `FIREBASE_ENABLED=false` in production
- Accept tokens from untrusted sources
- Store tokens in client localStorage without encryption
- Use localhost in production
- Expose Firebase project ID publicly

## Production Deployment

For production deployment via GCP Load Balancer:

1. **Store credentials in Cloud Secret Manager**
   ```bash
   gcloud secrets create firebase-service-account \
     --data-file=firebase-service-account.json
   ```

2. **Configure Cloud Run or GKE**
   ```yaml
   # Cloud Run: Set environment variable
   env:
     - name: FIREBASE_CREDENTIALS_PATH
       valueFrom:
         secretKeyRef:
           name: firebase-service-account
           key: key
   ```

3. **Enable OAuth consent screen in Firebase**
   - Go to Authentication > Settings
   - Configure OAuth consent screen
   - Add authorized redirect URIs

4. **Test via GCP Load Balancer**
   ```bash
   curl -H "Authorization: Bearer <token>" \
     https://elevatediq.ai/ollama/api/v1/health
   ```

## Additional Resources

- [Firebase Admin SDK](https://firebase.google.com/docs/admin/setup)
- [Firebase Authentication](https://firebase.google.com/docs/auth)
- [Gov-AI-Scout Repository](https://github.com/kushin77/Gov-AI-Scout)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
