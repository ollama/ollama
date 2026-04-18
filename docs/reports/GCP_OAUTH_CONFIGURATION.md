# ✅ GCP OAuth 2.0 Configuration - Updated

**Date**: January 13, 2026
**Status**: Configuration Updated

---

## GCP Project Credentials

| Setting | Value |
|---------|-------|
| **GCP Project ID** | `project-131055855980` |
| **OAuth Client ID** | `131055855980-6e0t5fjcnq4akk9rfgjsea801ceu8lcd.apps.googleusercontent.com` |
| **Email** | `akushnir@bioenergystrategies.com` |
| **Service Account** | `ollama-service@project-131055855980.iam.gserviceaccount.com` |

---

## Configuration Files Updated

### 1. `ollama/config.py` - Settings Model

Added GCP OAuth configuration fields:

```python
# GCP OAuth 2.0 Configuration
gcp_oauth_client_id: str = Field(
    default="131055855980-6e0t5fjcnq4akk9rfgjsea801ceu8lcd.apps.googleusercontent.com",
    description="GCP OAuth 2.0 Client ID for authentication"
)
gcp_project_id: str = Field(
    default="project-131055855980",
    description="GCP Project ID"
)
gcp_service_account_email: str = Field(
    default="ollama-service@project-131055855980.iam.gserviceaccount.com",
    description="GCP Service Account email"
)
```

Also updated Firebase project ID:
```python
firebase_project_id: str = Field(
    default="project-131055855980",  # Updated from "ollama-elite-platform"
    description="Firebase/GCP project ID"
)
root_admin_email: str = Field(
    default="akushnir@bioenergystrategies.com",  # Updated from "admin@elevatediq.ai"
    description="Root admin email (has all permissions)"
)
```

### 2. `.env` - Environment Variables

Added GCP OAuth environment variables:

```bash
# Firebase OAuth Configuration
FIREBASE_ENABLED=false
FIREBASE_PROJECT_ID=project-131055855980
ROOT_ADMIN_EMAIL=akushnir@bioenergystrategies.com
FIREBASE_CREDENTIALS_PATH=/secrets/firebase-service-account.json

# GCP OAuth 2.0 Configuration
GCP_OAUTH_CLIENT_ID=131055855980-6e0t5fjcnq4akk9rfgjsea801ceu8lcd.apps.googleusercontent.com
GCP_PROJECT_ID=project-131055855980
GCP_SERVICE_ACCOUNT_EMAIL=ollama-service@project-131055855980.iam.gserviceaccount.com
```

---

## How to Access Configuration

### In Python Code

```python
from ollama.config import get_settings

settings = get_settings()

# Access OAuth settings
print(settings.gcp_oauth_client_id)
print(settings.gcp_project_id)
print(settings.firebase_project_id)
```

### In Environment

```bash
# View settings
echo $GCP_OAUTH_CLIENT_ID
echo $GCP_PROJECT_ID
echo $FIREBASE_PROJECT_ID
```

---

## Next Steps

### 1. Store Firebase Service Account (Required for OAuth)

```bash
# Download from GCP Console:
# https://console.cloud.google.com/iam-admin/serviceaccounts

# Store locally:
mkdir -p /home/akushnir/ollama/secrets
cp firebase-service-account.json /home/akushnir/ollama/secrets/
chmod 600 /home/akushnir/ollama/secrets/firebase-service-account.json
```

### 2. Enable OAuth in Application

```bash
# Update .env
FIREBASE_ENABLED=true

# Restart server
pkill -f uvicorn
cd /home/akushnir/ollama && source venv/bin/activate && \
  uvicorn ollama.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Configure OAuth Redirect URIs (In GCP Console)

Add to Authorized redirect URIs:
- `https://elevatediq.ai/oauth/callback`
- `http://localhost:8000/oauth/callback` (development)
- `http://localhost:3000/oauth/callback` (frontend dev)

### 4. Test OAuth Integration

```bash
# Without token (should fail)
curl https://elevatediq.ai/ollama/api/v1/health
# 401 Unauthorized

# With valid token (should succeed)
TOKEN="<firebase-jwt-token>"
curl -H "Authorization: Bearer $TOKEN" \
  https://elevatediq.ai/ollama/api/v1/health
# 200 OK
```

---

## Integration with Firebase

### Firebase Project Link

- **Console**: https://console.firebase.google.com/project/project-131055855980
- **Project ID**: `project-131055855980`

### Service Account Setup

1. Go to GCP Console
2. Select project `project-131055855980`
3. Navigation menu → Service Accounts
4. Create key for `ollama-service` account
5. Download JSON key file
6. Store at `/secrets/firebase-service-account.json`

### OAuth 2.0 Client Configuration

1. Go to GCP Console
2. APIs & Services → OAuth 2.0 Client IDs
3. Configure redirect URIs
4. Add scopes needed by Gov-AI-Scout client

---

## Integration with Gov-AI-Scout

The OAuth Client ID can be used by Gov-AI-Scout to authenticate with Ollama:

```typescript
// Gov-AI-Scout frontend
import { initializeApp } from 'firebase/app';
import { getAuth, signInWithPopup, GoogleAuthProvider } from 'firebase/auth';

const firebaseConfig = {
  projectId: 'project-131055855980',
  clientId: '131055855980-6e0t5fjcnq4akk9rfgjsea801ceu8lcd.apps.googleusercontent.com',
  // ... other config
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const provider = new GoogleAuthProvider();

// Sign in with Google
const result = await signInWithPopup(auth, provider);
const token = await result.user.getIdToken();

// Use token to call Ollama API
const response = await fetch('https://elevatediq.ai/ollama/api/v1/health', {
  headers: {
    'Authorization': `Bearer ${token}`
  }
});
```

---

## Configuration Reference

### Production Settings

For production deployment:

```bash
# GCP Load Balancer Configuration
OLLAMA_PUBLIC_URL=https://elevatediq.ai/ollama
GCP_PROJECT_ID=project-131055855980

# Firebase OAuth Production
FIREBASE_ENABLED=true
FIREBASE_PROJECT_ID=project-131055855980

# CORS (only GCP LB and Gov-AI-Scout)
CORS_ORIGINS=["https://elevatediq.ai","https://gov-ai-scout.com"]
```

### Development Settings

For development:

```bash
# Local development
OLLAMA_PUBLIC_URL=http://localhost:8000
FIREBASE_ENABLED=false  # Enable after service account setup

# Allow localhost + Gov-AI-Scout dev
CORS_ORIGINS=["http://localhost:3000","http://localhost:8000","http://localhost:8080"]
```

---

## Troubleshooting

### Token Verification Fails

**Issue**: `401 Unauthorized - Invalid token`

**Solution**:
1. Verify Firebase service account has correct permissions
2. Check token is from same GCP project
3. Ensure FIREBASE_CREDENTIALS_PATH points to correct file

### OAuth Client Not Found

**Issue**: `404 - OAuth client configuration not found`

**Solution**:
1. Verify GCP_OAUTH_CLIENT_ID matches OAuth 2.0 client in GCP Console
2. Check project ID: `project-131055855980`
3. Confirm OAuth redirect URIs are configured

### CORS Errors

**Issue**: `CORS policy: No 'Access-Control-Allow-Origin' header`

**Solution**:
1. Add Gov-AI-Scout domain to CORS_ORIGINS
2. Set `cors_allow_credentials: true` in config
3. Verify GCP LB CORS policy matches

---

## Security Notes

✅ **Never commit service account JSON** - Use secrets management
✅ **Rotate OAuth Client ID** - Regularly update credentials
✅ **Monitor token usage** - Check GCP audit logs
✅ **Restrict scopes** - Only request needed permissions
✅ **Use HTTPS** - All OAuth flows must use TLS 1.3+

---

## Additional Resources

- [GCP OAuth 2.0 Setup](https://cloud.google.com/docs/authentication)
- [Firebase Admin SDK](https://firebase.google.com/docs/admin/setup)
- [Gov-AI-Scout OAuth Integration](https://github.com/kushin77/Gov-AI-Scout)

---

**Status**: ✅ OAuth Configuration Complete and Ready for Activation
