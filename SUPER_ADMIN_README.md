# Super Admin Implementation

A complete super-admin interface has been added to manage GitHub OAuth settings and API keys.

## 🎯 Features

- **Admin Authentication**: Secure login with password-based authentication
- **GitHub OAuth Configuration**: Set up GitHub Client ID, Secret, and Redirect URI via UI
- **API Key Management**: Generate secure API keys and revoke them as needed
- **Redis-based Config**: All settings stored in Redis with automatic .env seeding

## 🚀 Quick Start

### 1. Update Environment Variables

Edit `.env` and set a strong admin password:

```bash
SUPER_ADMIN_PASSWORD=your-very-strong-password-here
```

**⚠️ IMPORTANT**: Change the default password before deploying!

### 2. Start the Services

```bash
docker-compose up -d --build
```

### 3. Access Admin Interface

Navigate to: **http://localhost:8080/admin**

Login with your `SUPER_ADMIN_PASSWORD`

### 4. Configure GitHub OAuth (Optional)

1. Register a GitHub OAuth App at https://github.com/settings/developers
2. Set **Authorization callback URL** to: `http://localhost:8080/api/auth/github/callback`
3. Copy the **Client ID** and **Client Secret**
4. In the admin UI, paste these values and save

### 5. Generate API Keys

1. Click **"+ Generate New API Key"**
2. Copy the generated key immediately (it's only shown once!)
3. Share with users who need access to the chat interface

## 📋 Architecture

```
┌─────────────────┐
│  /admin         │  Super Admin UI (login + settings)
│  React SPA      │
└────────┬────────┘
         │ POST /api/admin/login (password)
         │ GET/PUT /api/admin/config
         │ POST /api/admin/api-keys/generate
         │
┌────────▼────────┐
│  FastAPI        │
│  API Gateway    │
├─────────────────┤
│ Redis Config    │  ← Stores: API_KEYS, GITHUB_CLIENT_ID, etc.
└─────────────────┘
```

## 🔒 Security Features

- **Separate Authentication**: Admin uses different auth than regular API keys
- **Session-based**: Admin tokens expire after 1 hour
- **Secrets Hidden**: GitHub Client Secret only writable, never returned in API
- **Masked Keys**: API keys displayed with last 4 chars only
- **One-time Display**: Generated keys shown only once

## 📁 Files Added/Modified

### Backend (`api-gateway/`)
- `main.py`: Added Redis config store, admin auth, and CRUD endpoints

### Frontend (`frontend/src/`)
- `adminApi.ts`: Admin API client
- `AdminApp.tsx`: Container component
- `AdminLogin.tsx`: Login form
- `AdminSettings.tsx`: Settings management UI
- `Admin.css`: Admin styling
- `main.tsx`: Router to split `/admin` from chat app

### Configuration
- `.env`: Added `SUPER_ADMIN_PASSWORD` and GitHub config placeholders
- `docker-compose.yml`: Pass admin password to API container
- `frontend/Dockerfile`: Added nginx config for SPA routing
- `frontend/nginx.conf`: SPA fallback for `/admin` route

## 🔄 Backward Compatibility

✅ **Existing flows continue to work**:
- API keys in `.env` are automatically seeded to Redis on first startup
- Chat UI (`/`) works exactly as before
- Existing API key authentication unchanged

## 🎨 Admin UI Screenshots

### Login
```
┌─────────────────────────────────┐
│      Super Admin                │
│  Enter admin password to        │
│  manage gateway settings        │
│                                 │
│  [Password: ••••••••••••]       │
│                                 │
│  [     Sign In     ]            │
└─────────────────────────────────┘
```

### Settings Dashboard
```
┌─────────────────────────────────────────────┐
│  Super Admin Settings          [Logout]     │
├─────────────────────────────────────────────┤
│                                             │
│  GitHub OAuth Settings                      │
│  ├─ Client ID: [________________]           │
│  ├─ Client Secret: [•••• (set)] │
│  └─ Redirect URI: [http://...]             │
│     [Save GitHub Settings]                  │
│                                             │
│  API Keys Management                        │
│  [+ Generate New API Key]                   │
│  ┌─────────────────────────────┐            │
│  │ •••••••••••••••••••••••abcd │ [Revoke]   │
│  │ •••••••••••••••••••••••wxyz │ [Revoke]   │
│  └─────────────────────────────┘            │
└─────────────────────────────────────────────┘
```

## 🧪 Testing

### Test Admin Login
```bash
curl -X POST http://localhost:8080/api/admin/login \
  -H "Content-Type: application/json" \
  -d '{"password": "your-password"}'
```

### Generate API Key (with admin token)
```bash
curl -X POST http://localhost:8080/api/admin/api-keys/generate \
  -H "Authorization: Bearer <admin-token>"
```

### Update GitHub Config
```bash
curl -X PUT http://localhost:8080/api/admin/config \
  -H "Authorization: Bearer <admin-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "github_client_id": "your-client-id",
    "github_client_secret": "your-secret"
  }'
```

## 🚨 Production Deployment

1. **Change default password** in `.env`
2. **Use HTTPS** for GitHub OAuth callback
3. **Restrict `/admin`** by IP in nginx (optional)
4. **Rate-limit** `/api/admin/login` to prevent brute force
5. **Monitor** Redis for config changes
6. **Backup** Redis data regularly

## 📝 Notes

- Admin sessions last **1 hour** (stored in Redis)
- Config changes take effect immediately (no restart needed)
- Revoking an API key logs out affected users instantly
- `.env` is only read on first startup for seeding

## 🐛 Troubleshooting

**Can't login to admin?**
- Check `SUPER_ADMIN_PASSWORD` is set in `.env`
- Verify Redis is running: `docker-compose ps`
- Check API logs: `docker-compose logs api`

**API keys not working?**
- Old keys from `.env` are automatically migrated to Redis
- New keys generated via admin UI work immediately
- Check `/api/health` to verify API keys configured

**GitHub OAuth not working?**
- Verify callback URL matches exactly in GitHub app settings
- Client Secret must be saved at least once via admin UI
- Check browser console for CORS errors

## 🎉 Success!

Your super-admin interface is now ready. Access it at `/admin` to configure GitHub integration and manage API keys!
