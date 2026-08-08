# GitHub OAuth Integration - Implementation Guide

## Overview

Complete GitHub OAuth integration has been implemented, allowing users to:
- Connect their GitHub account via OAuth
- Browse their repositories
- View repository files and directories
- Access file contents directly from the chat interface

## What Was Implemented

### Backend (FastAPI)

#### New Endpoints

**OAuth Flow:**
- `GET /auth/github/login` - Initiate GitHub OAuth flow
- `GET /auth/github/callback` - Handle OAuth callback
- `GET /auth/github/status` - Check connection status
- `DELETE /auth/github/disconnect` - Disconnect GitHub account

**Repository Management:**
- `GET /github/repos` - List user's repositories
- `GET /github/repos/{owner}/{repo}` - Get repository details
- `GET /github/repos/{owner}/{repo}/branches` - List branches
- `GET /github/repos/{owner}/{repo}/contents/{path}` - Browse repository contents
- `GET /github/repos/{owner}/{repo}/file/{path}` - Get raw file content

#### Security Features

1. **State Validation:** CSRF protection with secure state tokens (10-minute expiry)
2. **Token Storage:** User tokens stored in Redis, never exposed to frontend
3. **User Isolation:** Each user's GitHub token linked to their API key
4. **Constant-time Comparison:** API key validation uses `secrets.compare_digest`

### Frontend (React + TypeScript)

#### New Components

1. **GitHubBrowser** (`src/components/GitHubBrowser.tsx`)
   - Connection status display
   - Repository list with search
   - File browser with navigation
   - File content viewer with syntax highlighting ready

2. **Updated Components:**
   - `Sidebar.tsx` - Added GitHub button
   - `App.tsx` - OAuth callback handler

#### New API Client Methods

```typescript
api.getGithubStatus()        // Check connection
api.connectGithub()          // Start OAuth flow
api.disconnectGithub()       // Remove connection
api.listGithubRepos()        // Get repositories
api.getGithubContents()      // Browse files
api.getGithubFileContent()   // Read file
```

## Setup Instructions

### 1. Register GitHub OAuth App

1. Go to [GitHub Developer Settings](https://github.com/settings/developers)
2. Click **"New OAuth App"**
3. Fill in the details:
   - **Application name:** `Ollama Gateway` (or your preferred name)
   - **Homepage URL:** `http://localhost:8080`
   - **Authorization callback URL:** `http://localhost:8080/api/auth/github/callback`
4. Click **"Register application"**
5. Copy the **Client ID** and **Client Secret**

### 2. Configure via Admin UI (Recommended)

1. Start the services:
   ```bash
   docker-compose up -d --build
   ```

2. Navigate to the admin panel: `http://localhost:8080/admin`

3. Login with your `SUPER_ADMIN_PASSWORD` from `.env`

4. In the **GitHub OAuth Settings** section:
   - Paste your **Client ID**
   - Paste your **Client Secret**
   - Verify the **Redirect URI** is: `http://localhost:8080/api/auth/github/callback`
   - Click **Save GitHub Settings**

### 3. Alternative: Configure via .env

Add to your `.env` file:

```bash
GITHUB_CLIENT_ID=your_client_id_here
GITHUB_CLIENT_SECRET=your_client_secret_here
GITHUB_REDIRECT_URI=http://localhost:8080/api/auth/github/callback
```

Then restart the services:
```bash
docker-compose restart api
```

## Usage Flow

### For End Users

1. **Login** to the chat interface with your API key

2. **Click the GitHub button** in the sidebar (bottom, above Logout)

3. **Connect GitHub:**
   - Click "Connect GitHub"
   - You'll be redirected to GitHub
   - Authorize the application
   - You'll be redirected back to the chat

4. **Browse Repositories:**
   - See all your repositories
   - Click on any repository to browse files
   - Navigate directories
   - Click on files to view content

5. **Disconnect** (optional):
   - Click "Disconnect" in the GitHub browser header

## Scopes Requested

The OAuth app requests the following scopes:
- `repo` - Full access to public and private repositories
- `read:user` - Read user profile information

## Architecture

```
User Browser
    ↓
[1] Click "Connect GitHub"
    ↓
FastAPI: GET /auth/github/login (with API key)
    ↓
Generate state token → Store in Redis (user_id)
    ↓
Return GitHub authorization URL
    ↓
[2] Browser redirects to GitHub
    ↓
User authorizes app
    ↓
[3] GitHub redirects to: /api/auth/github/callback?code=...&state=...
    ↓
FastAPI: Validate state
    ↓
Exchange code for access_token
    ↓
Store token in Redis: github:token:{user_id}
    ↓
Redirect to: /?github=connected
    ↓
[4] Frontend shows GitHub browser
    ↓
User browses repos
    ↓
FastAPI proxies requests to GitHub API
    ↓
Returns data to frontend
```

## Redis Data Structure

```
# OAuth state (temporary)
oauth:state:{random_token} → user_id (TTL: 600s)

# GitHub access tokens
github:token:{user_id} → {
  "access_token": "gho_...",
  "token_type": "bearer",
  "scope": "repo read:user",
  "saved_at": "2026-06-09T16:00:00"
}
```

## API Examples

### Check GitHub Status

```bash
curl http://localhost:8080/api/auth/github/status \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Response (Connected):**
```json
{
  "connected": true,
  "login": "octocat",
  "avatar_url": "https://avatars.githubusercontent.com/u/..."
}
```

**Response (Not Connected):**
```json
{
  "connected": false
}
```

### List Repositories

```bash
curl http://localhost:8080/api/github/repos \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Response:**
```json
[
  {
    "id": 123456,
    "name": "my-repo",
    "full_name": "octocat/my-repo",
    "description": "My awesome project",
    "private": false,
    "html_url": "https://github.com/octocat/my-repo",
    "default_branch": "main",
    "updated_at": "2026-06-09T10:00:00Z"
  }
]
```

### Get File Content

```bash
curl "http://localhost:8080/api/github/repos/octocat/my-repo/file/README.md" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Response:**
```json
{
  "path": "README.md",
  "name": "README.md",
  "size": 1234,
  "content": "# My Project\n\nThis is my awesome project...",
  "sha": "abc123..."
}
```

## Testing

### 1. Test Backend

```bash
# Check health
curl http://localhost:8080/api/health

# Test OAuth start (requires API key)
curl http://localhost:8080/api/auth/github/login \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### 2. Test Frontend

1. Open browser: `http://localhost:8080`
2. Login with API key
3. Click GitHub button
4. Verify connect button appears
5. Click "Connect GitHub"
6. Authorize on GitHub
7. Verify redirect back to app
8. See repository list

## Production Deployment

### ⚠️ Important Changes for Production

1. **Update Redirect URI:**
   - In GitHub OAuth app settings
   - In admin UI or `.env`:
     ```bash
     GITHUB_REDIRECT_URI=https://yourdomain.com/api/auth/github/callback
     ```

2. **Use HTTPS:**
   - GitHub OAuth requires HTTPS in production
   - Configure SSL/TLS certificates

3. **Secure Admin Password:**
   ```bash
   SUPER_ADMIN_PASSWORD=$(openssl rand -base64 32)
   ```

4. **Configure CORS:**
   Update `api-gateway/main.py`:
   ```python
   allow_origins=["https://yourdomain.com"]
   ```

## Troubleshooting

### "GitHub OAuth is not configured"

**Solution:** Configure Client ID and Secret via admin UI or `.env`

### OAuth callback shows "Invalid or expired OAuth state"

**Possible causes:**
- State token expired (10-minute limit)
- Redis connection lost
- User tried to reuse callback URL

**Solution:** Start OAuth flow again

### "GitHub token expired or invalid"

**Solution:** Disconnect and reconnect GitHub

### Repository list is empty

**Possible causes:**
- User has no repositories
- Token lacks `repo` scope

**Solution:** Check GitHub account or reconnect

### Can't see private repositories

**Cause:** OAuth app needs `repo` scope

**Solution:** Verify scope in OAuth configuration

## File Structure

```
api-gateway/
  main.py (updated)
    - Added OAuth endpoints
    - Added GitHub API proxy
    - Added token management

frontend/src/
  api.ts (updated)
    - GitHub API methods
  
  App.tsx (updated)
    - OAuth callback handler
    - GitHub browser toggle
  
  components/
    GitHubBrowser.tsx (new)
      - Repository browser UI
    Sidebar.tsx (updated)
      - GitHub button
    Sidebar.css (updated)
      - GitHub button styles
  
  styles/
    GitHubBrowser.css (new)
      - Browser component styles
```

## Future Enhancements

- [ ] Add file search functionality
- [ ] Support for GitHub Gists
- [ ] Commit history viewer
- [ ] Pull request integration
- [ ] Code snippet insertion into chat
- [ ] Repository cloning to chat context
- [ ] Webhook support for real-time updates
- [ ] Organization repository support
- [ ] Branch switching in UI
- [ ] Syntax highlighting for code files

## Security Notes

1. **Never expose Client Secret** in frontend or logs
2. **Tokens are stored per user** - no cross-user access
3. **State tokens expire** after 10 minutes
4. **All requests require** valid API key authentication
5. **GitHub tokens never sent** to browser
6. **HTTPS required** in production

## Support

For issues or questions:
1. Check logs: `docker-compose logs api`
2. Verify Redis connection: `docker-compose ps`
3. Test endpoints with curl
4. Review GitHub OAuth app settings

## Success!

Your GitHub OAuth integration is now complete and ready to use. Users can seamlessly connect their GitHub accounts and browse repositories directly from the chat interface.
