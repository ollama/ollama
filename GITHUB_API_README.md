# GitHub API Integration Guide - kushin77/ollama

Complete guide to securely integrating with GitHub API for updating issues, pull requests, and automating workflows in the kushin77/ollama repository.

## Quick Start (2 minutes)

```bash
# 1. Run interactive setup
./github-api-helper.sh setup

# 2. Verify your setup
./github-api-helper.sh verify

# 3. Check rate limits
./github-api-helper.sh status

# 4. Update your first issue
./github-api-helper.sh update-issue 1234 closed
```

## Documentation Files

| File | Purpose | Audience |
|------|---------|----------|
| **GITHUB_API_QUICKSTART.md** | Step-by-step setup guide | Developers starting out |
| **GITHUB_API_SECRETS_POLICY.md** | Security best practices | All developers & DevOps |
| **.env.example** | Environment variable template | Reference |
| **github-api-helper.sh** | Interactive CLI tool | Daily use |
| **.github/instructions/github-api-secrets.instructions.md** | Code examples & patterns | When implementing |

## Overview

This integration enables:
- ✅ Programmatic issue updates (state, labels, assignees)
- ✅ Automated commenting on issues/PRs
- ✅ Batch operations with rate limiting
- ✅ Secure credential management
- ✅ CI/CD workflow automation
- ✅ Webhook event handling
- ✅ Audit logging and monitoring

## 30-Second Setup

### Step 1: Get GitHub Token
Go to: https://github.com/settings/tokens → Generate new token (classic)

Select scopes:
- ✅ `repo`
- ✅ `write:repo_hook`
- ✅ `admin:repo_hook`
- ✅ `workflow`

### Step 2: Run Setup Script
```bash
./github-api-helper.sh setup
# Follow prompts to enter token and repository info
```

### Step 3: Verify
```bash
./github-api-helper.sh verify
# Should show: ✓ Token is valid!
```

## Common Tasks

### Update Issue State
```bash
# Close issue
./github-api-helper.sh update-issue 1234 closed

# Reopen issue
./github-api-helper.sh update-issue 1234 open
```

### Add Labels
```bash
./github-api-helper.sh label 1234 bug critical urgent
```

### Add Comment
```bash
./github-api-helper.sh comment 1234 "Fixed in PR #5678"
```

### Check API Rate Limits
```bash
./github-api-helper.sh status
```

### List Issues
```bash
./github-api-helper.sh list-issues
./github-api-helper.sh list-issues bug      # Filter by label
```

## Code Examples

### Python
```python
import os
import requests

token = os.getenv("GITHUB_TOKEN")
headers = {"Authorization": f"token {token}"}

# Update issue state
response = requests.patch(
    f"https://api.github.com/repos/kushin77/ollama/issues/1234",
    json={"state": "closed"},
    headers=headers
)
print(f"✓ Issue closed" if response.ok else f"✗ Error: {response.text}")
```

### Go
```go
package main

import (
    "context"
    "os"
    "github.com/google/go-github/v60/github"
    "golang.org/x/oauth2"
)

func main() {
    ctx := context.Background()
    ts := oauth2.StaticTokenSource(&oauth2.Token{
        AccessToken: os.Getenv("GITHUB_TOKEN"),
    })
    client := github.NewClient(oauth2.NewClient(ctx, ts))
    
    _, _, err := client.Issues.Edit(ctx, "kushin77", "ollama", 1234,
        &github.IssueRequest{State: github.String("closed")})
}
```

### Bash
```bash
#!/bin/bash
token=$GITHUB_TOKEN
issue=1234
state=closed

curl -X PATCH \
  -H "Authorization: token $token" \
  https://api.github.com/repos/kushin77/ollama/issues/$issue \
  -d "{\"state\":\"$state\"}"
```

## Environment Setup

### Option 1: .env File (Recommended)
```bash
# Copy template
cp .env.example .env

# Edit .env with your values
nano .env

# Load before running commands
source .env
```

### Option 2: Export Environment Variables
```bash
export GITHUB_TOKEN="ghp_your_token"
export GITHUB_USER="kushin77"
export GITHUB_REPO="ollama"
```

### Option 3: GitHub CLI
```bash
gh auth login
# Follow prompts to authenticate
# GitHub CLI will handle token management
```

## GitHub Actions Workflow Example

```yaml
# .github/workflows/auto-label.yml
name: Auto-Label Issues

on:
  issues:
    types: [opened]

jobs:
  label:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Label new issues
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          LABELS="triage"
          curl -X PATCH \
            -H "Authorization: token $GITHUB_TOKEN" \
            https://api.github.com/repos/kushin77/ollama/issues/${{ github.event.issue.number }} \
            -d "{\"labels\":[\"$LABELS\"]}"
```

## Security Best Practices

### ✅ DO
- Store tokens in environment variables (`GITHUB_TOKEN`)
- Use `.gitignore` for `.env` files
- Rotate tokens every 90 days
- Use GitHub Secrets in CI/CD workflows
- Verify webhook signatures (HMAC-SHA256)
- Use minimal token scopes
- Log operations (not token values)

### ❌ DON'T
- Commit `.env` files to git
- Hardcode PAT tokens in scripts
- Share tokens via email or chat
- Use overly broad scopes
- Enable unnecessary GitHub App permissions
- Log full token values
- Trust unverified webhooks

## Troubleshooting

### "Bad credentials" Error
```bash
# 1. Verify token format
echo $GITHUB_TOKEN | grep "^ghp_"

# 2. Check token hasn't expired
# Go to: https://github.com/settings/tokens

# 3. Regenerate if needed
# Delete old token, create new one
```

### Rate Limit Exceeded
```bash
# Check your limit
./github-api-helper.sh status

# GitHub API limits:
# - 5,000 requests/hour (authenticated)
# - 60 requests/hour (public)
```

### Webhook Not Triggering
1. Verify webhook URL is HTTPS and publicly accessible
2. Check webhook secret is configured correctly
3. Review recent deliveries in webhook settings
4. Check application logs

## API Rate Limits

| Metric | Authenticated | Public |
|--------|---------------|--------|
| Requests per hour | 5,000 | 60 |
| Requests per second | ~1.4 | 0.017 |
| Safe rate | 0.8/sec | N/A |

**Implementation tip**: Use 250ms delay between requests (4 req/sec) to stay well under limits.

## Security Policy

See **GITHUB_API_SECRETS_POLICY.md** for comprehensive details on:
- Token lifecycle management
- Git credentials setup
- CI/CD secret management
- Webhook security
- Audit logging
- Emergency procedures
- Compliance checklist

## API Documentation

- [GitHub REST API](https://docs.github.com/en/rest)
- [Personal Access Tokens](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)
- [go-github Library](https://github.com/google/go-github)
- [Webhooks](https://docs.github.com/en/developers/webhooks-and-events/webhooks)
- [Rate Limiting](https://docs.github.com/en/rest/overview/rate-limits-for-the-rest-api)

## File Reference

```
Repository Root:
├── GITHUB_API_QUICKSTART.md           ← Start here for setup
├── GITHUB_API_SECRETS_POLICY.md       ← Security & best practices
├── GITHUB_API_README.md               ← This file
├── github-api-helper.sh               ← Interactive CLI tool
└── .env.example                        ← Environment template

.github/
└── instructions/
    └── github-api-secrets.instructions.md  ← Code examples & patterns

.github/
└── copilot-instructions.md            ← Copilot config (includes API guidance)
```

## Support & Feedback

- **Questions?** Ask in [Ollama Discord](https://discord.gg/ollama)
- **Issues?** Open a GitHub issue or discussion
- **Security concerns?** See [SECURITY.md](./SECURITY.md)

## Key Takeaways

1. **Use the helper script** for quick operations: `./github-api-helper.sh`
2. **Secure your token** in `.env` (gitignored) or GitHub Secrets
3. **Rotate tokens** every 90 days
4. **Monitor rate limits** to avoid throttling
5. **Verify webhooks** with HMAC-SHA256 signatures
6. **Log operations** for audit trails (not tokens!)
7. **Read the policy** at `GITHUB_API_SECRETS_POLICY.md`

---

**Last Updated**: April 17, 2026  
**Repository**: kushin77/ollama  
**Status**: ✅ Fully configured and ready to use
