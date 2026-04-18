# GitHub Issues Integration - Quick Start Guide

## Setup GitHub API Access for kushin77/ollama

This guide explains how to set up and use GitHub API to update issues securely in the kushin77/ollama repository.

## Step 1: Create Personal Access Token (PAT)

### On GitHub.com:
1. Go to **Settings** → **Developer settings** → **Personal access tokens** → **Tokens (classic)**
2. Click **"Generate new token (classic)"**
3. Name: `ollama-issue-updater-dev`
4. Expiration: 90 days
5. Select scopes:
   - ✅ `repo` - Full control of private repositories
   - ✅ `write:repo_hook` - Write access to hooks
   - ✅ `admin:repo_hook` - Manage repository hooks
   - ✅ `workflow` - Update GitHub Actions workflows

6. Click **"Generate token"** and copy it immediately
7. Store safely: **NEVER share or commit this token**

## Step 2: Local Development Setup

### Option A: Environment Variables (Quick)

```bash
# Add to your shell profile (~/.bashrc, ~/.zshrc, etc.)
export GITHUB_TOKEN="ghp_your_token_here"
export GITHUB_USER="kushin77"
export GITHUB_REPO="ollama"
```

Then reload:
```bash
source ~/.bashrc
```

### Option B: .env File (Recommended for projects)

```bash
# In the ollama repository root
cat > .env << 'EOF'
GITHUB_TOKEN=ghp_your_token_here
GITHUB_USER=kushin77
GITHUB_REPO=ollama
GITHUB_WEBHOOK_SECRET=your_webhook_secret_here
EOF

# Make sure .env is in .gitignore
echo ".env" >> .gitignore
```

Load before running:
```bash
source .env
```

### Option C: Git Credentials Helper (Best for git operations)

```bash
# Configure credential helper to store credentials securely
git config --global credential.helper store

# First time pulling from the repo:
git clone https://github.com/kushin77/ollama.git
# It will prompt for username and token
# Username: your-github-username
# Password: ghp_your_token_here
```

## Step 3: Verify Your Setup

### Check Environment
```bash
# Verify token is set
echo $GITHUB_TOKEN

# Output should be: ghp_...
```

### Test API Access
```bash
# Check rate limits
curl -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/rate_limit

# Or use GitHub CLI
gh auth status
```

## Step 4: Update Issues Programmatically

### Simple CLI Example

```bash
# Using Go - compile the issue updater tool
cd cmd/github-issues
go build -o issue-updater main.go

# List current issues
./issue-updater -owner kushin77 -repo ollama -state open -limit 10

# Or use the existing query tool
go run ./main.go -owner kushin77 -repo ollama
```

### Python Example

```python
#!/usr/bin/env python3
import os
import requests

TOKEN = os.getenv("GITHUB_TOKEN")
OWNER = os.getenv("GITHUB_USER", "kushin77")
REPO = os.getenv("GITHUB_REPO", "ollama")

headers = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github+json"
}

def update_issue(issue_num, **kwargs):
    """Update an issue with given fields"""
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/issues/{issue_num}"
    
    response = requests.patch(url, json=kwargs, headers=headers)
    
    if response.status_code == 200:
        print(f"✓ Issue #{issue_num} updated")
        return response.json()
    else:
        print(f"✗ Failed: {response.status_code} - {response.text}")
        return None

# Examples:

# Close an issue
update_issue(1234, state="closed")

# Add labels
update_issue(1234, labels=["bug", "critical"])

# Assign users
update_issue(1234, assignees=["kushin77", "another-user"])

# Add comment
url = f"https://api.github.com/repos/{OWNER}/{REPO}/issues/1234/comments"
requests.post(url, 
    json={"body": "Fixed in v0.2.0"},
    headers=headers)
```

### Go Example

```go
package main

import (
    "context"
    "fmt"
    "os"
    
    "github.com/google/go-github/v60/github"
    "golang.org/x/oauth2"
)

func main() {
    // Get credentials
    token := os.Getenv("GITHUB_TOKEN")
    owner := os.Getenv("GITHUB_USER")
    repo := os.Getenv("GITHUB_REPO")
    
    if token == "" {
        fmt.Println("GITHUB_TOKEN not set")
        return
    }
    
    // Authenticate
    ctx := context.Background()
    ts := oauth2.StaticTokenSource(&oauth2.Token{AccessToken: token})
    tc := oauth2.NewClient(ctx, ts)
    client := github.NewClient(tc)
    
    // Update issue
    update := &github.IssueRequest{
        State:     github.String("closed"),
        Labels:    []string{"bug", "fixed"},
    }
    
    issue, _, err := client.Issues.Edit(ctx, owner, repo, 1234, update)
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }
    
    fmt.Printf("Issue #%d updated: %s\n", issue.GetNumber(), issue.GetState())
}
```

### Bash Script Example

```bash
#!/bin/bash
# update-issue.sh - Update GitHub issue from command line

GITHUB_TOKEN="${GITHUB_TOKEN}"
GITHUB_USER="${GITHUB_USER:-kushin77}"
GITHUB_REPO="${GITHUB_REPO:-ollama}"

if [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: GITHUB_TOKEN not set"
    exit 1
fi

ISSUE_NUM=$1
STATE=$2

if [ -z "$ISSUE_NUM" ] || [ -z "$STATE" ]; then
    echo "Usage: $0 <issue_number> <state>"
    echo "Example: $0 1234 closed"
    exit 1
fi

curl -X PATCH \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/repos/$GITHUB_USER/$GITHUB_REPO/issues/$ISSUE_NUM \
  -d "{\"state\":\"$STATE\"}"

echo ""
echo "✓ Issue #$ISSUE_NUM state updated to: $STATE"
```

Usage:
```bash
chmod +x update-issue.sh
./update-issue.sh 1234 closed
```

## Step 5: CI/CD Integration (GitHub Actions)

### Workflow Example

```yaml
# .github/workflows/update-issues.yml
name: Auto-Update Issues

on:
  issues:
    types: [opened]
  workflow_dispatch:
    inputs:
      issue_number:
        description: 'Issue number to update'
        required: true

jobs:
  process-issue:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Go
        uses: actions/setup-go@v4
        with:
          go-version: '1.24'
      
      - name: Auto-label new issues
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GITHUB_REPO: kushin77/ollama
        run: |
          # Your automation script here
          echo "Processing issue: ${{ github.event.issue.number }}"
```

## Security Best Practices

### ✅ DO

- [ ] Store GITHUB_TOKEN in environment variables
- [ ] Use .gitignore for .env files
- [ ] Rotate tokens every 90 days
- [ ] Use GitHub Repo Secrets in CI/CD
- [ ] Verify webhook signatures
- [ ] Use minimal required scopes
- [ ] Log operations (not tokens)

### ❌ DON'T

- ❌ Commit tokens to version control
- ❌ Hardcode PAT in scripts
- ❌ Share tokens via email/chat
- ❌ Use expired tokens
- ❌ Enable unnecessary scopes
- ❌ Log full tokens
- ❌ Trust unverified webhooks

## Troubleshooting

### "Bad credentials" Error
```bash
# Verify token format
echo $GITHUB_TOKEN | grep "^ghp_"

# Check token hasn't expired
# Go to Settings → Personal Access Tokens and verify expiration date

# Regenerate if needed:
# 1. Go to GitHub Settings
# 2. Delete old token
# 3. Create new one
# 4. Update GITHUB_TOKEN
```

### Rate Limit Exceeded
```bash
# Check your limit
curl -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/rate_limit | jq '.rate_limit'

# Rate limits:
# - 5,000 requests/hour (authenticated)
# - 60 requests/hour (unauthenticated)

# Solution: Implement backoff/rate limiting
# See: GITHUB_API_SECRETS_POLICY.md
```

### Webhook Not Working
1. Verify webhook URL is publicly accessible (HTTPS)
2. Check webhook secret matches
3. Verify GitHub IP is whitelisted in firewall
4. Review recent deliveries in webhook settings
5. Check application logs

## Advanced Usage

### Batch Update Issues

```go
package main

import (
    "context"
    "fmt"
    "os"
    "time"
    
    "github.com/google/go-github/v60/github"
    "golang.org/x/oauth2"
)

func batchUpdateIssues(issueNums []int, newLabels []string) {
    ctx := context.Background()
    ts := oauth2.StaticTokenSource(&oauth2.Token{
        AccessToken: os.Getenv("GITHUB_TOKEN"),
    })
    client := github.NewClient(oauth2.NewClient(ctx, ts))
    
    // Rate limit: ~4 requests/second to stay under 5000/hour
    ticker := time.NewTicker(250 * time.Millisecond)
    defer ticker.Stop()
    
    for _, num := range issueNums {
        <-ticker.C
        
        update := &github.IssueRequest{Labels: newLabels}
        _, _, err := client.Issues.Edit(ctx, "kushin77", "ollama", num, update)
        
        if err != nil {
            fmt.Printf("Error updating #%d: %v\n", num, err)
            continue
        }
        fmt.Printf("✓ Updated issue #%d\n", num)
    }
}

func main() {
    batchUpdateIssues([]int{1, 2, 3, 4, 5}, []string{"automated"})
}
```

## Additional Resources

- [GitHub API Documentation](https://docs.github.com/en/rest)
- [Creating Personal Access Tokens](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)
- [go-github Library](https://github.com/google/go-github)
- [GitHub Webhooks](https://docs.github.com/en/developers/webhooks-and-events/webhooks)
- [Rate Limiting](https://docs.github.com/en/rest/overview/rate-limits-for-the-rest-api)

## Security Policy

See [GITHUB_API_SECRETS_POLICY.md](./GITHUB_API_SECRETS_POLICY.md) for:
- Token rotation schedule
- Audit logging
- Emergency procedures
- Compliance checklist

---

**Need help?** Ask in the [Ollama Discord](https://discord.gg/ollama)
