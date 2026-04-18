# GitHub API Secrets & Git Credentials Policy

**Repository**: kushin77/ollama  
**Last Updated**: April 17, 2026

## Overview

This document establishes the security policy for managing GitHub API tokens, git credentials, PATs, and other sensitive credentials when integrating with GitHub API to update issues, PRs, and automations.

## 1. Secret Management Strategy

### 1.1 Secret Storage Locations

| Secret Type | Storage | Use Case | Security Level |
|------------|---------|----------|-----------------|
| **GITHUB_TOKEN** (PAT) | `$GITHUB_TOKEN` env var | API authentication | 🔴 Critical |
| **Git Credentials** | `~/.git-credentials` | Git clone/push | 🔴 Critical |
| **Webhook Secret** | `$GITHUB_WEBHOOK_SECRET` | Webhook validation | 🔴 Critical |
| **CI/CD Secrets** | GitHub Secrets (Settings) | Actions workflows | 🔴 Critical |
| **Local .env file** | `$(ROOT)/.env` (gitignored) | Development only | 🟡 High |

### 1.2 Secret Lifecycle

```
Create → Store Securely → Use Safely → Rotate → Decommission
  ↓         ↓              ↓           ↓       ↓
[GitHub]  [Env/Secrets]  [Code]    [Quarterly] [Revoke]
```

## 2. Personal Access Token (PAT) Management

### 2.1 Creating PATs

**Location**: GitHub Settings → Developer Settings → Personal Access Tokens → Tokens (classic)

**Required Scopes for kushin77/ollama**:
```
✅ repo               - Full control of private repositories
✅ write:repo_hook   - Write access to hooks
✅ admin:repo_hook   - Manage repository hooks
✅ workflow          - Update GitHub Actions workflows
```

**NOT Required** (minimize scope):
```
❌ delete_repo        - Can delete repositories
❌ admin:org_hook    - Manage organization hooks
❌ admin:gpg_key     - Manage GPG keys
```

### 2.2 Naming Convention

```
ghp_{{ environment }}_{{ purpose }}_{{ timestamp }}

Examples:
- ghp_dev_issue_updater_20260417
- ghp_ci_automation_20260417
- ghp_local_testing_20260417
```

### 2.3 Token Rotation Schedule

- **Initial**: 90 days
- **Quarterly rotation**: Every 3 months
- **On compromise**: Immediately revoke and regenerate
- **On team member departure**: Revoke all personal tokens

### 2.4 Token Revocation

```bash
# List all personal access tokens (requires GitHub CLI)
gh auth token

# Revoke from GitHub Settings → Developer Settings → Personal Access Tokens
# Or via API:
curl -X DELETE \
  -H "Authorization: token CURRENT_TOKEN" \
  https://api.github.com/authorizations/TOKEN_ID
```

## 3. Git Credentials Configuration

### 3.1 SSH Keys (Recommended)

```bash
# Generate key
ssh-keygen -t ed25519 -C "kushin77@ollama" -f ~/.ssh/ollama_rsa

# Add to SSH agent
ssh-add ~/.ssh/ollama_rsa

# Add public key to GitHub Settings → SSH and GPG Keys
cat ~/.ssh/ollama_rsa.pub

# Configure git
git config user.email "your-email@example.com"
git config user.name "Your Name"
git config core.sshCommand "ssh -i ~/.ssh/ollama_rsa"
```

### 3.2 HTTPS with Git Credentials Helper

```bash
# Configure credential helper
git config --global credential.helper store
# OR for macOS (keychain):
git config --global credential.helper osxkeychain
# OR for Linux (pass):
git config --global credential.helper pass

# First git operation will prompt for credentials
git clone https://github.com/kushin77/ollama.git

# Credentials stored in ~/.git-credentials (set permissions: 600)
chmod 600 ~/.git-credentials
```

### 3.3 GitHub CLI Authentication

```bash
# Login to GitHub
gh auth login

# Choose: HTTPS or SSH
# It will guide token creation and setup

# Verify authentication
gh auth status
```

## 4. Using Secrets in Code

### 4.1 Local Development Setup

```bash
# 1. Create .env file (NEVER commit)
cat > .env << 'EOF'
GITHUB_TOKEN=ghp_your_token_here
GITHUB_USER=kushin77
GITHUB_REPO=ollama
EOF

# 2. Add to .gitignore (already there by default)
echo ".env" >> .gitignore

# 3. Load environment before running
source .env
go run ./cmd/github-issues -issue 1234 -state closed
```

### 4.2 CI/CD with GitHub Actions

```yaml
# .github/workflows/issue-automation.yml
name: Issue Automation

on:
  issues:
    types: [opened, edited]

jobs:
  process:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Process Issue
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GITHUB_REPO: kushin77/ollama
        run: |
          go run ./cmd/github-issues/main.go \
            -issue ${{ github.event.issue.number }} \
            -action auto-label
```

**GitHub Secrets Setup**:
1. Go to: Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Name: `GITHUB_TOKEN`
4. Value: Your PAT (starts with `ghp_`)

### 4.3 Code Best Practices

```go
// ✅ GOOD: Use environment variables
token := os.Getenv("GITHUB_TOKEN")
if token == "" {
    log.Fatal("GITHUB_TOKEN not set")
}

// ❌ BAD: Hardcoded tokens
const token = "ghp_abc123xyz"

// ✅ GOOD: Validate before use
if !strings.HasPrefix(token, "ghp_") {
    return fmt.Errorf("invalid token format")
}

// ✅ GOOD: Never log full token
fmt.Printf("Using token: %s***\n", token[:10])

// ✅ GOOD: Use secrets manager for sensitive operations
secret := retrieveSecret("GITHUB_WEBHOOK_SECRET")
hmacValid := verifyWebhookSignature(signature, secret, payload)
```

## 5. API Authentication Methods

### 5.1 Priority Order (Recommended)

1. **GitHub App** (for automation) - Highest security
2. **Personal Access Token** (for individuals) - High security
3. **OAuth tokens** (for web apps) - Medium security
4. **SSH keys** (for git operations) - High security

### 5.2 Method Coverage

```
GitHub App:
├─ Scoped to specific repos/orgs
├─ Higher rate limits (15,000 req/hr)
├─ No personal credentials exposed
└─ Best for CI/CD

Personal Access Token (PAT):
├─ Tied to user account
├─ 5,000 req/hr rate limit
├─ Easy to revoke
└─ Good for individual developers

SSH Keys:
├─ For git operations only
├─ No HTTP credentials in memory
├─ Per-key rotation possible
└─ Best for local development
```

## 6. Updating Issues Securely

### 6.1 Basic Pattern

```go
// Example: Update issue securely
func safeMergeIssue(issueNum int, pat string) error {
    // 1. Validate inputs
    if issueNum <= 0 || pat == "" {
        return errors.New("invalid inputs")
    }
    
    // 2. Authenticate
    ctx := context.Background()
    ts := oauth2.StaticTokenSource(&oauth2.Token{AccessToken: pat})
    tc := oauth2.NewClient(ctx, ts)
    client := github.NewClient(tc)
    
    // 3. Update with error handling
    _, _, err := client.Issues.Edit(ctx, "kushin77", "ollama", issueNum,
        &github.IssueRequest{
            State: github.String("closed"),
        })
    
    if err != nil {
        // Log error WITHOUT token
        log.Printf("Failed to update issue: %v", err)
        return fmt.Errorf("update failed: %w", err)
    }
    
    return nil
}
```

### 6.2 Batch Operations

```go
func batchUpdateIssuesSecurely(issueNums []int, updates map[int]*github.IssueRequest) error {
    pat := os.Getenv("GITHUB_TOKEN")
    if pat == "" {
        return fmt.Errorf("GITHUB_TOKEN not configured")
    }
    
    ctx := context.Background()
    client := setupClient(ctx, pat)
    
    // Implement rate limiting (5000 req/hr)
    ticker := time.NewTicker(800 * time.Millisecond) // ~4.5 req/sec
    defer ticker.Stop()
    
    for _, issueNum := range issueNums {
        <-ticker.C // Wait for rate limit window
        
        _, _, err := client.Issues.Edit(ctx, "kushin77", "ollama", issueNum, updates[issueNum])
        if err != nil {
            log.Printf("Failed to update %d: %v", issueNum, err)
            continue
        }
    }
    
    return nil
}
```

## 7. Webhook Security

### 7.1 Webhook Signature Verification

```go
// ALWAYS verify webhook signatures
func verifyWebhookSignature(signature string, secret []byte, payload []byte) bool {
    mac := hmac.New(sha256.New, secret)
    mac.Write(payload)
    expected := "sha256=" + hex.EncodeToString(mac.Sum(nil))
    
    return subtle.ConstantTimeCompare([]byte(expected), []byte(signature)) == 1
}

// In request handler
func handleGitHubWebhook(w http.ResponseWriter, r *http.Request) {
    signature := r.Header.Get("X-Hub-Signature-256")
    secret := []byte(os.Getenv("GITHUB_WEBHOOK_SECRET"))
    
    body, _ := io.ReadAll(r.Body)
    
    if !verifyWebhookSignature(signature, secret, body) {
        http.Error(w, "Invalid signature", http.StatusUnauthorized)
        return
    }
    
    // Safe to process webhook
}
```

### 7.2 Webhook Delivery Events

Configure in: Settings → Webhooks → Payload URL

**Secure payload delivery**:
- ✅ Always use HTTPS URL
- ✅ Enable SSL verification
- ✅ Set a strong webhook secret (32+ random chars)
- ✅ Verify signature on every delivery

## 8. Audit & Monitoring

### 8.1 Token Usage Logging

```go
// Log token usage (WITHOUT the token itself)
type TokenAudit struct {
    Timestamp  time.Time
    TokenID    string // First 10 chars only
    Operation  string
    Repository string
    Success    bool
    Error      string
}

func logTokenUsage(operation string, success bool, err error) {
    audit := TokenAudit{
        Timestamp:  time.Now(),
        TokenID:    os.Getenv("GITHUB_TOKEN")[:10],
        Operation:  operation,
        Repository: "kushin77/ollama",
        Success:    success,
        Error:      fmt.Sprint(err),
    }
    
    // Log to file or monitoring service
}
```

### 8.2 Rate Limit Monitoring

```bash
# Check rate limit status
curl -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/rate_limit | jq '.rate_limit'

# Output:
{
  "limit": 5000,
  "remaining": 4999,
  "reset": 1713360000
}
```

## 9. Emergency Procedures

### 9.1 Token Compromise

**If you suspect token compromise:**

1. **Immediately revoke** (Settings → Personal Access Tokens → Delete)
2. **Generate new token** with same name + timestamp
3. **Update all systems** using the old token:
   - CI/CD secrets
   - Local machines
   - Documentation
4. **Audit recent API calls** for unauthorized activity
5. **Log the incident** for security review

### 9.2 Repository Access Revoke

```bash
# Remove all OAuth tokens for this repo
curl -H "Authorization: token $GITHUB_TOKEN" \
  -X DELETE \
  https://api.github.com/repos/kushin77/ollama/authorization

# Remove user access
curl -H "Authorization: token $GITHUB_TOKEN" \
  -X DELETE \
  https://api.github.com/repos/kushin77/ollama/collaborators/{username}
```

## 10. Compliance Checklist

- [ ] All secrets stored in environment variables (never in code)
- [ ] `.env` files added to `.gitignore`
- [ ] PAT tokens scoped to minimum permissions
- [ ] Webhook signatures verified on every delivery
- [ ] Token usage audited and logged
- [ ] Tokens rotated quarterly
- [ ] API calls made at rate-aware intervals
- [ ] Error handling never exposes tokens
- [ ] CI/CD secrets masked in logs
- [ ] SSH keys protected with passphrases

## 11. References

- [GitHub REST API Docs](https://docs.github.com/en/rest)
- [Creating PATs](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)
- [go-github Library](https://github.com/google/go-github)
- [GitHub Actions Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [Webhook Security](https://docs.github.com/en/developers/webhooks-and-events/webhooks)

---

**Questions?** Reach out in the [Discord community](https://discord.gg/ollama)
