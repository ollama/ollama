---
name: github-api-secrets-instructions
description: "Use when: integrating GitHub API for issue updates, handling PAT/API tokens, managing secrets in CI/CD, or automating GitHub workflows with credentials"
applyTo: "**"
---

# GitHub API & Secrets Integration Guide

This guide covers secure integration with GitHub API for updating issues, managing pull requests, and automating workflows using Personal Access Tokens (PAT) and other credentials.

## Quick Start: Update Issues via GitHub API

### 1. Create a Personal Access Token (PAT)

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Select scopes:
   - `repo` - Full control of private repositories
   - `write:gists` - Write access to gists
   - `admin:repo_hook` - Full control of repository hooks
   - `admin:org_hook` - Full control of organization hooks (if org member)
4. Copy the token immediately (you won't see it again)
5. Store securely - **NEVER commit to version control**

### 2. Store Token Securely

#### Local Development
```bash
# Option A: Environment variable (temporary)
export GITHUB_TOKEN="ghp_your_token_here"

# Option B: .env file (add to .gitignore)
# .env
GITHUB_TOKEN=ghp_your_token_here
GITHUB_USER=kushin77

# Option C: Git credentials helper (recommended)
git config --global credential.helper store
# This stores credentials in ~/.git-credentials (still keep secure!)
```

#### CI/CD (GitHub Actions)
```yaml
# .github/workflows/issue-updater.yml
name: Update Issues

on:
  workflow_dispatch:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM UTC

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Update Issues
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          go run ./cmd update-issues
```

## Authentication Methods

### Method 1: Personal Access Token (Recommended)

```bash
# Direct HTTP with Basic Auth
curl -H "Authorization: token ghp_your_token_here" \
  https://api.github.com/repos/kushin77/ollama/issues/1234

# With Go client
package main

import (
    "context"
    "github.com/google/go-github/v60/github"
    "golang.org/x/oauth2"
)

func main() {
    ctx := context.Background()
    ts := oauth2.StaticTokenSource(
        &oauth2.Token{AccessToken: os.Getenv("GITHUB_TOKEN")},
    )
    tc := oauth2.NewClient(ctx, ts)
    client := github.NewClient(tc)
    
    // Now use client to interact with GitHub API
}
```

### Method 2: GitHub App (For Automation)

```go
// Use github.com/bradleyfalzon/ghinstallation for GitHub Apps
import (
    "github.com/bradleyfalzon/ghinstallation/v2"
)

func setupGitHubAppClient() *github.Client {
    itr, err := ghinstallation.NewKeyFromFile(
        http.DefaultClient,
        12345,              // App ID
        67890,              // Installation ID
        []byte(privateKey), // Private key
    )
    if err != nil {
        panic(err)
    }
    
    client := github.NewClient(&http.Client{Transport: itr})
    return client
}
```

### Method 3: OAuth2 (Web Apps)

```go
// For web applications with user authorization
config := &oauth2.Config{
    ClientID:     os.Getenv("GITHUB_OAUTH_ID"),
    ClientSecret: os.Getenv("GITHUB_OAUTH_SECRET"),
    Scopes:       []string{"repo"},
    Endpoint:     github.Endpoint,
}

token, err := config.Token(context.Background())
client := github.NewClient(config.Client(context.Background(), token))
```

## Updating Issues Programmatically

### Update Issue State

```go
package main

import (
    "context"
    "github.com/google/go-github/v60/github"
    "os"
)

func updateIssue(ctx context.Context, client *github.Client) error {
    // Update issue state (open/closed)
    issueUpdate := &github.IssueRequest{
        State: github.String("closed"),
    }
    
    _, _, err := client.Issues.Edit(ctx, "kushin77", "ollama", 1234, issueUpdate)
    return err
}
```

### Add/Update Labels

```go
func updateIssueLabels(ctx context.Context, client *github.Client) error {
    labels := []string{"bug", "critical", "needs-review"}
    
    _, _, err := client.Issues.ReplaceLabelsForIssue(ctx, "kushin77", "ollama", 1234, labels)
    return err
}
```

### Add Comments

```go
func commentOnIssue(ctx context.Context, client *github.Client) error {
    comment := &github.IssueComment{
        Body: github.String("Fixed in PR #5678"),
    }
    
    _, _, err := client.Issues.CreateComment(ctx, "kushin77", "ollama", 1234, comment)
    return err
}
```

### Update Milestones & Assignees

```go
func assignAndMilestone(ctx context.Context, client *github.Client) error {
    issueUpdate := &github.IssueRequest{
        Assignees: []string{"kushin77", "other-user"},
        Milestone: github.Int(5),
    }
    
    _, _, err := client.Issues.Edit(ctx, "kushin77", "ollama", 1234, issueUpdate)
    return err
}
```

## Batch Operations

```go
func batchUpdateIssues(ctx context.Context, client *github.Client, issueNumbers []int) error {
    updates := []struct {
        number int
        update *github.IssueRequest
    }{
        {1234, &github.IssueRequest{State: github.String("closed")}},
        {1235, &github.IssueRequest{State: github.String("open")}},
    }
    
    for _, item := range updates {
        _, _, err := client.Issues.Edit(ctx, "kushin77", "ollama", item.number, item.update)
        if err != nil {
            return fmt.Errorf("failed to update issue #%d: %w", item.number, err)
        }
    }
    
    return nil
}
```

## CLI Tool Example

Create a reusable CLI tool in `cmd/github-issues/`:

```go
// cmd/github_issues/main.go
package main

import (
    "context"
    "flag"
    "fmt"
    "os"
    
    "github.com/google/go-github/v60/github"
    "golang.org/x/oauth2"
)

func main() {
    issueNum := flag.Int("issue", 0, "Issue number to update")
    state := flag.String("state", "", "New state (open/closed)")
    labels := flag.String("labels", "", "Comma-separated labels")
    comment := flag.String("comment", "", "Comment to add")
    
    flag.Parse()
    
    token := os.Getenv("GITHUB_TOKEN")
    if token == "" {
        fmt.Fprintln(os.Stderr, "GITHUB_TOKEN environment variable not set")
        os.Exit(1)
    }
    
    ctx := context.Background()
    ts := oauth2.StaticTokenSource(&oauth2.Token{AccessToken: token})
    tc := oauth2.NewClient(ctx, ts)
    client := github.NewClient(tc)
    
    // Perform updates
    if *state != "" {
        update := &github.IssueRequest{State: github.String(*state)}
        _, _, err := client.Issues.Edit(ctx, "kushin77", "ollama", *issueNum, update)
        if err != nil {
            fmt.Fprintf(os.Stderr, "Failed to update state: %v\n", err)
            os.Exit(1)
        }
        fmt.Printf("Issue #%d state updated to: %s\n", *issueNum, *state)
    }
    
    if *comment != "" {
        c := &github.IssueComment{Body: github.String(*comment)}
        _, _, err := client.Issues.CreateComment(ctx, "kushin77", "ollama", *issueNum, c)
        if err != nil {
            fmt.Fprintf(os.Stderr, "Failed to add comment: %v\n", err)
            os.Exit(1)
        }
        fmt.Printf("Comment added to issue #%d\n", *issueNum)
    }
}
```

Usage:
```bash
export GITHUB_TOKEN="ghp_..."
go run ./cmd/github_issues -issue 1234 -state closed -comment "Fixed in v0.2.0"
```

## Security Best Practices

### ✅ DO

- [ ] Store tokens in environment variables
- [ ] Use `.gitignore` for `.env` files
- [ ] Rotate tokens regularly
- [ ] Use GitHub Secrets in CI/CD workflows
- [ ] Scope tokens to minimum required permissions
- [ ] Use read-only tokens when possible
- [ ] Log API calls for audit trails
- [ ] Validate webhook signatures (HMAC-SHA256)

### ❌ DON'T

- Never commit tokens to version control
- Don't hardcode PATs in source code
- Don't share tokens via email/chat
- Don't use expired tokens without rotation
- Don't use overly broad scopes
- Don't log full tokens
- Don't disable webhook signature verification
- Don't trust user input without validation

## Environment Variables

```bash
# .env (NEVER commit!)
GITHUB_TOKEN=ghp_...               # PAT for authentication
GITHUB_USER=kushin77               # Repository owner
GITHUB_REPO=ollama                 # Repository name
GITHUB_WEBHOOK_SECRET=secret...    # Webhook signature secret
GITHUB_APP_ID=12345                # GitHub App ID
GITHUB_APP_INSTALLATION_ID=67890   # Installation ID
```

## Webhook Integration

```go
// Handle GitHub webhooks securely
func handleWebhook(w http.ResponseWriter, r *http.Request) {
    // Verify signature
    signature := r.Header.Get("X-Hub-Signature-256")
    body, _ := io.ReadAll(r.Body)
    
    expected := "sha256=" + hmacSHA256(os.Getenv("GITHUB_WEBHOOK_SECRET"), string(body))
    if signature != expected {
        http.Error(w, "Invalid signature", http.StatusUnauthorized)
        return
    }
    
    // Parse event
    var event github.WebhookPayload
    json.Unmarshal(body, &event)
    
    // Handle event (e.g., update issue on PR)
}

func hmacSHA256(secret, body string) string {
    h := hmac.New(sha256.New, []byte(secret))
    h.Write([]byte(body))
    return hex.EncodeToString(h.Sum(nil))
}
```

## Rate Limiting

```go
// GitHub API rate limits:
// - 60 requests/hour (unauthenticated)
// - 5,000 requests/hour (authenticated)
// - Check headers for current limits

func checkRateLimit(ctx context.Context, client *github.Client) {
    rate, _, err := client.RateLimits(ctx)
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Remaining: %d/%d\n", rate.Core.Remaining, rate.Core.Limit)
    fmt.Printf("Reset at: %v\n", rate.Core.Reset)
}
```

## Testing with Mocked API

```go
func TestIssueUpdate(t *testing.T) {
    // Use httptest to mock GitHub API
    server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.WriteHeader(http.StatusOK)
        json.NewEncoder(w).Encode(&github.Issue{
            Number: github.Int(1234),
            State:  github.String("closed"),
        })
    }))
    defer server.Close()
    
    client := github.NewClient(&http.Client{})
    // Override base URL for testing
    client.BaseURL, _ = url.Parse(server.URL + "/")
    
    // Run tests
}
```

## Documentation References

- [GitHub REST API Documentation](https://docs.github.com/en/rest)
- [go-github Library](https://github.com/google/go-github)
- [GitHub Personal Access Tokens](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)
- [GitHub Apps Authentication](https://docs.github.com/en/developers/apps/building-github-apps/authenticating-with-github-apps)
- [Webhooks](https://docs.github.com/en/developers/webhooks-and-events/webhooks)

## Troubleshooting

**"Bad credentials" error:**
- Verify token is valid and not expired
- Check token has correct scopes
- Ensure token is passed correctly to client

**Rate limit exceeded:**
- Implement exponential backoff retry logic
- Use conditional requests (ETag headers)
- Consider using GitHub App (higher limits)

**Webhook not triggering:**
- Verify webhook URL is accessible
- Check webhook secret matches
- Review GitHub Actions logs for delivery status

---
