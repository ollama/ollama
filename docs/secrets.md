# Secrets Management and GitHub Integration

This document describes how to configure Ollama to use Google Secret Manager (GSM) for credential storage and GitHub integration.

## ⚠️ This Workspace: Canonical GSM + Git Credential Config

The `git-credential-gsm` helper is installed at `/usr/local/bin/git-credential-gsm`
and configured globally (`~/.gitconfig → credential.helper = gsm`).

| Setting | Value |
|---------|-------|
| GCP Project | `gcp-eiq` (`GSM_PROJECT` env var) |
| Secret Name | `prod-github-token` (`GSM_SECRET_NAME` env var) |
| GitHub user | `kushin77` |
| Repository | `kushin77/ollama` |

**gcloud must be authenticated** before the helper or orchestrator can fetch the PAT:
```bash
gcloud auth login
# OR service account:
gcloud auth activate-service-account --key-file=/path/to/sa.json
```

To invoke the orchestrator with the correct GSM-backed auth:
```bash
export OLLAMA_GSM_ENABLED=true
export GSM_PROJECT=gcp-eiq
export GSM_SECRET_NAME=prod-github-token
python3 cmd/github-issues/orchestrator_enhanced.py \
    --repo kushin77/ollama --all-severities --execute \
    --output .github/orchestrator_report_kushin77_ollama_live.json
```

**Do NOT use** `OLLAMA_GSM_PROJECT_ID` with a placeholder value or `github-token`
as the secret name — the actual secret is `prod-github-token` in project `gcp-eiq`.

GitHub Actions workflows that need the PAT should authenticate to Google Cloud with `WIF_PROVIDER` and `WIF_SERVICE_ACCOUNT`, then resolve the token through `scripts/github-actions-token.sh` with `OLLAMA_GSM_ENABLED=true`, `GSM_PROJECT=gcp-eiq`, and `GSM_SECRET_NAME=prod-github-token`.

The helper script returns the PAT so downstream steps can export it as `GITHUB_TOKEN` or pass it directly to GitHub Actions that need a token input.

---

## Google Secret Manager (GSM) Integration

### Overview

GSM integration allows Ollama to securely retrieve secrets from Google Cloud's Secret Manager service. This is useful for managing API keys, authentication tokens, and other sensitive credentials without storing them locally.

### Setup

#### 1. Enable Google Secret Manager API

First, ensure the Secret Manager API is enabled in your Google Cloud project:

```bash
gcloud services enable secretmanager.googleapis.com
```

#### 2. Create Secrets in GSM

Create secrets in your Google Cloud project:

```bash
gcloud secrets create prod-github-token --data-file=- <<< "your_github_token_here"
gcloud secrets create api-key --data-file=- <<< "your_api_key_here"
```

#### 3. Configure Ollama

Set the following environment variables:

```bash
# Enable GSM
export OLLAMA_GSM_ENABLED=true

# Set your Google Cloud project ID
export OLLAMA_GSM_PROJECT_ID=your-project-id

# Authenticate with Google Cloud (using default credentials)
gcloud auth application-default login
```

#### 4. Use GSM in Your Application

```go
package main

import (
	"context"
	"log"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/internal/secrets"
)

func main() {
	ctx := context.Background()

	// Create GSM configuration
	gsmConfig := &secrets.GSMConfig{
		ProjectID: envconfig.GSMProjectID(),
		Enabled:   envconfig.GSMEnabled(),
	}

	// Create GSM client
	client, err := secrets.NewGSMClient(gsmConfig)
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	// Retrieve a secret
	secret, err := client.GetSecret(ctx, "prod-github-token")
	if err != nil {
		log.Fatal(err)
	}

	println("Secret retrieved:", secret)
}
```

### Environment Variables

- `OLLAMA_GSM_ENABLED` - Set to `true` to enable GSM (default: `false`)
- `OLLAMA_GSM_PROJECT_ID` - Your Google Cloud project ID (required when GSM is enabled)

### Authentication

GSM authentication uses Google Cloud Application Default Credentials (ADC). You can authenticate using any of these methods:

1. **Service Account Key** (for CI/CD):
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
   gcloud auth activate-service-account --key-file=/path/to/service-account-key.json
   ```

2. **User Authentication** (for local development):
   ```bash
   gcloud auth application-default login
   ```

3. **Workload Identity** (for GKE):
   ```bash
   gcloud iam service-accounts add-iam-policy-binding \
     your-sa@your-project.iam.gserviceaccount.com \
     --role roles/iam.workloadIdentityUser \
     --member "serviceAccount:your-project.svc.id.goog[namespace/ksa-name]"
   ```

## GitHub Integration

### Overview

Ollama provides GitHub integration for:

- Authenticating with GitHub API
- Accessing repository information
- Token validation

### Setup

#### 1. Create a GitHub Personal Access Token

1. Go to https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Select scopes needed for your use case:
   - `repo` - Full control of private repositories
   - `read:user` - Read user profile data
   - `user:email` - Read user email addresses
4. Copy the token

#### 2. Configure Ollama

Store the token either as an environment variable or in GSM:

**Option A: Environment Variable**
```bash
export OLLAMA_GITHUB_TOKEN=your_github_token
```

**Option B: Google Secret Manager**
```bash
gcloud secrets create prod-github-token --data-file=- <<< "your_github_token"
export OLLAMA_GSM_ENABLED=true
export OLLAMA_GSM_PROJECT_ID=your-project-id
export OLLAMA_GITHUB_TOKEN=$(gcloud secrets versions access latest --secret=prod-github-token)
```

#### 3. Use GitHub Integration

```go
package main

import (
	"context"
	"log"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/internal/secrets/github"
)

func main() {
	ctx := context.Background()

	// Get token from environment
	token := envconfig.GitHubToken()
	if token == "" {
		log.Fatal("OLLAMA_GITHUB_TOKEN not set")
	}

	// Create GitHub client
	client := github.NewClient(token)

	// Get authenticated user
	user, err := client.GetAuthenticatedUser(ctx)
	if err != nil {
		log.Fatal(err)
	}

	log.Printf("Authenticated as: %s (%s)", user.Login, user.Name)

	// Get repository information
	repo, err := client.GetRepository(ctx, "ollama", "ollama")
	if err != nil {
		log.Fatal(err)
	}

	log.Printf("Repository: %s - %s", repo.FullName, repo.Description)

	// List open issues
	issues, err := client.ListIssues(ctx, "ollama", "ollama", &github.IssueListOptions{
		State:   "open",
		Sort:    "created",
		PerPage: 10,
	})
	if err != nil {
		log.Fatal(err)
	}

	log.Printf("Found %d open issues", len(issues))
	for _, issue := range issues {
		log.Printf("  #%d: %s (by %s)", issue.Number, issue.Title, issue.User.Login)
	}

	// Get a specific issue by number
	issue, err := client.GetIssue(ctx, "ollama", "ollama", 1)
	if err != nil {
		log.Fatal(err)
	}

	log.Printf("Issue #%d: %s", issue.Number, issue.Title)
}
```

#### 4. Check Issues from Command Line

Ollama provides a command-line tool to check GitHub issues:

```bash
# View open issues
go run cmd/github-issues/main.go

# View with options
go run cmd/github-issues/main.go -state closed -limit 20 -sort updated

# Filter by labels
go run cmd/github-issues/main.go -labels "bug,help wanted"

# Check a different repository
go run cmd/github-issues/main.go -owner golang -repo go
```

For more details, see [cmd/github-issues/README.md](../cmd/github-issues/README.md).

### Environment Variables

- `OLLAMA_GITHUB_TOKEN` - GitHub personal access token or OAuth token
- `OLLAMA_GITHUB_USER` - GitHub username (optional, can be retrieved from token)

### GitHub Enterprise Server

For GitHub Enterprise Server, use `NewClientWithURL`:

```go
token := envconfig.GitHubToken()
client := github.NewClientWithURL(token, "https://github.enterprise.com/api/v3")
```

## CI/CD Integration

### GitHub Actions

Store secrets securely in GitHub Actions:

```yaml
name: CI

on: [push]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up environment
        env:
          OLLAMA_GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          OLLAMA_GSM_ENABLED: 'false'
        run: |
          # Your build steps here
```

### Google Cloud Build

For Google Cloud Build with GSM:

```yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    env:
      - 'OLLAMA_GSM_ENABLED=true'
      - 'OLLAMA_GSM_PROJECT_ID=$PROJECT_ID'
    args:
      - 'build'
      - '-t'
      - 'gcr.io/$PROJECT_ID/ollama:latest'
      - '.'
```

## Security Best Practices

1. **Never commit credentials** to version control
2. **Use service accounts** for CI/CD (not personal access tokens)
3. **Rotate tokens regularly** in GitHub settings
4. **Use least privilege** - grant only necessary permissions
5. **Monitor secret access** using Google Cloud audit logs
6. **Use Workload Identity** in Kubernetes instead of service account keys
7. **Encrypt environment variables** in transit and at rest

## Troubleshooting

### GSM Connection Issues

```bash
# Test GSM authentication
gcloud secrets versions access latest --secret=prod-github-token

# Check permissions
gcloud projects get-iam-policy your-project-id \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:*"
```

### GitHub Token Issues

```bash
# Test GitHub token
curl -H "Authorization: Bearer $OLLAMA_GITHUB_TOKEN" https://api.github.com/user

# Check token scopes
curl -H "Authorization: Bearer $OLLAMA_GITHUB_TOKEN" https://api.github.com/user | jq '.scopes'
```

## API Reference

### `secrets.GSMClient`

```go
type GSMConfig struct {
    ProjectID string
    Enabled   bool
}

// Create a new GSM client
client, err := secrets.NewGSMClient(&GSMConfig{
    ProjectID: "my-project",
    Enabled:   true,
})

// Get latest version of a secret
secret, err := client.GetSecret(ctx, "secret-name")

// Get specific version
secret, err := client.GetSecretVersion(ctx, "secret-name", "1")

// Close client
client.Close()
```

### `secrets/github.Client`

```go
client := github.NewClient(token)

// Get authenticated user
user, err := client.GetAuthenticatedUser(ctx)

// Get repository
repo, err := client.GetRepository(ctx, "owner", "repo")

// List issues with filtering
issues, err := client.ListIssues(ctx, "owner", "repo", &github.IssueListOptions{
    State:   "open",     // "open", "closed", "all"
    Sort:    "created",  // "created", "updated", "comments"
    Order:   "desc",     // "asc", "desc"
    Labels:  "bug",      // comma-separated label names
    PerPage: 30,         // items per page (max 100)
    Page:    1,          // page number
})

// Get a specific issue
issue, err := client.GetIssue(ctx, "owner", "repo", 123)

// Validate token
err := client.ValidateToken(ctx)

// Utility function
token := github.TokenFromString(authHeaderValue)
```

## See Also

- [Google Cloud Secret Manager Documentation](https://cloud.google.com/secret-manager/docs)
- [GitHub API Documentation](https://docs.github.com/en/rest)
- [Google Cloud Authentication](https://cloud.google.com/docs/authentication)
