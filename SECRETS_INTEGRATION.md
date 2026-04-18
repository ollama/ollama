# GSM & GitHub Integration - Quick Reference

## What Was Implemented

This implementation adds full support for Google Secret Manager (GSM) and GitHub integration to Ollama.

### New Files Created

1. **`internal/secrets/gsm.go`** - Google Secret Manager client
   - `NewGSMClient()` - Create GSM client
   - `GetSecret()` - Retrieve latest secret
   - `GetSecretVersion()` - Retrieve specific version
   - `Close()` - Clean up resources

2. **`internal/secrets/github.go`** - GitHub API client
   - `NewClient()` - Create GitHub client
   - `NewClientWithURL()` - Create client for GitHub Enterprise
   - `GetAuthenticatedUser()` - Get authenticated user info
   - `GetRepository()` - Get repository info
   - `ValidateToken()` - Validate GitHub token

3. **`internal/secrets/gsm_test.go`** - Unit tests for GSM
4. **`internal/secrets/github_test.go`** - Unit tests for GitHub
5. **`internal/secrets/README.md`** - Package documentation

### Updated Files

1. **`envconfig/config.go`** - Added environment variables
   - `OLLAMA_GSM_ENABLED` - Bool flag to enable GSM
   - `OLLAMA_GSM_PROJECT_ID` - Google Cloud project ID
   - `OLLAMA_GITHUB_TOKEN` - GitHub token storage
   - `OLLAMA_GITHUB_USER` - GitHub username storage

### New Documentation

1. **`docs/secrets.md`** - Comprehensive guide
   - GSM setup and configuration
   - GitHub integration setup
   - Environment variables reference
   - CI/CD integration examples
   - Security best practices
   - Troubleshooting guide

2. **`docs/github-workflows.md`** - Workflow-specific guide
   - Current release workflow description
   - GitHub Secrets and Variables used
   - How to update/rotate secrets
   - Integration with secrets package
   - Security best practices for CI/CD

3. **`api/examples/secrets_integration/main.go`** - Example program
   - Demonstrates GSM usage
   - Demonstrates GitHub usage
   - Shows combined workflow
   - Runnable reference implementation

## Quick Start

### Enable GSM

```bash
# Set environment variables
export OLLAMA_GSM_ENABLED=true
export OLLAMA_GSM_PROJECT_ID=your-project-id

# Authenticate with Google Cloud
gcloud auth application-default login

# Create a secret (if testing)
gcloud secrets create github-token --data-file=- <<< "your-token"
```

### Enable GitHub Integration

```bash
# Create/get a GitHub Personal Access Token
# https://github.com/settings/tokens

export OLLAMA_GITHUB_TOKEN=ghp_xxxxx
```

### Use in Code

```go
import "github.com/ollama/ollama/internal/secrets/github"

client := github.NewClient(token)
user, err := client.GetAuthenticatedUser(ctx)
```

## Environment Variables Reference

| Variable | Type | Purpose | Default |
|----------|------|---------|---------|
| `OLLAMA_GSM_ENABLED` | bool | Enable Google Secret Manager | `false` |
| `OLLAMA_GSM_PROJECT_ID` | string | Google Cloud project ID | `` |
| `OLLAMA_GITHUB_TOKEN` | string | GitHub personal access token | `` |
| `OLLAMA_GITHUB_USER` | string | GitHub username (optional) | `` |

## Example Usage Patterns

### Pattern 1: Get Secret from GSM and Authenticate to GitHub

```go
ctx := context.Background()

// Get GitHub token from GSM
gsmClient, _ := secrets.NewGSMClient(&secrets.GSMConfig{
    ProjectID: envconfig.GSMProjectID(),
    Enabled: true,
})
token, _ := gsmClient.GetSecret(ctx, "github-token")
gsmClient.Close()

// Use token with GitHub
ghClient := github.NewClient(token)
user, _ := ghClient.GetAuthenticatedUser(ctx)
```

### Pattern 2: Use GitHub Token from Environment

```go
token := envconfig.GitHubToken()
client := github.NewClient(token)
repo, _ := client.GetRepository(ctx, "owner", "repo")
```

### Pattern 3: GitHub Enterprise Server

```go
client := github.NewClientWithURL(token, "https://github.enterprise.com/api/v3")
user, _ := client.GetAuthenticatedUser(ctx)
```

## Testing the Implementation

### Run Unit Tests

```bash
cd internal/secrets
go test -v ./...
```

### Run Example Program

```bash
# With GSM
OLLAMA_GSM_ENABLED=true OLLAMA_GSM_PROJECT_ID=my-project go run api/examples/secrets_integration/main.go -gsm

# With GitHub
OLLAMA_GITHUB_TOKEN=ghp_xxx go run api/examples/secrets_integration/main.go -test-github

# Combined
OLLAMA_GSM_ENABLED=true OLLAMA_GSM_PROJECT_ID=my-project OLLAMA_GITHUB_TOKEN=ghp_xxx go run api/examples/secrets_integration/main.go -gsm -test-github
```

## Security Checklist

- [ ] GSM credentials never logged or printed
- [ ] GitHub tokens used only for API calls
- [ ] All HTTP headers properly set
- [ ] Context-based request handling
- [ ] Secrets disabled by default (GSM)
- [ ] Regular secret rotation implemented
- [ ] Service accounts have minimal permissions
- [ ] Audit logging enabled for secret access

## Integration with Existing Workflows

The implementation integrates seamlessly with:

1. **Release workflow** (`.github/workflows/release.yaml`)
   - Uses existing `GOOGLE_SIGNING_CREDENTIALS`
   - Can access additional secrets via new GSM support

2. **Test workflow** (`.github/workflows/test.yaml`)
   - Can use GitHub token for private repo access
   - Can retrieve build secrets from GSM

3. **CI/CD pipelines**
   - Google Cloud Build support
   - GitHub Actions support
   - Custom CI/CD systems via environment variables

## Troubleshooting

### "google secret manager is disabled"
- Set `OLLAMA_GSM_ENABLED=true`
- Set `OLLAMA_GSM_PROJECT_ID=your-project-id`

### "GitHub API error: Invalid authentication credentials"
- Verify `OLLAMA_GITHUB_TOKEN` is set
- Check token has not expired
- Verify token has required scopes

### GSM Connection Failed
```bash
# Test Google Cloud authentication
gcloud auth list
gcloud auth application-default print-access-token

# Test GSM access
gcloud secrets versions access latest --secret=test-secret
```

### GitHub API Rate Limiting
- Use authenticated requests (higher limits)
- Check rate limit status: `curl -H "Authorization: Bearer $TOKEN" https://api.github.com/rate_limit`

## Next Steps

1. **Review documentation**
   - Read [docs/secrets.md](../docs/secrets.md)
   - Read [docs/github-workflows.md](../docs/github-workflows.md)

2. **Run examples**
   - Try the integration example program
   - Test with your own GSM and GitHub setup

3. **Integrate into workflows**
   - Update CI/CD to use new GSM support
   - Add GitHub token management to build scripts
   - Implement secret rotation procedures

4. **Future enhancements**
   - AWS Secrets Manager support
   - HashiCorp Vault integration
   - Azure Key Vault support
   - Secret caching and TTL
   - Audit logging

## Support

For issues or questions:
- Check [docs/secrets.md](../docs/secrets.md) troubleshooting section
- Review [internal/secrets/README.md](../internal/secrets/README.md)
- Check unit tests in `internal/secrets/*_test.go`
- Review example in `api/examples/secrets_integration/main.go`
