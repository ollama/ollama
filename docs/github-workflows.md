# GitHub Integration in Release Workflows

This document describes how Ollama uses GitHub integration in its CI/CD workflows, particularly for release builds.

## Current Workflow

### Release Build Workflow (`.github/workflows/release.yaml`)

The release workflow uses GitHub Actions and Google Cloud integration for:

1. **Code Signing** - Windows binaries are signed using Google Cloud KMS
2. **Artifact Upload** - Compiled binaries are uploaded to GitHub Releases
3. **Build Orchestration** - GitHub Actions manages the build pipeline across multiple platforms

## GitHub Secrets Used

The following secrets are configured in the GitHub repository:

- `GOOGLE_SIGNING_CREDENTIALS` - Google Cloud service account key (JSON format)
- `APPLE_IDENTITY` - Apple Developer identity for macOS signing
- `APPLE_PASSWORD` - App-specific password for Apple notarization
- `MACOS_SIGNING_KEY` - Base64-encoded macOS signing certificate
- `MACOS_SIGNING_KEY_PASSWORD` - Password for macOS signing key

## GitHub Variables Used

The following variables are configured in the GitHub repository:

- `APPLE_TEAM_ID` - Apple Developer Team ID
- `APPLE_ID` - Apple ID for notarization
- `OLLAMA_CERT` - Ollama code signing certificate
- `KEY_CONTAINER` - Windows code signing container name

## How to Update Secrets

### Adding a New Secret

1. Go to repository settings: `https://github.com/ollama/ollama/settings/secrets/actions`
2. Click "New repository secret"
3. Enter the secret name and value
4. Click "Add secret"

### Rotating Secrets

For security best practices, rotate secrets periodically:

```bash
# For service account keys:
gcloud iam service-accounts keys create new-key.json \
  --iam-account=service-account@project.iam.gserviceaccount.com

# Delete old key when verification is complete
gcloud iam service-accounts keys delete OLD_KEY_ID \
  --iam-account=service-account@project.iam.gserviceaccount.com
```

## Using GitHub Integration in Your Build

### Example: Accessing GitHub Secrets in a Workflow

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Use GitHub secret
        env:
          MY_SECRET: ${{ secrets.MY_SECRET_NAME }}
        run: |
          # Secret is available as environment variable
          # Never print or log the secret
          echo "Configured secret"
```

### Example: Accessing Google Cloud Credentials

```yaml
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GOOGLE_SIGNING_CREDENTIALS }}
      
      - name: Use Google Cloud
        run: |
          gcloud auth list
```

## Integration with Ollama Secrets Package

The `internal/secrets` package can be used in Ollama builds to:

1. Retrieve secrets from Google Secret Manager
2. Validate GitHub tokens
3. Access GitHub repository information

### Example Build Script Using Secrets

```go
package main

import (
	"context"
	"log"
	"os"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/internal/secrets"
	"github.com/ollama/ollama/internal/secrets/github"
)

func main() {
	ctx := context.Background()

	// Setup GSM for retrieving build secrets
	if envconfig.GSMEnabled() {
		gsmConfig := &secrets.GSMConfig{
			ProjectID: envconfig.GSMProjectID(),
			Enabled:   true,
		}
		gsmClient, err := secrets.NewGSMClient(gsmConfig)
		if err != nil {
			log.Fatal(err)
		}
		defer gsmClient.Close()

		// Retrieve a build signing key from GSM
		signingKey, err := gsmClient.GetSecret(ctx, "build-signing-key")
		if err != nil {
			log.Fatal(err)
		}
		
		// Use signing key in build process
		_ = signingKey
	}

	// Validate GitHub token
	token := envconfig.GitHubToken()
	if token != "" {
		valid, err := github.ValidateGitHubToken(ctx, token)
		if !valid || err != nil {
			log.Fatal("Invalid GitHub token")
		}

		// Get authenticated user
		client := github.NewClient(token)
		user, err := client.GetAuthenticatedUser(ctx)
		if err != nil {
			log.Fatal(err)
		}

		log.Printf("Authenticated as: %s", user.Login)
	}
}
```

## Security Best Practices

1. **Never commit secrets** - Always use GitHub Secrets
2. **Use least privilege** - Grant only necessary permissions to service accounts
3. **Rotate credentials regularly** - Update secrets every 90 days
4. **Audit access** - Review GitHub Actions logs and Google Cloud audit logs
5. **Limit secret scope** - Use environment-specific secrets
6. **Use OIDC tokens** - Prefer OIDC token authentication over long-lived credentials

## Troubleshooting

### Secret Not Found in Workflow

```bash
# Check if secret is properly configured
gh secret list --repo kushin77/ollama

# Verify the secret name matches exactly (case-sensitive)
```

### GitHub API Rate Limiting

If you hit rate limits:

```bash
# Check your rate limit status
curl -H "Authorization: Bearer $TOKEN" https://api.github.com/rate_limit

# Authenticated requests have higher limits (5000 vs 60 per hour)
```

### Google Cloud Authentication Issues

```bash
# Test Google Cloud authentication
gcloud auth list
gcloud auth application-default print-access-token

# Check service account permissions
gcloud projects get-iam-policy PROJECT_ID
```

## See Also

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [GitHub Secrets Documentation](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [Google Cloud Build Documentation](https://cloud.google.com/build/docs)
- [Secrets Package Documentation](../internal/secrets/README.md)
- [Main Secrets Documentation](../docs/secrets.md)
