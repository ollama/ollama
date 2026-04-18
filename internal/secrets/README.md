# Secrets Package

This package provides secret management functionality for Ollama, including Google Secret Manager (GSM) integration and GitHub API client.

## Components

### GSM Client (`gsm.go`)

Provides integration with Google Cloud's Secret Manager service.

**Features:**
- Retrieve secrets by name (latest version)
- Retrieve specific secret versions
- Automatic lazy initialization of GSM client
- Comprehensive error handling

**Example:**
```go
config := &secrets.GSMConfig{
    ProjectID: "my-project",
    Enabled: true,
}
client, err := secrets.NewGSMClient(config)
if err != nil {
    log.Fatal(err)
}
defer client.Close()

secret, err := client.GetSecret(ctx, "my-secret")
```

### GitHub Client (`github.go`)

Provides GitHub API client for authentication and repository operations.

**Features:**
- Retrieve authenticated user information
- Access repository details
- List and retrieve issues
- Validate GitHub tokens
- Support for GitHub Enterprise Server
- Type-safe API responses

**Example:**
```go
client := github.NewClient("github_token_here")

// Get authenticated user
user, err := client.GetAuthenticatedUser(ctx)
if err != nil {
    log.Fatal(err)
}

// Get repository
repo, err := client.GetRepository(ctx, "owner", "repo")

// List issues
issues, err := client.ListIssues(ctx, "owner", "repo", &github.IssueListOptions{
    State:   "open",
    Sort:    "created",
    PerPage: 30,
})

// Get single issue
issue, err := client.GetIssue(ctx, "owner", "repo", 123)
```

## Environment Variables

The following environment variables are configured in `envconfig/config.go`:

- `OLLAMA_GSM_ENABLED` - Enable Google Secret Manager (bool)
- `OLLAMA_GSM_PROJECT_ID` - Google Cloud project ID (string)
- `OLLAMA_GITHUB_TOKEN` - GitHub personal access token (string)
- `OLLAMA_GITHUB_USER` - GitHub username (string, optional)

## Security Considerations

1. **Credentials are never logged** - All API calls are made with proper error handling to avoid leaking sensitive data
2. **HTTP headers properly set** - Authorization headers follow GitHub and Google Cloud standards
3. **Context-based requests** - All operations support Go contexts for timeout and cancellation
4. **Secure defaults** - GSM is disabled by default; must be explicitly enabled

## Testing

Unit tests are provided for:
- GSM configuration validation
- GitHub token parsing
- Client initialization
- Error handling

Run tests with:
```bash
go test ./internal/secrets/...
```

## Future Enhancements

- Support for AWS Secrets Manager
- Support for HashiCorp Vault
- Support for Azure Key Vault
- Secret caching with TTL
- Token refresh mechanisms
- Secret audit logging
