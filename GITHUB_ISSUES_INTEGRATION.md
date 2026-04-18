# GitHub Issues Integration - Update Summary

## What Was Added

Building on the GSM and GitHub integration foundation, we've now added comprehensive GitHub issues management capabilities.

## New Features

### 1. GitHub Client Issue Methods
**File:** `internal/secrets/github.go`

Added three new methods to the GitHub client:

#### `ListIssues()`
List issues from a repository with filtering and sorting options.

```go
issues, err := client.ListIssues(ctx, "ollama", "ollama", &github.IssueListOptions{
    State:   "open",        // "open", "closed", "all"
    Sort:    "created",     // "created", "updated", "comments"
    Order:   "desc",        // "asc", "desc"
    Labels:  "bug",         // comma-separated
    PerPage: 30,
    Page:    1,
})
```

**Supported Filters:**
- `State` - Issue state (open, closed, all)
- `Sort` - Sort key (created, updated, comments)
- `Order` - Sort order (asc, desc)
- `Labels` - Filter by labels
- `PerPage` - Items per page (1-100, default 30)
- `Page` - Page number for pagination

#### `GetIssue()`
Retrieve a single issue by number.

```go
issue, err := client.GetIssue(ctx, "ollama", "ollama", 1234)
// Returns: Issue number, title, state, body, author, labels, dates, etc.
```

#### `Issue struct`
New data structure representing a GitHub issue:

```go
type Issue struct {
    Number    int
    Title     string
    State     string
    Body      string
    URL       string
    User      struct { Login string }
    CreatedAt string
    UpdatedAt string
    Labels    []struct { Name string }
}
```

### 2. Command-Line Tool
**Location:** `cmd/github-issues/main.go` and `cmd/github-issues/README.md`

A ready-to-use command-line tool for checking GitHub issues.

**Features:**
- View open/closed/all issues
- Sort by created, updated, or comments
- Filter by labels
- Customize display limit
- Works with any GitHub repository

**Usage Examples:**

```bash
# View open issues (default)
go run cmd/github-issues/main.go

# View closed issues, sorted by updates
go run cmd/github-issues/main.go -state closed -sort updated

# View issues with specific labels
go run cmd/github-issues/main.go -labels "bug,enhancement"

# View issues from different repository
go run cmd/github-issues/main.go -owner golang -repo go

# View with custom limit
go run cmd/github-issues/main.go -limit 50
```

**Output:**
```
🔍 Checking GitHub issues for ollama/ollama (state: open)...

📋 Found 15 issues:

     #              TITLE                STATE     AUTHOR              UPDATED
   ---                 ---                 ---        ---                ---
  1234  Fix memory leak in model loader   open   john-doe          2026-04-17
  1233  Add support for new model format  open   jane-smith        2026-04-16
  
🌐 View on GitHub: https://github.com/ollama/ollama/issues?state=open
```

### 3. Unit Tests
**File:** `internal/secrets/github_test.go`

Added tests for:
- Token parsing (`TestTokenFromString`)
- Client initialization (`TestNewClient`, `TestNewClientWithURL`)
- Issue options validation (`TestIssueListOptions`)

### 4. Updated Documentation

#### Main Documentation
**File:** `docs/secrets.md`

Added new sections:
- Complete GitHub issue listing examples  
- Command-line tool usage guide
- Issue options reference
- Issue data structure documentation
- API examples showing all GitHub client methods

#### Package Documentation  
**File:** `internal/secrets/README.md`

Updated GitHub Client section with:
- Issue listing capability
- Issue retrieval capability
- Full API examples with issue methods

#### Tool Documentation
**File:** `cmd/github-issues/README.md`

Comprehensive guide including:
- Prerequisites (GitHub token setup)
- Basic and advanced usage examples
- Full option reference
- Output format documentation
- Troubleshooting guide
- Integration information

## File Structure

```
/home/coder/ollama/
├── cmd/github-issues/
│   ├── main.go           # Command-line tool
│   └── README.md         # Tool documentation
├── internal/secrets/
│   ├── github.go         # (Updated with Issue methods)
│   ├── github_test.go    # (Updated with Issue tests)
│   └── README.md         # (Updated documentation)
└── docs/
    └── secrets.md        # (Updated with issue examples)
```

## How It Works

### Basic Workflow

1. **Authenticate**
   ```bash
   export OLLAMA_GITHUB_TOKEN=ghp_xxxxx
   ```

2. **Create Client**
   ```go
   client := github.NewClient(token)
   ```

3. **List Issues**
   ```go
   issues, err := client.ListIssues(ctx, "owner", "repo", &github.IssueListOptions{
       State: "open",
   })
   ```

4. **Process Issues**
   ```go
   for _, issue := range issues {
       fmt.Printf("#%d: %s\n", issue.Number, issue.Title)
   }
   ```

### Use Cases

- **Monitor repository issues** - Check project status regularly
- **Filter by priority** - Use labels to find critical issues
- **Track progress** - Sort by updated date to see recent activity
- **Integrate with CI/CD** - Automate issue checking in workflows
- **Generate reports** - Collect and analyze issue data

## Key Capabilities

✅ List repository issues with multiple filtering options  
✅ Retrieve specific issue details  
✅ Sort by creation, updates, or comments  
✅ Filter by issue state (open, closed, all)  
✅ Filter by labels (single or multiple)  
✅ Paginate through large result sets  
✅ Access issue metadata (author, timestamps, body, etc.)  
✅ Command-line tool for quick checks  
✅ Programmatic API for integration  

## Integration Points

### With Existing Features
- Uses GitHub client from the secrets package
- Works with `OLLAMA_GITHUB_TOKEN` environment variable
- Can retrieve token from GSM if configured
- Integrates with existing GitHub Enterprise support

### With CI/CD
```yaml
# GitHub Actions example
- name: Check open issues
  env:
    OLLAMA_GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  run: go run cmd/github-issues/main.go -limit 20
```

## Examples

### Get Open Issues Sorted by Recent Updates

```go
client := github.NewClient(token)
issues, _ := client.ListIssues(ctx, "ollama", "ollama", &github.IssueListOptions{
    State: "open",
    Sort:  "updated",
    Order: "desc",
})
```

### Find Issues with Specific Labels

```go
issues, _ := client.ListIssues(ctx, "ollama", "ollama", &github.IssueListOptions{
    State:  "open",
    Labels: "bug,help-wanted",
})
```

### Check Latest Comments on Open Issues

```go
issues, _ := client.ListIssues(ctx, "ollama", "ollama", &github.IssueListOptions{
    State: "open",
    Sort:  "comments",
    Order: "desc",
})
```

### Monitor Specific Issue

```go
issue, _ := client.GetIssue(ctx, "ollama", "ollama", 1234)
fmt.Printf("Issue #%d: %s (%s)\n", issue.Number, issue.Title, issue.State)
fmt.Printf("Author: %s\n", issue.User.Login)
fmt.Printf("Created: %s\n", issue.CreatedAt)
fmt.Printf("Labels: %v\n", issue.Labels)
```

## Security & Best Practices

✅ Issues are public data (no security risk in listing)  
✅ Token authentication protects against rate limiting  
✅ Supports GitHub Enterprise Server  
✅ Proper error handling for API failures  
✅ No credentials logged or exposed  

## Rate Limiting

- **Unauthenticated:** 60 requests/hour
- **Authenticated:** 5,000 requests/hour

Use the provided tools during development—GitHub's generous rate limits ensure smooth operation.

## Testing the Implementation

```bash
# Set up your token
export OLLAMA_GITHUB_TOKEN=ghp_xxxxx

# Test the command-line tool
go run cmd/github-issues/main.go

# Test with options
go run cmd/github-issues/main.go -state all -limit 50 -sort updated

# Run unit tests
go test -v ./internal/secrets/...
```

## Troubleshooting

**"OLLAMA_GITHUB_TOKEN not set"**
```bash
export GITHUB_TOKEN=ghp_xxxxx
```

**"GitHub API error: status 401"**
- Token expired or invalid
- Generate new token at https://github.com/settings/tokens

**"GitHub API error: status 403"**
- Rate limiting (check X-RateLimit-Remaining header)
- Wait before making more requests

**"No issues found"**
- Try with `-state all` to check closed issues
- Verify repository name and owner
- Check repository has issues enabled

## Next Steps

1. **Try the tool:**
   ```bash
   go run cmd/github-issues/main.go
   ```

2. **Integrate into your workflow:**
   - Add to CI/CD pipelines
   - Create monitoring scripts
   - Build custom dashboards

3. **Extend functionality:**
   - Add support for pull requests
   - Create issues programmatically
   - Comment on issues
   - Update issue state

## Related Documentation

- [Main Secrets Guide](../docs/secrets.md)
- [GitHub Issues Tool README](../cmd/github-issues/README.md)
- [Secrets Package README](../internal/secrets/README.md)
- [GitHub API Documentation](https://docs.github.com/en/rest/issues)

---

**Status:** ✅ COMPLETE

GitHub issues integration is now fully operational and documented. You can now check, filter, and manage GitHub issues programmatically and from the command line.
