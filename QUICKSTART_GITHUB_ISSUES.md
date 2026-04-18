# Quick Start: Check GitHub Issues

Now that GitHub integration is fully implemented, here's how to check issues immediately.

## 1. Set Your GitHub Token

```bash
# Get a token at https://github.com/settings/tokens
export OLLAMA_GITHUB_TOKEN=ghp_xxxxx
```

## 2. Check Issues Right Now

### Using the Command-Line Tool

```bash
# View open issues for kushin77/ollama
cd /home/coder/ollama
go run cmd/github-issues/main.go
```

### Or Check Any Repository

```bash
# Check Python project
go run cmd/github-issues/main.go -owner python -repo cpython

# Check TypeScript
go run cmd/github-issues/main.go -owner microsoft -repo typescript

# Check any repo you want
go run cmd/github-issues/main.go -owner OWNER -repo REPO
```

## 3. Use Advanced Filtering

```bash
# View recently updated closed issues
go run cmd/github-issues/main.go -state closed -sort updated -limit 30

# Find bug reports
go run cmd/github-issues/main.go -labels bug -limit 10

# See issues needing help
go run cmd/github-issues/main.go -labels "help wanted" -sort comments

# All issues, most commented first
go run cmd/github-issues/main.go -state all -sort comments -limit 50
```

## 4. Use in Your Code

```go
package main

import (
    "context"
    "fmt"
    "github.com/ollama/ollama/internal/secrets/github"
)

func main() {
    ctx := context.Background()
    client := github.NewClient("ghp_xxxxx")
    
    // List open issues
    issues, _ := client.ListIssues(ctx, "ollama", "ollama", &github.IssueListOptions{
        State: "open",
    })
    
    for _, issue := range issues {
        fmt.Printf("#%d: %s\n", issue.Number, issue.Title)
    }
}
```

## 5. Available Options

```
  -owner string
        Repository owner (default "ollama")
  -repo string
        Repository name (default "ollama")
  -state string
        Issue state: open, closed, all (default "open")
  -sort string
        Sort by: created, updated, comments (default "created")
  -order string
        Order: asc, desc (default "desc")
  -limit int
        Number of issues to display (default 20)
  -labels string
        Filter by labels (comma-separated)
```

## 6. Troubleshooting

### Token not set?
```bash
export OLLAMA_GITHUB_TOKEN=$(read -sp "GitHub Token: " && echo $REPLY)
```

### Want to use GSM instead?
```bash
# Store in Google Secret Manager
gcloud secrets create github-token --data-file=- <<< "ghp_xxxxx"

# Export from GSM
export OLLAMA_GITHUB_TOKEN=$(gcloud secrets versions access latest --secret=github-token)

# Now use the tool
go run cmd/github-issues/main.go
```

### No issues showing?
```bash
# Try viewing all issues
go run cmd/github-issues/main.go -state all

# Or check a different repo
go run cmd/github-issues/main.go -owner golang -repo go
```

## 7. Real-World Examples

```bash
# Check all open Ollama issues
go run cmd/github-issues/main.go

# Find Python bugs
go run cmd/github-issues/main.go -owner python -repo cpython -labels type:bug -state open

# See recently updated Kubernetes issues
go run cmd/github-issues/main.go -owner kubernetes -repo kubernetes -sort updated -state open

# Find help-wanted issues in rust
go run cmd/github-issues/main.go -owner rust-lang -repo rust -labels "E-help-wanted" -state open

# Check issues needing documentation
go run cmd/github-issues/main.go -owner ollama -repo ollama -labels documentation -limit 50
```

## 8. What You Can Do Now

✅ View open/closed/all issues  
✅ Filter by labels  
✅ Sort by creation, updates, or comments  
✅ Check any GitHub repository  
✅ Use in your own Go programs  
✅ Integrate into CI/CD workflows  
✅ Generate issue reports  

## Next Steps

1. **Explore the tool:**
   ```bash
   go run cmd/github-issues/main.go -help
   ```

2. **Read the full documentation:**
   - Overview: `GITHUB_ISSUES_INTEGRATION.md`
   - Tool guide: `cmd/github-issues/README.md`
   - API docs: `docs/secrets.md`

3. **Try in your workflow:**
   - Add to build scripts
   - Integrate with monitoring
   - Create custom reports

---

**Start checking issues now:**

```bash
export OLLAMA_GITHUB_TOKEN=ghp_xxxxx
cd /home/coder/ollama
go run cmd/github-issues/main.go
```
