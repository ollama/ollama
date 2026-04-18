# Checking Ollama Repository Issues

Two simple tools are now available to check open issues in the kushin77/ollama repository.

## Quick Start

### Prerequisites
Get a GitHub personal access token:
1. Visit https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Select minimal scope: `public_repo` (read-only access)
4. Copy the token

### Option 1: Python Script (Recommended)

```bash
# Using command line argument
python3 check-issues.py ghp_xxxxxxxxxxxxx

# Using environment variable
export OLLAMA_GITHUB_TOKEN=ghp_xxxxxxxxxxxxx
python3 check-issues.py
```

### Option 2: Bash Script

```bash
# Using command line argument
./check-issues.sh ghp_xxxxxxxxxxxxx

# Using environment variable
export OLLAMA_GITHUB_TOKEN=ghp_xxxxxxxxxxxxx
./check-issues.sh
```

## What You'll See

The report shows:
- **Total open issues count**
- **Top 15 most recently updated issues** (with #, title, author, date)
- **Top issue labels** with counts
- **Priority breakdown** (bugs, features, docs, help wanted)
- **Direct GitHub links**

Example output:
```
📊 Ollama Repository Issues Report
===================================

📈 Summary Statistics
─────────────────────
Total Open Issues: 42

⏰ Most Recently Updated Issues (Top 15)
─────────────────────────────────────────
#1234   Fix memory leak in model loader @john-doe    2026-04-17
#1233   Add support for new model format @jane-smith 2026-04-16

🏷️  Top Issue Labels
─────────────────
  bug                20 issues
  enhancement        15 issues
  documentation      8 issues

⚠️  Issues by Priority
────────────────────
  🐛 Bug reports: 20
  ✨ Feature requests: 15
  📚 Documentation needed: 8
  🤝 Help wanted: 3

✅ Links
─────
  View all issues: https://github.com/ollama/ollama/issues
  Repository: https://github.com/ollama/ollama

✓ Report generated successfully
```

## Using with GSM (Google Secret Manager)

If you've stored your GitHub token in GSM:

```bash
# Retrieve token from GSM and use with Python script
export OLLAMA_GITHUB_TOKEN=$(gcloud secrets versions access latest --secret=github-token)
python3 check-issues.py

# Or with bash script
export OLLAMA_GITHUB_TOKEN=$(gcloud secrets versions access latest --secret=github-token)
./check-issues.sh
```

## Integration

### In CI/CD Workflows

**GitHub Actions:**
```yaml
- name: Check Ollama Issues
  env:
    OLLAMA_GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  run: python3 check-issues.py
```

**Google Cloud Build:**
```yaml
steps:
  - name: 'python:3.11'
    env:
      - 'OLLAMA_GITHUB_TOKEN=$_GITHUB_TOKEN'
    entrypoint: 'python3'
    args:
      - 'check-issues.py'
```

### As a Scheduled Report

```bash
# Add to crontab to get daily report
0 8 * * * cd /path/to/ollama && python3 check-issues.py > /tmp/ollama-issues.txt 2>&1
```

## Troubleshooting

### "Error 401: Bad credentials"
- Token is invalid or expired
- Check token has not been revoked
- Generate a new token

### "Error 403: API rate limit exceeded"
- You've made too many requests
- Wait 1 hour for limit reset
- Use authentication (token) for higher limits

### "Connection refused"
- No internet connection
- GitHub API is down (check status.github.com)

### "Token not provided"
Make sure to set the token either:
- As command line: `python3 check-issues.py ghp_xxxxx`
- As environment: `export OLLAMA_GITHUB_TOKEN=ghp_xxxxx`

## Architecture

These tools use the GitHub API integration we built earlier:
- Leverage `OLLAMA_GITHUB_TOKEN` environment variable
- Can retrieve token from GSM if configured
- Talk directly to GitHub API (no additional dependencies)

## Files

- `check-issues.py` - Python implementation (preferred, more reliable)
- `check-issues.sh` - Bash implementation (lightweight, shell-based)

Both produce identical reports.

## See Also

- [GitHub Issues Integration](GITHUB_ISSUES_INTEGRATION.md)
- [Quick Start Guide](QUICKSTART_GITHUB_ISSUES.md)
- [Full Documentation](docs/secrets.md)
