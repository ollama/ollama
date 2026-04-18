# How to Check Ollama Issues - Step by Step

## Step 1: Get Your GitHub Token (One Time Only)

1. Go to https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Give it a name like "Ollama Issues Checker"
4. Select scope: `public_repo` (minimum required)
5. Click "Generate token"
6. Copy the token (it starts with `ghp_`)

## Step 2: Check Issues

### Easiest Way: Copy-Paste Command

Replace `YOUR_TOKEN_HERE` with your actual token:

```bash
cd /home/coder/ollama
python3 check-issues.py YOUR_TOKEN_HERE
```

### Or: Set Environment Variable (Reusable)

```bash
export OLLAMA_GITHUB_TOKEN=YOUR_TOKEN_HERE
cd /home/coder/ollama
python3 check-issues.py
```

### Or: Use Bash Script

```bash
export OLLAMA_GITHUB_TOKEN=YOUR_TOKEN_HERE
cd /home/coder/ollama
./check-issues.sh
```

## Step 3: Read the Report

You'll see:
- Total number of open issues
- Most recently updated 15 issues (with number, title, author, update date)
- Labels breakdown (bugs, features, documentation, etc.)
- Priority summary

## Integration with GSM

If you're using Google Secret Manager:

```bash
# Store token in GSM (one time)
gcloud secrets create github-token --data-file=- <<< "ghp_xxxxxxxxxxxxx"

# Use token from GSM
export OLLAMA_GITHUB_TOKEN=$(gcloud secrets versions access latest --secret=github-token)
python3 check-issues.py
```

## Real Examples

### Check open issues right now:
```bash
python3 check-issues.py YOUR_TOKEN
```

### Save report to file:
```bash
OLLAMA_GITHUB_TOKEN=YOUR_TOKEN python3 check-issues.py > issues-report.txt
```

### Email the report:
```bash
OLLAMA_GITHUB_TOKEN=YOUR_TOKEN python3 check-issues.py | mail -s "Ollama Issues Report" you@example.com
```

### Run daily at 9 AM (crontab):
```bash
0 9 * * * cd /home/coder/ollama && OLLAMA_GITHUB_TOKEN=YOUR_TOKEN python3 check-issues.py >> /var/log/ollama-issues.log
```

## What Information You Get

✅ **Total count** of open issues  
✅ **Top 15 recent** issues (sorted by update date)  
✅ **All labels** with issue counts  
✅ **Bug count** - Issues tagged as bugs  
✅ **Feature count** - Enhancement requests  
✅ **Documentation count** - Docs needed  
✅ **Help wanted count** - Community help needed  

## Files Available

- `check-issues.py` - Python script (recommended, most reliable)
- `check-issues.sh` - Bash script (lightweight alternative)
- `CHECK_ISSUES_README.md` - Full documentation

## Quick Answers

**Q: Do I need a token?**
A: Yes. Get one free at https://github.com/settings/tokens

**Q: What scope do I need?**
A: Minimum: `public_repo` (read-only to public repositories)

**Q: Can I use it with GSM?**
A: Yes! Store token in GSM, retrieve it, and set `OLLAMA_GITHUB_TOKEN`

**Q: Will it work without internet?**
A: No, it needs to reach GitHub API. Check your network.

**Q: How often can I run it?**
A: Unlimited with authentication (5000 requests/hour limit with token)

**Q: Can I check other repositories?**
A: Not currently, but you can modify the scripts to use different owner/repo

---

**Ready? Here's the full command:**

```bash
export OLLAMA_GITHUB_TOKEN=ghp_xxxxxxxxxxxxx
python3 /home/coder/ollama/check-issues.py
```

That's it! You'll get a full report of all open issues in kushin77/ollama.
