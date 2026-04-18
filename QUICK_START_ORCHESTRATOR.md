# Quick Start: GitHub Issue Orchestration

## 5-Minute Setup

### 1. Get GitHub Token
```bash
# Option A: Token exists
export GITHUB_TOKEN="ghp_your_token_here"

# Option B: Create new token
# Go to: https://github.com/settings/tokens
# Click "Generate new token (classic)"
# Select scopes: repo, write:gists
# Copy token and run:
export GITHUB_TOKEN="ghp_..."
```

### 2. Test Repository Access
```bash
cd /home/coder/ollama

# Quick test
python3 << 'EOF'
from cmd.github_issues.github_api_handler import GitHubAPIHandler

gh = GitHubAPIHandler("ollama/ollama")
accessible, msg = gh.check_repo_access()
print(f"{'✅' if accessible else '❌'} {msg}")
EOF
```

### 3. Fetch Issues (Dry Run)
```bash
# Just analyze, don't update
python3 cmd/github-issues/orchestrator_enhanced.py \
  --max-issues 5 \
  --severity high \
  --output demo_report.json

# View results
cat .github/orchestrator_report.json | jq '.issues_processed'
```

### 4. Execute Updates (Live)
```bash
# Actually add labels and comments to GitHub issues
python3 cmd/github-issues/orchestrator_enhanced.py \
  --max-issues 3 \
  --severity high \
  --execute \
  --output execution_report.json
```

## Command Reference

### Fetch & Analyze Only (Dry-Run)
```bash
python3 cmd/github-issues/orchestrator_enhanced.py \
  --repo ollama/ollama \
  --max-issues 20 \
  --severity high
```

**Output**: Reports + AI analysis (no GitHub updates)

### Execute with GitHub Updates
```bash
python3 cmd/github-issues/orchestrator_enhanced.py \
  --repo ollama/ollama \
  --max-issues 10 \
  --severity high \
  --token "$GITHUB_TOKEN" \
  --execute
```

**Output**: Reports + Labels + Comments on GitHub issues

### All Severity Levels
```bash
# High Priority
python3 cmd/github-issues/orchestrator_enhanced.py --severity high

# Medium Priority  
python3 cmd/github-issues/orchestrator_enhanced.py --severity medium

# Low Priority
python3 cmd/github-issues/orchestrator_enhanced.py --severity low
```

## What Gets Updated on GitHub

When `--execute` is enabled, the orchestrator:

✅ **Adds Labels**
- `priority/high`, `priority/medium`, `priority/low`
- `bug`, `enhancement`, `documentation`
- `status/triaged`

✅ **Posts AI Analysis Comment**
```
## 🤖 Automated Triage & Analysis Report

Issue: #15649
Analyzed: 2026-04-17T19:50:14Z

### Analysis
- Type: root_cause
- Severity: medium  
- Components: unknown

### Implementation Plan
- Effort: medium
- Steps:
  1. identify root cause
  2. implement fix
  3. write tests

This issue has been triaged and marked for implementation.
```

❌ **Does NOT Auto-Close** (safety feature)
- Prevents accidental closure
- Requires manual verification first

## Workflow Example

### Day 1: Analyze
```bash
# Get report on all HIGH priority issues
python3 cmd/github-issues/orchestrator_enhanced.py \
  --max-issues 20 \
  --severity high \
  --output day1_analysis.json

# Review which issues need fixing
jq '.issues_processed[] | {number, severity, type, title}' day1_analysis.json
```

### Day 2: Label & Comment
```bash
# Update top 5 issues with analysis and labels
python3 cmd/github-issues/orchestrator_enhanced.py \
  --max-issues 5 \
  --severity high \
  --execute \
  --output day2_execution.json

# Verify uploads succeeded
jq '.execution_results[].actions[] | select(.success==true)' day2_execution.json
```

### Day 3: Track Progress
```bash
# Check state transitions
jq '.state_transitions' day2_execution.json

# Verify labels on GitHub
curl -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/ollama/ollama/issues/15649/labels" | jq '.[] | .name'
```

## Troubleshooting

### ❌ "Bad credentials"
```bash
# Check token
echo "Token: $GITHUB_TOKEN"

# If empty, set it
export GITHUB_TOKEN="ghp_..."

# If invalid, regenerate at: https://github.com/settings/tokens
```

### ❌ "Repository not found"
```bash
# Verify repo name and access
curl -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/ollama/ollama"

# Should return 200 with repo details
```

### ❌ "Rate limit exceeded"
```bash
# Check remaining requests
python3 << 'EOF'
from cmd.github_issues.github_api_handler import GitHubAPIHandler
gh = GitHubAPIHandler("ollama/ollama", "$GITHUB_TOKEN")
print(gh.get_rate_limit_status())
EOF

# Limit: 5,000 requests/hour
# Or use GitHub App for higher limits
```

### ❌ "Invalid state transition"
```bash
# Valid transitions from 'new': only 'triaged'
# Check state_chart.STATES in immutable_state.py

# To see current state of an issue:
python3 << 'EOF'
from cmd.github_issues.immutable_state import IssueStateChart
chart = IssueStateChart(15649)
print(f"Current state: {chart.current_state}")
print(f"Valid next states: {chart.STATES[chart.current_state]['next']}")
EOF
```

## Output Files

### `.github/orchestrator_report.json`
Main execution report with:
- Total issues processed
- Execution results per issue
- State transitions
- Pipeline status

### `.github/triage_snapshot.json`
Triage decisions with:
- Issue classification
- Content hashes (for verification)
- Timestamp

### `.github/ai_execution_log.json`
AI analysis details:
- All 4 execution phases
- Input/output data
- Execution status

## Real-World Usage

### Continuous Issue Management (GitHub Actions)
```yaml
# .github/workflows/issue-orchestrator.yml
name: Daily Issue Triage
on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM UTC daily

jobs:
  triage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Triage & Analyze Issues
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          python3 cmd/github-issues/orchestrator_enhanced.py \
            --max-issues 30 \
            --severity high \
            --execute
      
      - name: Upload Report
        uses: actions/upload-artifact@v3
        with:
          name: issue-reports
          path: .github/*_report.json
```

### Team Workflow
```bash
# Monday: Triage all open issues
python3 cmd/github-issues/orchestrator_enhanced.py \
  --max-issues 100 \
  --output monday_full_triage.json

# Wednesday: Execute HIGH priority analysis
python3 cmd/github-issues/orchestrator_enhanced.py \
  --max-issues 20 \
  --severity high \
  --execute

# Friday: Review and close resolved issues
python3 cmd/github-issues/orchestrator_enhanced.py \
  --max-issues 50 \
  --severity low
```

## Safety Features

🔒 **Built-in Safety:**
1. **Dry-Run Default**: Must use `--execute` to update GitHub
2. **No Auto-Close**: Comments only, requires manual closure
3. **Rate Limiting**: Respects GitHub's 5,000 req/hour limit
4. **Audit Logs**: All actions tracked and timestamped
5. **Immutable Records**: Cannot modify past decisions

⚠️ **Before First --execute Run:**
```bash
# 1. Test with dry-run first
python3 cmd/github-issues/orchestrator_enhanced.py \
  --max-issues 3 \
  --severity high

# 2. Review the report
cat .github/orchestrator_report.json

# 3. Check target issues on GitHub
open "https://github.com/ollama/ollama/issues?labels=bug"

# 4. Only then enable --execute
```

## Next Steps

1. ✅ Export GITHUB_TOKEN
2. ✅ Test with `python3 orchestrator_enhanced.py --max-issues 5`
3. ✅ Review `.github/orchestrator_report.json`
4. ✅ Run with `--execute` for real updates
5. ✅ Set up GitHub Actions for automation
6. ✅ Customize `.github/triage.rules.json` for your repos

---

**Questions?** Check `ORCHESTRATOR_GUIDE.md` for detailed documentation.
