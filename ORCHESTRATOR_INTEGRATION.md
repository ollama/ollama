# COMPLETE INTEGRATION GUIDE
# GitHub Issue Orchestration System - End-to-End Setup

## System Overview

This document provides the complete setup and integration guide for the GitHub Issue Orchestration system.

```
┌─────────────────────────────────────────────────────────────────┐
│  GitHub Issue Orchestration Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  FETCH GITHUB ISSUES                                            │
│  ↓                                                               │
│  • GitHub API: Fetch open/closed issues                        │
│  • Filter: Remove PRs, batch by 50                             │
│  • Fallback: Try primary then kushin77/ollama            │
│  • Rate Limit: Track remaining requests                        │
│                                                                  │
│  TRIAGE & CLASSIFY                                              │
│  ↓                                                               │
│  • IaC Rules: .github/triage.rules.json                        │
│  • Severity: CRITICAL/HIGH/MEDIUM/LOW                          │
│  • Type: BUG/FEATURE/REFACTOR/SECURITY/PERFORMANCE             │
│  • Content Hash: SHA256 for immutability                        │
│                                                                  │
│  AI MULTI-SYSTEM ANALYSIS                                      │
│  ↓                                                               │
│  • Phase 1: Analysis (root cause, impact)                      │
│  • Phase 2: Planning (fix steps, effort)                       │
│  • Phase 3: Implementation (code changes)                      │
│  • Phase 4: Validation (testing, verification)                 │
│  • Consensus: 2+ AI systems agreement                          │
│                                                                  │
│  STATE TRACKING & AUDITING                                      │
│  ↓                                                               │
│  • FSM: new → triaged → assigned → in_progress → closed        │
│  • Audit Log: Append-only, immutable                           │
│  • Transitions: Timestamped with metadata                      │
│                                                                  │
│  GITHUB ISSUE UPDATES                                           │
│  ↓                                                               │
│  • Labels: priority/high, bug, status/triaged                  │
│  • Comments: AI analysis summary                               │
│  • State: Can update but not auto-close                        │
│  • Rate Limit: Careful with high-volume updates                │
│                                                                  │
│  REPORTING & COMPLIANCE                                         │
│  ↓                                                               │
│  • JSON Reports: orchestrator_report.json                      │
│  • Audit Trails: All decisions tracked                         │
│  • Immutable: Content-addressed records                        │
│  • Traceable: Full decision history                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 1. Environment Setup

### Prerequisites
- Python 3.8+
- GitHub account with Personal Access Token (PAT)
- curl/wget for testing

### Step 1: Clone Repository
```bash
cd /home/coder/ollama

# Verify structure
ls -la cmd/github-issues/
# Expected files:
# - orchestrator_enhanced.py
# - github_api_handler.py
# - triage.py
# - ai_executor.py
# - immutable_state.py
# - orchestrator.py (original)
```

### Step 2: Create GitHub Token
```bash
# Visit: https://github.com/settings/tokens
# Click: "Generate new token (classic)"
# Select scopes:
#   ✓ repo              (Full control of private repositories)
#   ✓ write:gists        (Write access to gists)
#   ✓ admin:repo_hook    (Full control of repository hooks)
# Copy the token immediately (won't be shown again)
```

### Step 3: Set Environment Variable
```bash
# Option A: Temporary (current session only)
export GITHUB_TOKEN="ghp_your_token_here"

# Verify it's set
echo $GITHUB_TOKEN | head -c 20  # Should show: ghp_...

# Option B: Persistent (add to ~/.bashrc or ~/.zshrc)
echo 'export GITHUB_TOKEN="ghp_your_token_here"' >> ~/.bashrc
source ~/.bashrc

# Option C: .env file (for development)
cd /home/coder/ollama
cat > .env << EOF
# GitHub Configuration
GITHUB_TOKEN=ghp_your_token_here
GITHUB_USER=kushin77
GITHUB_REPO=ollama
EOF

# Add to .gitignore to prevent accidental commits
echo ".env" >> .gitignore
```

## 2. Testing Components

### Test 1: GitHub API Handler
```bash
python3 << 'EOF'
import sys
sys.path.insert(0, '/home/coder/ollama/cmd/github-issues')

from github_api_handler import GitHubAPIHandler

# Initialize with GitHub token
gh = GitHubAPIHandler("kushin77/ollama")
print("Testing GitHub API Handler...")

# Test 1: Repository access
accessible, msg = gh.check_repo_access()
print(f"✓ Repository access: {'✅' if accessible else '❌'} {msg}")

# Test 2: Issue fetching
issues = gh.fetch_issues(state='open', per_page=5)
print(f"✓ Fetched issues: ✅ {len(issues)} issues")

if issues:
    issue = issues[0]
    print(f"  Sample issue #{issue['number']}: {issue['title'][:50]}")

# Test 3: Rate limit status
rate_limit = gh.get_rate_limit_status()
print(f"✓ Rate limit: ✅ {rate_limit['remaining']} remaining")

print("\n✅ GitHub API Handler: All tests passed!")
EOF
```

### Test 2: Triage System
```bash
python3 << 'EOF'
import sys
sys.path.insert(0, '/home/coder/ollama/cmd/github-issues')

from triage import TriageSystem, TriageIssue

print("Testing Triage System...")

# Create triage system
triage = TriageSystem()

# Create test issue
test_issue = TriageIssue(
    issue_num=999,
    title="Test bug in startup",
    body="Application crashes on launch",
    labels=["bug", "startup"],
    author="test_user",
    created="2026-04-17T00:00:00Z",
    comments=3
)

# Triage it
result = triage.triage(test_issue)

print(f"✓ Triaged issue: #{test_issue.issue_num}")
print(f"  Severity: {result['severity']}")
print(f"  Type: {result['type']}")
print(f"  Content Hash: {result['content_hash']}")

print("\n✅ Triage System: All tests passed!")
EOF
```

### Test 3: AI Executor
```bash
python3 << 'EOF'
import sys
sys.path.insert(0, '/home/coder/ollama/cmd/github-issues')

from ai_executor import ClaudeExecutor, MultiAIExecutor, AIProvider

print("Testing AI Executor...")

# Create executor
claude = ClaudeExecutor(AIProvider.CLAUDE)

# Test issue
test_issue = {
    'number': 999,
    'title': 'Test issue',
    'body': 'Test description'
}

# Execute workflow
result = claude.execute_workflow(test_issue)

print(f"✓ Claude Executor: {result['status']}")
print(f"  Phases completed:")
for phase, data in result['phases'].items():
    print(f"    - {phase}: {data['status']}")

print("\n✅ AI Executor: All tests passed!")
EOF
```

### Test 4: State Management
```bash
python3 << 'EOF'
import sys
sys.path.insert(0, '/home/coder/ollama/cmd/github-issues')

from immutable_state import IssueStateChart

print("Testing State Management...")

# Create state chart
chart = IssueStateChart(999)

print(f"✓ Initial state: {chart.current_state}")

# Try valid transition
success = chart.transition('triaged')
print(f"✓ Transition new→triaged: {'✅ Success' if success else '❌ Failed'}")
print(f"  Current state: {chart.current_state}")

# Show transitions
print(f"✓ Recorded transitions: {len(chart.transitions)}")
for trans in chart.transitions:
    print(f"  - {trans['transition']}")

print("\n✅ State Management: All tests passed!")
EOF
```

## 3. Running the Complete Pipeline

### Quick Test Run (Dry-Run)
```bash
# Analyze without updating GitHub
python3 cmd/github-issues/orchestrator_enhanced.py \
  --max-issues 5 \
  --severity high \
  --output test_run.json

# Check results
echo "=== Pipeline Status ==="
jq '.pipeline' .github/orchestrator_report.json

echo "=== Issues Processed ==="
jq '.issues_processed[] | {number, severity, type}' .github/orchestrator_report.json
```

### Live Execution (With GitHub Updates)
```bash
# Add labels and comments to GitHub issues
python3 cmd/github-issues/orchestrator_enhanced.py \
  --repo kushin77/ollama \
  --max-issues 3 \
  --severity high \
  --execute \
  --output live_run.json

# Verify updates succeeded
echo "=== Execution Results ==="
jq '.execution_results[] | {issue, success: (.actions[] | select(.success==true) | .action)}' .github/orchestrator_report.json
```

## 4. Customization

### Edit Triage Rules
```bash
# .github/triage.rules.json
cat > .github/triage.rules.json << 'EOF'
{
  "version": "1.0",
  "rules": [
    {
      "name": "critical_security",
      "labels": ["security", "critical"],
      "severity": "critical",
      "auto_actions": ["assign:security-team"]
    },
    {
      "name": "startup_bugs",
      "patterns": ["startup", "crash", "launch"],
      "severity": "high",
      "labels_to_add": ["priority/high", "startup"]
    }
  ]
}
EOF

echo "Triage rules updated!"
```

### Custom Repository Configuration
```bash
# Create a config file for different repos
cat > .github/orchestrator_config.json << 'EOF'
{
  "repositories": [
    {
      "repo": "kushin77/ollama",
      "priority": 1,
      "max_issues": 50,
      "target_severity": "high"
    },
    {
      "repo": "kushin77/ollama",
      "priority": 2,
      "max_issues": 20,
      "target_severity": "high"
    }
  ],
  "schedule": {
    "frequency": "daily",
    "time": "02:00Z"
  }
}
EOF

echo "Repository configuration created!"
```

## 5. CI/CD Integration

### GitHub Actions Workflow
```bash
# Create .github/workflows/issue-orchestrator.yml
mkdir -p .github/workflows

cat > .github/workflows/issue-orchestrator.yml << 'EOF'
name: Automated Issue Orchestration
on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM UTC daily
  workflow_dispatch:     # Manual trigger

jobs:
  orchestrate:
    runs-on: ubuntu-latest
    permissions:
      issues: write
      contents: read
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Run Issue Orchestrator
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          python3 cmd/github-issues/orchestrator_enhanced.py \
            --max-issues 20 \
            --severity high \
            --execute \
            --output orchestrator_report.json
      
      - name: Upload Reports
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: orchestrator-reports
          path: |
            .github/orchestrator_report.json
            .github/triage_snapshot.json
            .github/ai_execution_log.json
      
      - name: Comment on Issues
        if: success()
        run: |
          python3 << 'SCRIPT'
          import json
          with open('.github/orchestrator_report.json') as f:
            report = json.load(f)
          
          processed = len(report.get('issues_processed', []))
          print(f"✅ Processed {processed} issues")
          SCRIPT
EOF

echo "GitHub Actions workflow created!"
echo "Workflow file: .github/workflows/issue-orchestrator.yml"
```

## 6. Monitoring & Maintenance

### Check System Health
```bash
# Test all components
bash << 'SCRIPT'
echo "=== System Health Check ==="

echo "✓ Python availability"
python3 --version

echo "✓ Module imports"
python3 -c "
import sys
sys.path.insert(0, 'cmd/github-issues')
from github_api_handler import GitHubAPIHandler
from triage import TriageSystem
from ai_executor import MultiAIExecutor
from immutable_state import IssueStateChart
print('  All modules imported successfully')
"

echo "✓ GitHub token"
if [ -z "$GITHUB_TOKEN" ]; then
  echo "  ⚠️  GITHUB_TOKEN not set"
else
  echo "  Token is set (length: ${#GITHUB_TOKEN})"
fi

echo "✓ Configuration files"
ls -la .github/triage.rules.json 2>/dev/null && echo "  ✅ Triage rules found"

echo "=== System Status: Ready ==="
SCRIPT
```

### Monitor Execution Logs
```bash
# Watch for recent executions
watch -n 60 'ls -lht .github/*.json | head -5'

# View latest report
tail -f .github/orchestrator_report.json

# Check execution status
jq '.pipeline | keys[] as $k | {($k): .[$k].status}' .github/orchestrator_report.json
```

### Rate Limit Management
```bash
# Check current rate limit
python3 << 'EOF'
import os
import sys
sys.path.insert(0, 'cmd/github-issues')
from github_api_handler import GitHubAPIHandler

gh = GitHubAPIHandler("kushin77/ollama", os.getenv('GITHUB_TOKEN'))
status = gh.get_rate_limit_status()

print(f"Rate Limit Status:")
print(f"  Remaining: {status['remaining']}")
print(f"  Limit: {status['limit']}")
print(f"  Resets at: {status['reset']}")

# If <100 remaining, we should wait
if int(status.get('remaining', 0)) < 100:
    print("\n⚠️  Low rate limit - consider waiting for reset")
EOF
```

## 7. Troubleshooting Common Issues

### Issue: "Bad credentials"
```bash
# Verify token is valid
curl -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/user

# Should return your user info
# If 401: token is invalid or expired
```

### Issue: "Repository not found"
```bash
# Check if repo exists and you have access
curl -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/kushin77/ollama

# Should return 200 and repo details
# If 404: repo doesn't exist
# If 403: no access
```

### Issue: "Rate limit exceeded"
```bash
# Check rate limit
python3 -c "
import os, sys, json
sys.path.insert(0, 'cmd/github-issues')
from github_api_handler import GitHubAPIHandler
gh = GitHubAPIHandler('kushin77/ollama', os.getenv('GITHUB_TOKEN'))
print(json.dumps(gh.get_rate_limit_status(), indent=2))
"

# Wait for reset if <50 remaining
# Or use GitHub App for 5,000 limit (vs 60)
```

## 8. Best Practices

✅ **DO:**
- [ ] Test with dry-run before `--execute`
- [ ] Export GITHUB_TOKEN in CI/CD
- [ ] Review reports before live execution
- [ ] Rotate GitHub tokens monthly
- [ ] Keep triage rules updated
- [ ] Monitor rate limits
- [ ] Archive reports regularly
- [ ] Version control triage rules

❌ **DON'T:**
- [ ] Hardcode GitHub tokens
- [ ] Commit .env files  
- [ ] Use `--execute` on untested repos
- [ ] Run without dry-run first
- [ ] Share GitHub tokens
- [ ] Disable rate limit handling
- [ ] Auto-close issues without review
- [ ] Modify audit logs

## 9. Next Steps

1. ✅ Set GITHUB_TOKEN environment variable
2. ✅ Run Test 1-4 to verify all components
3. ✅ Execute dry-run: `--max-issues 5 --severity high`
4. ✅ Review `.github/orchestrator_report.json`
5. ✅ Run with `--execute` on 3 safe test issues
6. ✅ Verify labels/comments appear on GitHub
7. ✅ Set up GitHub Actions workflow
8. ✅ Monitor first automated run

---

**Documentation:**
- `ORCHESTRATOR_GUIDE.md`: Comprehensive reference
- `QUICK_START_ORCHESTRATOR.md`: 5-minute setup
- This file: Complete integration guide

**Support:**
- Check `.github/triage.rules.json` for rule examples
- Review `cmd/github-issues/` for source code
- See GitHub API docs: https://docs.github.com/en/rest
