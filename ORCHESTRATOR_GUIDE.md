#!/usr/bin/env python3
"""
Complete Issue Orchestration System - Comprehensive Guide
Triage → Analysis → Execution → GitHub Updates

This guide explains how to use the complete pipeline for intelligent issue management.
"""

# COMPLETE ISSUE ORCHESTRATION SYSTEM
# ====================================

## Overview

The Issue Orchestrator is a complete end-to-end automation system that:
1. **Fetches** issues from GitHub (kushin77/ollama or fallback)
2. **Triages** issues by severity and type  (IaC-based rules)
3. **Analyzes** issues with multi-AI systems (Claude, Grok, Gemini)
4. **Plans** fixes with detailed implementation steps
5. **Tracks** state transitions (immutable audit logs)
6. **Updates** issues on GitHub (labels, comments, closure)

## Architecture Components

### 1. GitHub API Handler (`github_api_handler.py`)
Manages all GitHub API operations:
- Repository access verification
- Issue fetching and filtering
- Label management (add/remove)
- Issue comments (with markdown support)
- State updates (open/closed)
- Rate limit tracking

**Key Classes:**
- `GitHubAPIHandler`: Direct GitHub API interactions
- `IssueExecutionHandler`: High-level issue operations

**Example Usage:**
```python
from github_api_handler import GitHubAPIHandler, IssueExecutionHandler

# Initialize
gh = GitHubAPIHandler("kushin77/ollama", token="ghp_...")

# Check access
accessible, msg = gh.check_repo_access()
print(f"✅ {msg}")

# Fetch issues
issues = gh.fetch_issues(state='open', per_page=50)

# Add labels and comment
gh.add_label(15649, ['priority/high', 'bug/startup'])
gh.add_comment(15649, "Automated analysis: Root cause identified")

# Close issue
gh.update_issue_state(15649, 'closed')
```

### 2. Triage System (`triage.py`)
Classifies issues by severity and type using IaC rules:
- Declarative rule engine (.github/triage.rules.json)
- Severity levels: CRITICAL, HIGH, MEDIUM, LOW
- Issue types: BUG, FEATURE, REFACTOR, DOCUMENTATION, SECURITY, PERFORMANCE
- Immutable audit logs of all triage decisions
- Content-addressable issue IDs (SHA256)

**Key Classes:**
- `IssueSeverity`: Enum for severity levels
- `IssueType`: Enum for issue types  
- `TriageIssue`: Immutable issue record with content hash
- `TriageSystem`: Rule-based classification engine

**Example Usage:**
```python
from triage import TriageSystem, TriageIssue, IssueSeverity

# Initialize
triage = TriageSystem()

# Convert GitHub issue to TriageIssue
triage_issue = TriageIssue(
    issue_num=15649,
    title="Ollama startup issue",
    body="Application fails to start on macOS",
    labels=["bug", "startup"],
    author="user123",
    created="2026-04-17T12:00:00Z",
    comments=5
)

# Triage
result = triage.triage(triage_issue)
print(f"Severity: {result['severity']}")  # "high"
print(f"Type: {result['type']}")          # "bug"
```

### 3. AI Executor Framework (`ai_executor.py`)
Multi-AI analysis with consensus and immutable records:
- Abstract `AIExecutor` base class
- Implementations: Claude, Grok, Gemini, GPT-4
- 4-phase workflow: Analysis → Planning → Implementation → Validation
- Immutable execution records with content-addressing
- Multi-AI consensus mechanism

**Key Classes:**
- `ExecutionRecord`: Immutable record of AI execution
- `AIExecutor`: Abstract interface for AI systems
- `ClaudeExecutor`, `GrokExecutor`, `GeminiExecutor`: Implementations
- `MultiAIExecutor`: Consensus across multiple AIs

**Example Usage:**
```python
from ai_executor import ClaudeExecutor, MultiAIExecutor, AIProvider

# Single AI
claude = ClaudeExecutor(AIProvider.CLAUDE)
result = claude.execute_workflow({
    'number': 15649,
    'title': 'Issue title',
    'body': 'Issue description'
})
print(result['status'])  # "completed"

# Multi-AI consensus
multi = MultiAIExecutor([AIProvider.CLAUDE, AIProvider.GROK])
consensus = multi.execute_parallel(issue)
print(consensus['consensus']['consensus_status'])  # "completed"
```

### 4. Immutable State Management (`immutable_state.py`)
Append-only state tracking with content-addressed records:
- Finite state machine for issue resolution
- State transitions: new → triaged → assigned → in_progress → testing → validation → closed
- Append-only audit logs
- State history tracking per issue

**Key Classes:**
- `ImmutableState`: Content-addressed state record
- `StateTransition`: Immutable state change record
- `IssueStateChart`: FSM for issue workflow

**Example Usage:**
```python
from immutable_state import IssueStateChart

# Create FSM for issue
state_chart = IssueStateChart(issue_num=15649)

# Transition states
state_chart.transition('triaged')      # new → triaged ✓
state_chart.transition('assigned')     # triaged → assigned ✓
state_chart.transition('in_progress')  # assigned → in_progress ✓

# Invalid transitions are rejected
state_chart.transition('new')  # ❌ Invalid
```

### 5. Orchestrator (`orchestrator_enhanced.py`)
Complete pipeline orchestration:
- Automatic repository selection (primary/fallback)
- Sequential pipeline: fetch → triage → analyze → execute
- Severity-based filtering (HIGH/MEDIUM/LOW)
- Dry-run mode for safe testing
- Comprehensive JSON reports

**Configuration:**
```bash
python3 orchestrator_enhanced.py \
  --repo kushin77/ollama \           # Primary repo
  --fallback kushin77/ollama \         # Fallback repo
  --token ghp_xxx \                  # GitHub token
  --max-issues 20 \                  # Max issues to process
  --severity high \                  # Target severity (high/medium/low)
  --execute \                        # Actually update GitHub (default: dry-run)
  --output report.json               # Output report file
```

## Complete Workflow Example

### Step 1: Setup GitHub Token
```bash
export GITHUB_TOKEN="ghp_..."
# Or store in .env file (add to .gitignore):
# GITHUB_TOKEN=ghp_...
```

### Step 2: Run Triage & Analysis (Dry-Run)
```bash
cd /home/coder/ollama

# Fetch, triage, and analyze HIGH priority issues (no GitHub updates)
python3 cmd/github-issues/orchestrator_enhanced.py \
  --max-issues 10 \
  --severity high \
  --output analysis_report.json

# Review the report
cat .github/orchestrator_report.json | jq '.issues_processed'
```

### Step 3: Execute Updates (Live)
```bash
# Actually update GitHub issues with labels and comments
python3 cmd/github-issues/orchestrator_enhanced.py \
  --max-issues 5 \
  --severity high \
  --execute \
  --output execution_report.json
```

## Pipeline Outputs

### 1. Main Report (`.github/orchestrator_report.json`)
```json
{
  "timestamp": "2026-04-17T19:50:14Z",
  "repository": "ollama/ollama",
  "total_issues_processed": 2,
  "pipeline_stages": ["fetch", "triage", "analysis", "github_updates"],
  "execution_results": [
    {
      "issue": 15649,
      "timestamp": "2026-04-17T19:50:14Z",
      "actions": [
        {
          "action": "add_labels",
          "success": true,
          "message": "✅ Added labels: priority/high, bug"
        },
        {
          "action": "add_summary_comment",
          "success": true,
          "message": "✅ Comment added (ID: 123456789)"
        }
      ]
    }
  ],
  "state_transitions": {
    "15649": [
      {
        "issue_num": 15649,
        "transition": "new -> triaged",
        "timestamp": "2026-04-17T19:50:14Z"
      }
    ]
  }
}
```

### 2. Triage Snapshot (`.github/triage_snapshot.json`)
Per-issue triage decisions with immutable content hashes

### 3. AI Execution Log (`.github/ai_execution_log.json`)
Complete execution records for each AI phase

### 4. State Transitions
Full audit trail of state changes per issue

## Advanced Usage

### Custom Triage Rules

Edit `.github/triage.rules.json`:
```json
{
  "version": "1.0",
  "rules": [
    {
      "name": "gpu_issues",
      "patterns": ["gpu", "cuda", "mlx", "metal"],
      "severity": "high",
      "auto_actions": ["add_label:gpu", "assign:gpu-team"],
      "labels_to_add": ["area/gpu"]
    }
  ]
}
```

### Multi-Repository Support

```python
# Handle multiple repositories
repos = [
  ("kushin77/ollama", "primary"),
  ("ollama/ollama", "secondary"),
  ("ollama/examples", "tertiary")
]

for repo, priority in repos:
    orchestrator = IssueOrchestrator(repo=repo)
    summary = orchestrator.triage_and_execute(max_issues=10)
```

### Rate Limiting

```python
from github_api_handler import GitHubAPIHandler

gh = GitHubAPIHandler("kushin77/ollama", token)
status = gh.get_rate_limit_status()
print(f"Remaining: {status['remaining']}/{status['limit']}")
print(f"Resets at: {status['reset']}")
```

## Security & Credentials

✅ **Best Practices:**
1. Store GitHub token in `~/.env` or environment variables
2. Add `.env` to `.gitignore`
3. Use read-only tokens when possible
4. Rotate tokens regularly
5. Never commit tokens to version control

❌ **Never:**
- Hardcode tokens
- Share tokens via email/chat
- Log full tokens
- Commit to version control

## Architecture Principles

### Infrastructure as Code (IaC)
- All configurations declared in JSON/YAML
- Triage rules in `.github/triage.rules.json`
- Version controlled and reproducible
- Auditable decision logic

### Immutability
- All states are append-only
- Content-addressed records (SHA256)
- Audit trails are permanent
- No state mutations

### Independence
- Decoupled components
- Pluggable AI backends
- Fallback mechanisms
- Graceful degradation

### Multi-AI Support
- Abstract executor interface
- Consensus mechanism across AIs
- Independent AI analyses
- Ranked recommendations

## Troubleshooting

### Issue: "Repository not found"
```bash
# Check repository access
curl -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/kushin77/ollama

# If 404, repo doesn't exist or token lacks access
# Check: 1) Repo path, 2) Token scopes, 3) Token expiration
```

### Issue: "Bad credentials"
```bash
# Verify token is set
echo $GITHUB_TOKEN

# Regenerate token if needed:
# GitHub → Settings → Developer settings → Personal access tokens
```

### Issue: "Rate limit exceeded"
```bash
# Check current limit
python3 -c "
from github_api_handler import GitHubAPIHandler
gh = GitHubAPIHandler('kushin77/ollama', token)
print(gh.get_rate_limit_status())
"

# Wait for reset or use GitHub App for higher limits (5000 vs 60)
```

### Issue: State transition invalid
```bash
# Valid transitions:
# new → triaged → [assigned, wontfix]
# assigned → in_progress
# in_progress → [testing, blocked]
# blocked → in_progress  
# testing → [validation, in_progress]
# validation → [closed, in_progress]
# closed, wontfix → (terminal)

# Check allowed transitions in immutable_state.py:IssueStateChart.STATES
```

## Performance Considerations

- **Default Rate Limit**: 5,000 requests/hour (authenticated)
- **Issues Per Run**: Start with 5-10, scale to 50+
- **Batch Processing**: Group by severity for efficiency
- **Caching**: Triage rules cached in memory
- **Parallel AI**: Claude + Grok runs independently

## Integration with CI/CD

### GitHub Actions Workflow
```yaml
name: Issue Triage
on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM UTC
  workflow_dispatch:

jobs:
  triage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run Issue Orchestrator
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          python3 cmd/github-issues/orchestrator_enhanced.py \
            --max-issues 20 \
            --severity high \
            --execute
      
      - name: Upload Report
        uses: actions/upload-artifact@v3
        with:
          name: orchestrator-report
          path: .github/orchestrator_report.json
```

## Next Steps

1. **Test with Demo Data**: Run in dry-run mode first
2. **Set Up GitHub Token**: Export GITHUB_TOKEN environment variable
3. **Customize Triage Rules**: Edit `.github/triage.rules.json`
4. **Live Execution**: Enable `--execute` flag for real updates
5. **CI/CD Integration**: Add GitHub Actions workflow
6. **Multi-AI Setup**: Configure Claude/Grok/Gemini API keys

---

**System Status**: ✅ Production Ready
**Last Updated**: 2026-04-17
**Components**: 5 (API Handler, Triage, AI Executor, State Manager, Orchestrator)
**Pipeline Stages**: fetch → triage → analyze → execute
