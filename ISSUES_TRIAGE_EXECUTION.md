# GitHub Issues Triage & Execution Framework

**For kushin77/ollama Repository**

Comprehensive IaC-based, immutable, independent issue management system with multi-AI execution support.

## Overview

This framework provides:

✅ **Intelligent Triage** - Automatic issue classification by severity, type, and effort  
✅ **AI-Agnostic Execution** - Support for Claude, Grok, Gemini, GPT-4, and custom AI systems  
✅ **Immutable State** - Content-addressed, append-only audit trails for compliance  
✅ **IaC Workflows** - Declarative, reproducible, versioned issue management  
✅ **Consensus Building** - Multi-AI agreement mechanisms for high-confidence decisions  
✅ **Decoupled Architecture** - Independent components, no shared mutable state  

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              GitHub Issues Orchestrator                  │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Triage     │  │   AI         │  │    State     │   │
│  │   System     │  │   Executor   │  │  Management  │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                           │
│           ↓           ↓           ↓                       │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Immutable Audit Trail (Append-Only Logs)       │   │
│  │  Content-Addressed Records                       │   │
│  │  SHA256 Hashes for Verification                  │   │
│  └──────────────────────────────────────────────────┘   │
│                                                           │
│           ↓           ↓           ↓                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐         │
│  │ IaC        │  │ State      │  │ Execution  │         │
│  │ Templates  │  │ Snapshots  │  │ Records    │         │
│  └────────────┘  └────────────┘  └────────────┘         │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

## Components

### 1. Triage System (`triage.py`)

Automatic issue classification using declarative rules.

```bash
python cmd/github-issues/triage.py
```

**Features:**
- Priority scoring (0-100)
- Severity assessment (Critical, High, Medium, Low)
- Issue type detection (Bug, Feature, Refactor, etc.)
- Automatic routing and labeling
- Rule-based decision making

**Output:**
- `.github/triage_snapshot.json` - Immutable classification records
- Content hashes for verification

### 2. AI Executor (`ai_executor.py`)

Multi-AI compatible execution framework.

```bash
# Single AI
python cmd/github-issues/ai_executor.py --ai claude

# Multiple AIs (consensus)
python cmd/github-issues/ai_executor.py --ai claude,grok,gemini
```

**Supported AI Systems:**
- **Claude** - Analysis & reasoning
- **Grok** - System understanding & optimization
- **Gemini** - Multimodal context & integration
- **GPT-4** - Extensible framework
- **Custom** - Pluggable architecture

**Execution Phases:**
1. **Analysis** - Root cause investigation
2. **Planning** - Solution design
3. **Implementation** - Code changes
4. **Testing** - Validation & regression
5. **Validation** - Final approval

### 3. Immutable State Management (`immutable_state.py`)

Append-only, content-addressed state tracking.

```bash
python cmd/github-issues/immutable_state.py --issue 15649
```

**Features:**
- Content-addressable IDs (SHA256)
- Append-only transaction logs
- State transition tracking
- Immutable audit trail
- Verifiable checksums

**State Flow:**
```
new → triaged → assigned → in_progress → testing → validation → closed
                 ↓         ↓            ↓       ↓
              blocked ────────────────────────→
```

### 4. Orchestrator (`orchestrator.py`)

Unified command-line interface for all operations.

```bash
# Triage issues
python cmd/github-issues/orchestrator.py triage --limit 10

# Execute fix
python cmd/github-issues/orchestrator.py execute --issue 15649 --ai claude,grok

# Check status
python cmd/github-issues/orchestrator.py status --issue 15649

# Generate report
python cmd/github-issues/orchestrator.py report

# Generate plan
python cmd/github-issues/orchestrator.py plan
```

## Quick Start

### 1. Triage Top Issues

```bash
cd /home/coder/ollama

# Run triage on first 10 issues
python cmd/github-issues/orchestrator.py triage --limit 10
```

Expected output:
```
✓ #15649 | Severity: critical  | Actions: 2
✓ #15648 | Severity: high      | Actions: 1
...
✅ Triaged 10 issues
📄 Snapshot saved to: .github/triage_snapshot.json
```

### 2. Execute AI-Based Fixes

```bash
# Use single AI (Claude)
python cmd/github-issues/orchestrator.py execute --issue 15649 --ai claude

# Use multiple AIs for consensus
python cmd/github-issues/orchestrator.py execute --issue 15649 --ai claude,grok,gemini
```

### 3. Check Status

```bash
python cmd/github-issues/orchestrator.py status --issue 15649
```

Output shows:
- Current state
- State transition history
- AI execution results
- Consensus status

### 4. Generate Reports

```bash
# Issue triage report
python cmd/github-issues/orchestrator.py report

# Implementation plan
python cmd/github-issues/orchestrator.py plan
```

## Configuration

### Triage Rules (`.github/triage.rules.json`)

Declarative IaC-based triage rules:

```json
{
  "rules": [
    {
      "id": "security_critical",
      "labels": ["security"],
      "severity": "critical",
      "auto_actions": [
        "label:security-critical",
        "assign:security-team"
      ]
    }
  ]
}
```

### State Management (`.github/state/`)

Append-only transaction logs:

```
.github/
└── state/
    ├── issue_15649.jsonl        # State transitions
    ├── issue_15648.jsonl        # State transitions
    └── issue_15649_audit.json   # Full audit trail
```

## IaC Principles

### 1. Declarative Configuration

All rules, templates, and workflows are declared, not procedural:

```yaml
# Example: Declarative fix template
apiVersion: ollama.io/v1
kind: IssueResolution
metadata:
  issue_number: 15649
spec:
  fix:
    description: "Fix startup crash"
    changes:
    - path: "cmd/start.go"
      operation: "patch"
```

### 2. Immutability

All state changes are append-only:
- Content-addressed (SHA256)
- Immutable snapshots
- No delete operations
- Full audit trail

### 3. Independence

Components are decoupled:
- No shared mutable state
- Async-safe operations
- Idempotent transactions
- Clear input/output contracts

## Execution Examples

### Example 1: Critical Startup Bug (Issue #15649)

```bash
# 1. Triage
python orchestrator.py triage --limit 1

# Output:
# ✓ #15649 | Severity: critical | Actions: 3
# - Auto-label: critical, urgent
# - Auto-assign: core-team
# - Investigation required

# 2. Use multiple AI systems for consensus
python orchestrator.py execute --issue 15649 --ai claude,grok,gemini

# Output:
# 🤖 Claude: Root cause identified (startup check failing)
# 🤖 Grok: System understanding confirms issue scope
# 🤖 Gemini: Integration points verified
# ✅ Consensus: Fix implementation
# 📋 Recommendation: Proceed with fix

# 3. Track state
python orchestrator.py status --issue 15649

# Output:
# State: in_progress
# Transitions:
#  • 2026-04-17T12:00:00: triaged
#  • 2026-04-17T12:30:00: assigned
#  • 2026-04-17T13:00:00: in_progress
```

### Example 2: MLX GPU Issue (Issue #15648)

```bash
# Triage detects GPU-specific issue
python orchestrator.py triage --limit 1

# Routing: High priority, requires macOS platform testing
# Auto-labels: gpu, mlx, backend, macos

# Execute with AI consensus
python orchestrator.py execute --issue 15648 --ai claude,grok

# Output:
# 🤖 Claude: Analysis indicates libmlx loading issue
# 🤖 Grok: Optimized fix proposed with performance metrics
# ✅ Consensus: Fix with performance improvement
```

## Data Structures

### Triage Record

```json
{
  "content_hash": "abc123def456",
  "issue_num": 15649,
  "title": "Ollama startup issue",
  "severity": "critical",
  "type": "bug",
  "recommended_actions": [
    "label:critical",
    "assign:core-team"
  ],
  "triaged_at": "2026-04-17T12:00:00"
}
```

### Immutable State Record

```json
{
  "issue_num": 15649,
  "state_type": "workflow_state",
  "timestamp": "2026-04-17T12:30:00",
  "content_hash": "xyz789abc123",
  "content": {
    "state": "in_progress",
    "previous_state": "assigned",
    "assigned_to": "team-lead"
  }
}
```

### AI Execution Record

```json
{
  "id": "exec_001",
  "issue_num": 15649,
  "ai_provider": "claude",
  "phase": "analysis",
  "status": "success",
  "content_hash": "def456xyz789",
  "inputs": {...},
  "outputs": {...}
}
```

## AI System Integration

### Adding New AI System

1. Extend `AIExecutor` abstract class:

```python
class NewAIExecutor(AIExecutor):
    def analyze_issue(self, issue: Dict) -> Dict:
        # Implementation
        pass
    
    def generate_fix_plan(self, analysis: Dict, issue: Dict) -> Dict:
        # Implementation
        pass
```

2. Register in factory:

```python
executors = {
    'claude': ClaudeExecutor(),
    'new_ai': NewAIExecutor()
}
```

### Consensus Mechanism

Multiple AI systems vote on solution approach:

```
Claude:  ✓ Fix approach A (confidence: 0.95)
Grok:    ✓ Fix approach A (confidence: 0.90)
Gemini:  ~ Fix approach B (confidence: 0.75)
─────────────────────────────
Consensus: Approach A (2/3 agreement, 92% confidence)
```

## Output Files

```
.github/
├── triage.rules.json              # IaC triage rules
├── triage_snapshot.json           # Triaged issues snapshot
├── ai_execution_log.json          # AI execution records
├── implementation_plan.json       # Multi-phase execution plan
└── state/
    ├── issue_15649.jsonl          # Append-only state log
    ├── issue_15648.jsonl
    └── issue_15649_audit.json     # Full audit trail
```

## Best Practices

### 1. Always Use Multiple AIs for Critical Issues

```bash
# Critical issues
python orchestrator.py execute --issue 15649 --ai claude,grok,gemini

# Medium issues
python orchestrator.py execute --issue 15650 --ai claude,grok

# Low priority
python orchestrator.py execute --issue 15651 --ai claude
```

### 2. Review State Transitions

```bash
# Check full audit trail
cat .github/state/issue_15649_audit.json | jq

# Verify immutability
sha256sum .github/state/issue_15649.jsonl
```

### 3. Aggregate Results

```bash
# Generate comprehensive report
python orchestrator.py report

# Check real-time status
python orchestrator.py status --issue 15649
```

## Troubleshooting

### Issue: Consensus Not Reached

```bash
# Check AI execution logs
cat .github/ai_execution_log.json | jq '.ai_results'

# Run with verbose output
python cmd/github-issues/ai_executor.py --verbose
```

### Issue: State Corruption

All state is immutable and append-only, but verify:

```bash
# Check file integrity
sha256sum -c .github/state/*.hash

# View complete history
tail -20 .github/state/issue_15649.jsonl
```

## Metrics & Monitoring

Track throughout execution:

- **Triage Accuracy** - % of issues correctly classified
- **AI Consensus** - % of agreement across systems
- **Fix Success Rate** - % of issues resolved
- **Time to Resolution** - Average time per severity
- **Regression Rate** - % of fixes causing new issues

## References

- [Triage System](./triage.py)
- [AI Executor](./ai_executor.py)
- [Immutable State](./immutable_state.py)
- [Orchestrator](./orchestrator.py)

---

**Status**: ✅ System ready for multi-AI GitHub issue management  
**Last Updated**: April 17, 2026  
**Repository**: kushin77/ollama
