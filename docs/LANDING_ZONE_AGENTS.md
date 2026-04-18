"""# Landing Zone Agents System

## Overview

The Landing Zone Agents system provides automated governance, compliance tracking,
and repository management for the Ollama project across a hub-and-spoke
repository architecture.

**Two Primary Agents:**

1. **HubSpokeAgent**: Manages repository issues across hub and spoke repositories
2. **PMOAgent**: Ensures Landing Zone compliance and manages 24-label schema

## Architecture

### Hub & Spoke Repository Structure

```
         Hub Repository
        (google/ollama)
              ↑
         ┌────┴────┐
         │          │
    ┌────▼─┐   ┌───▼──┐
    │Team A│   │Team B│  Spoke Repositories
    │Spoke │   │Spoke │  (feature-specific)
    └──────┘   └──────┘
```

### Agent Communication Flow

```
GitHub Issue Events
         ↓
HubSpokeAgent (routes & synchronizes)
         ↓
─────────┼─────────
│        │        │
Hub   Spoke-A  Spoke-B
│        │        │
└────────┼────────┘
         ↓
PMOAgent (validates compliance)
         ↓
CloudSQL (audit trail)
```

## HubSpokeAgent

### Purpose

Routes issues between hub and spoke repositories and keeps them synchronized.

**Responsibilities:**

- Route incoming issues to appropriate repository
- Synchronize hub issues to spoke repositories
- Aggregate spoke updates back to hub
- Escalate critical spoke issues to hub for triage

### Methods

#### `route_issue(issue: RepositoryIssue) -> str`

Routes a new issue to the appropriate repository.

**Routing Logic:**

```
Bug (critical)   → Hub (for triage)
Bug (normal)     → Spoke (assigned team)
Feature          → Spoke (team lead decides)
Refactor         → Spoke (team-specific)
Documentation    → Hub (centralized)
Dependency       → Hub (impacts all)
Infrastructure   → Hub (cross-team)
```

**Example:**

```python
from ollama.agents.hub_spoke_agent import HubSpokeAgent, RepositoryIssue

agent = HubSpokeAgent(context)

issue = RepositoryIssue(
    id="issue-001",
    title="Critical security vulnerability in auth",
    issue_type=IssueType.BUG,
    priority="critical",
    source_repo="spoke-team-a"
)

# Routes to hub for security review
destination = await agent.route_issue(issue)
# Returns: "hub"
```

#### `sync_hub_to_spokes(hub_issue_id: int) -> dict`

Synchronizes hub issue to all spoke repositories.

**Behavior:**

- Creates corresponding issue in each spoke repo
- Maps labels from hub to spoke format
- Tracks synchronization status per repo
- Handles already-synced issues (skips duplicate)

**Example:**

```python
result = await agent.sync_hub_to_spokes(123)
# Returns:
# {
#   'status': 'completed',
#   'synced_repos': ['spoke-team-a', 'spoke-team-b'],
#   'failed_repos': [],
#   'spoke_issue_mapping': {
#     'spoke-team-a': 456,
#     'spoke-team-b': 789
#   }
# }
```

#### `escalate_to_hub(spoke_issue_id: str) -> dict`

Escalates critical spoke issue to hub for cross-team triage.

**When to Use:**

- P0/P1 severity issues affecting multiple teams
- Issues that require architectural changes
- Security vulnerabilities
- Performance regressions

**Example:**

```python
result = await agent.escalate_to_hub("spoke-team-a/issue-123")
# Returns:
# {
#   'status': 'escalated',
#   'hub_issue_id': 999,
#   'escalation_reason': 'P0 performance regression'
# }
```

#### `aggregate_spoke_updates() -> dict`

Pulls all spoke updates back to hub for centralized tracking.

**Aggregates:**

- Issue status changes
- Resolved issues
- New dependencies
- Infrastructure changes

**Example:**

```python
updates = await agent.aggregate_spoke_updates()
# Returns:
# {
#   'aggregation_time': '2026-01-26T14:00:00Z',
#   'total_updates': 12,
#   'spoke_contributions': {
#     'spoke-team-a': 5,
#     'spoke-team-b': 7
#   },
#   'ready_for_merge': ['issue-456', 'issue-789']
# }
```

### Issue Types

```python
class IssueType(str, Enum):
    BUG = "bug"              # Defects and regressions
    FEATURE = "feature"      # New functionality
    DOCUMENTATION = "documentation"  # Doc improvements
    REFACTOR = "refactor"    # Code quality improvements
    DEPENDENCY = "dependency"  # Dependency updates
    INFRASTRUCTURE = "infrastructure"  # Deployment/infra changes
```

### Audit Logging

All HubSpokeAgent actions are logged with:

- **Intent**: What was requested
- **Execution**: How it was performed
- **Result**: What happened

**Example Log:**

```json
{
  "timestamp": "2026-01-26T14:00:00Z",
  "action": "route_issue",
  "intent": {
    "issue_id": "001",
    "issue_type": "bug",
    "priority": "critical"
  },
  "execution": {
    "routing_logic": "critical_bug_to_hub",
    "destination": "hub"
  },
  "result": {
    "status": "routed",
    "destination": "hub"
  }
}
```

## PMOAgent

### Purpose

Ensures Landing Zone compliance and manages 24-label schema across all resources.

**Responsibilities:**

- Validate Landing Zone 8-point mandate
- Enforce 24-label schema on all resources
- Monitor compliance drift
- Generate compliance reports
- Recommend remediation

### Methods

#### `validate_landing_zone_compliance(resource_id: str) -> ComplianceCheckResult`

Validates resource against Landing Zone 8-point mandate.

**Checks:**

1. **Resource Labeling**: All 24 mandatory labels present
2. **Zero Trust Auth**: Workload Identity enabled
3. **CMEK Encryption**: Data encrypted with customer-managed keys
4. **Least-Privilege IAM**: Service accounts minimally scoped
5. **Audit Logging**: Cloud Audit Logs enabled, 7-year retention
6. **Naming Conventions**: Follow `{env}-{app}-{component}` pattern
7. **No Loose Files**: Directory structure enforced (5-level max)
8. **GPG-Signed Commits**: All main branch commits are signed

**Example:**

```python
from ollama.agents.pmo_agent import PMOAgent

agent = PMOAgent(context)

result = await agent.validate_landing_zone_compliance("prod-ollama-api")
# Returns:
# ComplianceCheckResult(
#   check_name="Landing Zone 8-Point Mandate",
#   status=ComplianceStatus.COMPLIANT,
#   details="8/8 checks passed",
#   resources_affected=["prod-ollama-api"]
# )
```

#### `enforce_label_schema(resource_id: str) -> dict`

Enforces 24-label schema on a resource.

**Mandatory Labels (24 Total):**

**Organizational (4):**

- `environment`: production|staging|development|sandbox
- `application`: ollama
- `team`: Team name (e.g., platform, inference)
- `cost_center`: Finance cost center code

**Lifecycle (5):**

- `created_by`: User or system that created resource
- `created_date`: RFC3339 timestamp
- `lifecycle_state`: active|maintenance|sunset
- `teardown_date`: Planned teardown date or "never"
- `retention_days`: How long to keep resource

**Business (4):**

- `product`: ollama
- `component`: api|inference|database|cache|etc
- `tier`: critical|high|medium|low
- `compliance`: fedramp|hipaa|pci|sox|none

**Technical (4):**

- `version`: Semantic version of component
- `stack`: Technology stack (python-3.11-gcp)
- `backup_strategy`: continuous|daily|weekly|none
- `monitoring_enabled`: true|false

**Financial (4):**

- `budget_owner`: Owner name/email
- `project_code`: Billing project code
- `monthly_budget_usd`: Monthly budget amount
- `chargeback_unit`: Department or cost center

**Git (3):**

- `git_repository`: github.com/kushin77/ollama
- `git_branch`: main|develop|feature/\*
- `auto_delete`: true|false

**Example:**

```python
result = await agent.enforce_label_schema("prod-ollama-api")
# Returns:
# {
#   'resource_id': 'prod-ollama-api',
#   'labels_applied': {
#     'environment': 'production',
#     'application': 'ollama',
#     'team': 'platform',
#     ...
#   },
#   'total_labels': 24,
#   'status': 'completed'
# }
```

#### `monitor_compliance_drift() -> dict`

Continuously monitors all resources for compliance drift.

**Detects:**

- Missing labels
- Outdated lifecycle information
- Drift from naming conventions
- IAM permission creep
- Encryption misconfigurations

**Example:**

```python
drift = await agent.monitor_compliance_drift()
# Returns:
# {
#   'timestamp': '2026-01-26T14:00:00Z',
#   'resources_scanned': 25,
#   'compliant': 24,
#   'non_compliant': 1,
#   'drifts_detected': [
#     {
#       'resource_id': 'dev-ollama-cache',
#       'issue': 'missing label: cost_center',
#       'severity': 'high'
#     }
#   ]
# }
```

#### `generate_compliance_report() -> str`

Generates comprehensive compliance report.

**Includes:**

- Landing Zone mandate status (per check)
- Key metrics (resources scanned, compliance %)
- Audit score (0-100)
- Trend analysis
- Remediation recommendations

**Example:**

```python
report = await agent.generate_compliance_report()
# Returns formatted markdown report:
#
# ════════════════════════════════════════════════════════════
#           OLLAMA PMO COMPLIANCE REPORT
# ════════════════════════════════════════════════════════════
#
# Resources Scanned:      25
# Compliant:              25 (100%)
# Non-Compliant:          0 (0%)
# Audit Score:            95/100
# Trend:                  ↗ Improving
```

### Compliance Status Levels

```python
class ComplianceStatus(str, Enum):
    COMPLIANT = "compliant"          # All checks pass
    NON_COMPLIANT = "non_compliant"  # One or more checks fail
    PARTIAL = "partial"              # Some checks pass
    UNKNOWN = "unknown"              # Cannot determine status
```

## Integration with Ollama

### Deployment in Production

**1. GitHub Actions Integration**

```yaml
name: Governance Check
on: [pull_request, push]

jobs:
  pmo-compliance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Validate Landing Zone Compliance
        run: |
          python -m ollama.agents.pmo_agent \
            --validate \
            --resource prod-ollama-api
```

**2. Cloud Scheduler (Daily Compliance Scan)**

```bash
gcloud scheduler jobs create app-engine daily-compliance-check \
  --schedule="0 6 * * *" \
  --http-method=POST \
  --uri=https://ollama.elevatediq.ai/agents/compliance-check \
  --message-body='{}'
```

**3. Cloud Tasks (Async Job Execution)**

```python
from google.cloud import tasks_v2
from ollama.agents.pmo_agent import PMOAgent

async def schedule_compliance_validation():
    """Schedule compliance validation job."""

    agent = PMOAgent(context)

    # Validate all production resources
    for resource_id in get_prod_resources():
        result = await agent.validate_landing_zone_compliance(resource_id)

        if result.status != ComplianceStatus.COMPLIANT:
            # Create alert
            notify_platform_team(result)
```

### Webhook Integration

**GitHub Webhook Triggers HubSpokeAgent:**

```python
@app.post("/webhook/github/issues")
async def handle_issue_event(event: GitHubIssueEvent):
    """Route issue events through HubSpokeAgent."""

    agent = HubSpokeAgent(context)

    issue = RepositoryIssue(
        id=event.issue.number,
        title=event.issue.title,
        issue_type=IssueType(event.issue.labels[0]),
        priority=extract_priority(event.issue.labels),
        source_repo=event.repository.name
    )

    destination = await agent.route_issue(issue)

    # Transfer to destination repo if needed
    if destination != event.repository.name:
        transfer_issue(event.issue, destination)
```

## Usage Examples

### Example 1: Handling Critical Security Bug

```python
async def handle_security_vulnerability():
    """Escalate critical security vulnerability."""

    agent = HubSpokeAgent(context)

    # Bug found in spoke-team-a
    issue = RepositoryIssue(
        id="GHSA-xxxx-yyyy-zzzz",
        title="SQL Injection in query builder",
        issue_type=IssueType.BUG,
        priority="critical",
        source_repo="spoke-team-a"
    )

    # Route to hub for immediate triage
    destination = await agent.route_issue(issue)
    # Result: "hub" (automatically routed due to criticality)

    # Escalate for cross-team coordination
    escalated = await agent.escalate_to_hub("spoke-team-a/issue-123")

    # All teams notified via hub issue
    print(f"Escalated to hub issue #{escalated['hub_issue_id']}")
```

### Example 2: Compliance Validation on Deployment

```python
async def validate_before_deployment(resource_id: str):
    """Ensure resource is Landing Zone compliant."""

    pmo_agent = PMOAgent(context)

    # Validate resource
    result = await pmo_agent.validate_landing_zone_compliance(resource_id)

    if result.status == ComplianceStatus.NON_COMPLIANT:
        # Enforce labels if missing
        labels_result = await pmo_agent.enforce_label_schema(resource_id)
        print(f"Applied {len(labels_result['labels_applied'])} labels")

        # Revalidate
        result = await pmo_agent.validate_landing_zone_compliance(resource_id)

    if result.status == ComplianceStatus.COMPLIANT:
        proceed_with_deployment()
    else:
        raise ComplianceError(f"Resource not compliant: {result.details}")
```

### Example 3: Periodic Compliance Monitoring

```python
async def daily_compliance_check():
    """Run daily compliance check."""

    pmo_agent = PMOAgent(context)

    # Monitor for drift
    drift = await pmo_agent.monitor_compliance_drift()

    if drift['non_compliant'] > 0:
        # Generate report
        report = await pmo_agent.generate_compliance_report()

        # Alert platform team
        send_slack_notification(
            channel="#platform-alerts",
            message=f"Compliance issues found:\n{report}"
        )

    # Audit trail
    log_compliance_check(drift)
```

## Testing Agents

Run tests:

```bash
# Unit tests
pytest tests/unit/test_agents.py -v

# Integration tests
pytest tests/integration/test_agents.py -v

# With coverage
pytest tests/integration/test_agents.py -v --cov=ollama.agents
```

## Troubleshooting

### Issue: Agent not routing correctly

**Check:**

1. Is `RepositoryIssue.issue_type` set correctly?
2. Are priority levels consistent with routing logic?
3. Are spoke repository names registered in configuration?

### Issue: Label enforcement failing

**Check:**

1. Service account has IAM permission: `compute.instances.setLabels`
2. Resource exists and is accessible
3. All required labels have values (not empty strings)

### Issue: Compliance report shows errors

**Check:**

1. GCP resource IDs are correctly formatted
2. Service account has `resourcemanager.organizations.get` permission
3. Cloud Audit Logs are enabled for the resource

## See Also

- [Landing Zone Compliance Audit](../LANDING_ZONE_COMPLIANCE_AUDIT_2026-01-19.md)
- [Hub & Spoke Repository Architecture](../architecture.md#hub-and-spoke)
- [PMO Mandate (pmo.yaml)](../../pmo.yaml)
- [Elite Standards Reference](../ELITE_STANDARDS_REFERENCE.md)

---

**Agent Framework Version:** 2.0
**Last Updated:** 2026-01-26
**Status:** Production-Ready
"""
