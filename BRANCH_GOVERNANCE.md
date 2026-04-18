# Branch Governance & Control Framework

## Overview
This document outlines policies and automated mechanisms to maintain a clean, manageable branch structure and prevent branch sprawl.

Production deployment mandate: every merge or direct push to `main` must be followed by a production redeploy using the approved CI/CD pipeline.

## 1. Branch Lifecycle Policy

### Branch Categories
```
Pattern                          Lifetime    Auto-cleanup
─────────────────────────────────────────────────────────
main                            ∞           Never
release/*                       1 year      Manual review
feature/*                       6 months    Auto-delete if stale + unmerged
fix/*                          3 months    Auto-delete if merged to main
chore/*                        2 months    Auto-delete if merged
automation/*                   3 months    Auto-delete after merge or if stale
dependabot/*                   3 weeks     Auto-delete after merge
bugfix/*                       3 months    Auto-delete if merged
experiment/*                   1 month     Auto-delete (disposable)
wip/*                          2 weeks     Auto-delete
poc/*                          2 weeks     Auto-delete
```

### Stale Branch Definition
- **No activity for 180 days** (commits, pushes, or PR activity)
- **Not merged to main** or any release branch
- **No linked PRs** in draft or open state

### Action Triggers
- **90 days old**: Add warning label to associated PRs
- **180 days old**: Delete if unmerged AND no linked PRs
- **Upon merge**: Delete branch immediately (configurable per branch pattern)

## 2. Automated Cleanup Workflow

### GitHub Actions Workflow
```yaml
name: Branch Hygiene (Scheduled)
on:
  schedule:
    - cron: '0 2 * * 0'  # Weekly, Sundays at 2 AM UTC
  workflow_dispatch:      # Manual trigger

jobs:
  cleanup:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Cleanup stale branches
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          python3 scripts/cleanup_stale_branches.py \
            --age-days 180 \
            --exclude-patterns main,release \
            --report-file .github/branch_hygiene_report.json

      - name: Report results
        if: always()
        run: cat .github/branch_hygiene_report.json | python3 -m json.tool
```

### Cleanup Script Location
- **File**: `scripts/cleanup_stale_branches.py`
- **Dependencies**: GitPython, `github` library
- **Features**:
  - Calculate branch age and activity
  - Check for linked PRs
  - Generate report before deletion
  - Dry-run mode for validation
  - Audit trail in `.github/branch_cleanup_audit.jsonl`

## 3. Branch Naming Conventions

### Required Format
```
<type>/<scope>/<short-description>

Examples:
  feature/ai-inference-migration
  fix/branch-protection-syntax
  chore/ci-install-dev
  automation/issue-orchestrator
  experiment/new-inference-engine
```

### Enforce via Hooks
- Pre-push hook validates branch name
- CI rejects PRs from incorrectly named branches
- GitHub branch protection rules match patterns

## 4. Preventive Controls

### A. Merge Strategy Configuration
**`.github/workflows/merge-cleanup.yml`**
```yaml
on:
  pull_request:
    types: [closed]

jobs:
  cleanup-branch:
    if: github.event.pull_request.merged
    runs-on: ubuntu-latest
    steps:
      - run: |
          gh api repos/${{ github.repository }}/git/refs/heads/${{ github.event.pull_request.head.ref }} \
            -X DELETE 2>/dev/null || true
```

### B. IaC Branch Policy Declaration
**`.github/branch-governance.iac.json`**
```json
{
  "governance": {
    "enabled": true,
    "enforcement_level": "strict",
    "auto_cleanup": {
      "enabled": true,
      "schedule": "0 2 * * 0",
      "dry_run_first": true
    },
    "branch_rules": {
      "feature/*": {
        "max_age_days": 180,
        "delete_after_merge": true,
        "warn_at_days": 90,
        "require_upstream": true
      },
      "fix/*": {
        "max_age_days": 90,
        "delete_after_merge": true
      },
      "automation/*": {
        "max_age_days": 90,
        "delete_after_merge": false,
        "protected": true
      }
    }
  }
}
```

### C. GitHub API Configuration
Deploy branch protection rules programmatically:
```bash
# Enforce branch naming patterns
# Require PR reviews before merge
# Auto-delete head branches after merge
# Require status checks to pass
# Require production redeploy workflow for main branch updates
```

### D. Main Branch Deployment Mandate
- Any commit that reaches `main` (merge commit, squash merge, rebase merge, or direct push) must trigger a production redeploy.
- The redeploy must target the exact commit SHA currently on `main`.
- If deployment fails, production incident handling starts immediately and the commit is treated as incomplete until redeploy succeeds.
- No issue, task, or rollout is considered complete until production redeploy verification is recorded.

## 5. Monitoring & Dashboards

### Key Metrics to Track
1. **Total branch count** (trend over time)
2. **Stale branch percentage** (>180 days without activity)
3. **Merged branches not deleted** (cleanup failures)
4. **Average branch lifetime** by type
5. **Deletion success rate** (auto-cleanup effectiveness)

### Dashboard File
**`.github/branch_stats.json`** (updated weekly)
```json
{
  "period": "2026-04-18",
  "total_branches": 30,
  "stale_branches": 0,
  "merged_not_deleted": 0,
  "avg_lifetime_days": 45,
  "cleanup_success_rate": 100,
  "target_metrics": {
    "max_total_branches": 50,
    "max_stale_percentage": 5,
    "min_cleanup_success": 95
  }
}
```

## 6. Developer Guidelines

### DO ✅
- Use descriptive branch names following the convention
- Delete your own branches after merge
- Clean up experiments weekly
- Update PR description with branch cleanup timeframe
- Use `[keep]` label for long-running branches requiring exceptions

### DON'T ❌
- Create branches without clear purpose
- Leave branches dangling after PR closes
- Use generic names like `fix`, `update`, `changes`
- Accumulate WIP branches
- Ignore stale branch warnings

### Branch Lifetime Expectations
```
Feature development:    6 months max
Bug fixes:             3 months max
Infrastructure work:   Until merged
Experiments:           2 weeks max
```

## 7. Exception Handling

### Marking Branches for Long-term Keeping
Add to PR description:
```
[keep-branch]
Reason: Active development on v2.0 features, expected completion Q3 2026
```

### Adding GitHub Labels
- `keep-branch` - Exempt from auto-cleanup
- `stale-warning` - 90 days old, activity required
- `ready-for-cleanup` - Manual approval for deletion

## 8. Implementation Roadmap

| Phase | Deliverable | Timeline |
|-------|-------------|----------|
| 1 | Document policy (this file) | Week 1 |
| 2 | Create cleanup script + workflow | Week 2 |
| 3 | Deploy automated schedule | Week 3 |
| 4 | Establish monitoring + dashboards | Week 4 |
| 5 | Developer training + enforcement | Ongoing |

## 9. Rollout Plan

### Week 1: Awareness
- Announce policy to team
- Distribute guidelines
- Enable dry-run mode

### Week 2: Soft Enforcement
- Auto-cleanup with approval notifications
- Manual pause button enabled
- Dashboard visibility

### Week 3: Full Automation
- Enable automatic deletion
- Weekly reporting
- Exception handling established

### Ongoing: Maintenance
- Monitor metrics weekly
- Adjust thresholds based on data
- Quarterly policy reviews

## 10. Emergency Procedures

### Restore Deleted Branch
If a branch is accidentally deleted:
```bash
# Recover from GitHub API using git reflog or branch history
git checkout <commit-sha>
git checkout -b <branch-name>
git push origin <branch-name>
```

### Disable Automation Temporarily
Set environment variable:
```bash
export BRANCH_CLEANUP_DISABLED=true
```

## 11. Governance Artifacts

### Files to Create/Update
- `.github/workflows/branch-cleanup.yml` - Scheduled cleanup
- `.github/workflows/merge-cleanup.yml` - Auto-delete on merge
- `.github/branch-governance.iac.json` - Policy configuration
- `.github/branch_cleanup_audit.jsonl` - Immutable audit log
- `.github/branch_stats.json` - Weekly metrics
- `scripts/cleanup_stale_branches.py` - Core cleanup logic
- `.githooks/pre-push` - Local validation (optional)
- `BRANCH_GOVERNANCE.md` - This file

### Audit Trail Location
All deletions logged to: `.github/branch_cleanup_audit.jsonl`
```json
{
  "timestamp": "2026-04-25T02:00:00Z",
  "branch": "feature/experiment-xyz",
  "reason": "stale-unmerged",
  "age_days": 195,
  "deleted_by": "automated-cleanup",
  "recoverable": true,
  "last_commit": "abc123def"
}
```

## Conclusion
This governance framework ensures:
- ✅ Automatic cleanup of stale branches
- ✅ Developer flexibility with exceptions
- ✅ Full audit trail and recovery options
- ✅ Clear policies and expectations
- ✅ Measurable metrics for continuous improvement
