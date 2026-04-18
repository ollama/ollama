# Branch Governance - Quick Setup & Implementation Guide

## 🎯 Goal
Maintain a clean, manageable branch structure with **max 50 active branches** and **0% stale branches** through automated governance.

## 📋 What's Been Implemented

### 1. **Automated Cleanup System**
- **File**: `scripts/cleanup_stale_branches.py`
- **Function**: Identifies and removes branches older than 180 days (configurable)
- **Safety**: Always runs in dry-run mode first, generates audit trail
- **Schedule**: Weekly (Sundays 2 AM UTC) via GitHub Actions

### 2. **GitHub Actions Workflows**
| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `branch-cleanup.yml` | Scheduled weekly + manual | Batch cleanup of stale branches |
| `auto-delete-merged-branch.yml` | PR merged | Auto-delete branch immediately after merge |

### 3. **Governance Configuration (IaC)**
- **File**: `.github/branch-governance.iac.json`
- **Contains**: Branch lifetime rules, protected patterns, metrics targets
- **Format**: JSON, version controlled, auditable
- **Updates**: Edit IaC file to change global policy

### 4. **Branch Naming Enforcement**
- **Pattern**: `^(feature|fix|chore|docs|automation|experiment|wip|poc|release)/[a-z0-9][a-z0-9-]*$`
- **Enforced at**: Pre-push hook, CI checks, PR requirements
- **Examples**:
  - ✅ `feature/ai-inference-migration`
  - ✅ `fix/race-condition-cache`
  - ❌ `feature_ai-migration` (uses underscore)
  - ❌ `update-stuff` (missing prefix)

### 5. **Audit Trail (Immutable)**
- **File**: `.github/branch_cleanup_audit.jsonl`
- **Format**: One JSON object per line (JSONL)
- **Contents**: Every deletion with timestamp, reason, commit hash
- **Retention**: 90 days minimum
- **Purpose**: Recovery, compliance, analysis

### 6. **Branch Lifetime Policy**
```
main                  ♾️ Never deleted
release/*             1 year max
feature/*             6 months max
fix/*                 3 months max
chore/*               2 months max
automation/*          3 months max (not deleted on merge)
docs/*                2 months max
dependabot/*          3 weeks max
experiment/*          1 month max (disposable)
wip/*                 2 weeks max (disposable)
```

## 🚀 For Developers: Quick Start

### Setup (One-time)
```bash
# Install local pre-push hook for branch name validation
cp .githooks/pre-push .git/hooks/pre-push
chmod +x .git/hooks/pre-push

# Configure git to use hooks from .githooks
git config core.hooksPath .githooks
```

### Create a Branch
```bash
# ✅ Good examples
git checkout -b feature/new-inference-engine    # Feature work
git checkout -b fix/memory-leak-in-tokenizer    # Bug fix
git checkout -b chore/update-dependencies       # Maintenance
git checkout -b experiment/alternative-backend  # Experiments

# ❌ Bad examples (will be rejected)
git checkout -b my-feature                      # Missing prefix
git checkout -b Feature/NewEngine                # Wrong case
git checkout -b fix_memory_leak                 # Uses underscore
```

### Work with Branches
```bash
# Check branch age and activity
git log -1 --format="%ci %s" origin/your-branch

# Merge your work
git checkout main
git pull origin main
git merge --no-ff your-branch
git push origin main your-branch

# Note: After merge, your branch will be auto-deleted
```

### Keep a Branch Longer (Exception)
```bash
# Add to PR description or commit message:
[keep-branch]
Reason: Long-running feature, completion expected Q3 2026

# Note: Still limited to 2x the max age for the branch type
```

## 📊 Governance in Action

### Weekly Automated Tasks
1. **Sunday 2 AM UTC**: Branch cleanup script runs
   - Identifies stale branches
   - Generates dry-run report
   - Posts summary to GitHub

2. **Any Time**: PR merged
   - Branch automatically deleted
   - Deletion logged to audit trail

3. **Weekly**: Metrics updated
   - Branch count
   - Cleanup success rate
   - Stale percentage

### Monitoring & Metrics
- **Check Status**: `.github/branch_stats.json` (updated weekly)
- **View Audit Trail**: `.github/branch_cleanup_audit.jsonl` (immutable log)
- **Workflow Runs**: [Actions](https://github.com/kushin77/ollama/actions)

## 🛠️ Managing the System

### Run Manual Cleanup (Dry-run)
```bash
# Dry-run mode - see what would be deleted
python3 scripts/cleanup_stale_branches.py \
  --age-days 180 \
  --exclude-patterns main,release,automation \
  --dry-run

# View report
cat .github/branch_cleanup_report.json | python3 -m json.tool
```

### Run Manual Cleanup (Execute)
```bash
# Actually delete branches
python3 scripts/cleanup_stale_branches.py \
  --age-days 180 \
  --exclude-patterns main,release,automation \
  --execute
```

### Adjust Governance Rules
Edit `.github/branch-governance.iac.json`:
```json
{
  "branch_rules": {
    "feature/*": {
      "max_age_days": 180,        // Change here
      "warn_at_days": 90,          // Change here
      ...
    }
  }
}
```

Commit changes:
```bash
git add .github/branch-governance.iac.json
git commit -m "chore: update branch governance policy

- Increase feature branch lifetime to 240 days
- Lower warning threshold to 60 days"
```

### Recover Deleted Branch
```bash
# Find in audit trail
grep "my-branch" .github/branch_cleanup_audit.jsonl

# Get commit hash from audit entry, then restore
git checkout <commit-hash>
git checkout -b my-branch
git push origin my-branch
```

## 📈 Expected Outcomes

### Before Governance (April 2026)
- 418 total branches
- 388 stale branches (93% waste)
- No visibility into cleanup
- No naming standards
- Manual interventions only

### After Governance (Ongoing)
- **Target**: ≤50 total branches
- **Target**: <5% stale branches
- **Automated**: Weekly cleanup
- **Enforced**: Branch naming via hooks
- **Audited**: Full immutable trail

## 🔧 Customization

### Change Cleanup Schedule
Edit `.github/workflows/branch-cleanup.yml`:
```yaml
schedule:
  - cron: '0 2 * * 0'  # Change the cron expression
```

### Add New Branch Lifetime Rule
Edit `.github/branch-governance.iac.json`:
```json
{
  "branch_rules": {
    "custom/*": {
      "max_age_days": 90,
      "delete_after_merge": true,
      "protected": false,
      "description": "Custom branches"
    }
  }
}
```

### Exclude Branch from Auto-Cleanup
Add `[keep-branch]` to PR description or branch label, OR:
Edit governance.iac.json to add to `exclude_patterns`.

## 📚 Files Reference

| File | Purpose | When to Edit |
|------|---------|--------------|
| `BRANCH_GOVERNANCE.md` | Policy documentation | When changing governance approach |
| `.github/branch-governance.iac.json` | Configuration & rules | When adjusting cleanup policy |
| `scripts/cleanup_stale_branches.py` | Cleanup engine | When changing cleanup logic |
| `.github/workflows/branch-cleanup.yml` | Scheduled cleanup | When changing schedule |
| `.github/workflows/auto-delete-merged-branch.yml` | Auto-delete on merge | When changing merge behavior |
| `.githooks/pre-push` | Local validation | When changing naming rules |
| `.github/branch_cleanup_audit.jsonl` | Immutable audit log | Never edit (only append) |

## 🚨 Emergency Procedures

### Disable Automated Cleanup Temporarily
```bash
# Set environment variable to disable
export BRANCH_CLEANUP_DISABLED=true

# Or delete the schedule from workflow:
# Edit .github/workflows/branch-cleanup.yml, remove 'schedule'
```

### Recover Recently Deleted Branch
```bash
# Check audit log for recent deletions
tail -20 .github/branch_cleanup_audit.jsonl

# Find commit SHA, restore branch
git checkout <commit-sha>
git checkout -b <branch-name>
git push origin <branch-name>
```

## 📞 Questions?

- **Policy Questions**: See `BRANCH_GOVERNANCE.md`
- **Configuration Questions**: Check `.github/branch-governance.iac.json`
- **Automation Issues**: Review workflow logs in GitHub Actions
- **Audit Trail**: Check `.github/branch_cleanup_audit.jsonl`

---

**Last Updated**: 2026-04-18
**Status**: Full automation active
**Next Review**: 2026-05-18
