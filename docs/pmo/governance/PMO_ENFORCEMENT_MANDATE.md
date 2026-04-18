# PMO Enforcement Mandate

**Migrated from**: gcp-landing-zone/docs/governance/PMO_ENFORCEMENT_MANDATE.md
**Purpose**: Master governance specification for PMO compliance

---

## Executive Summary

This document establishes **mandatory** PMO (Program Management Office) governance requirements for all repositories in the Ollama ecosystem. Compliance is **non-negotiable** and enforced via automated checks.

**Compliance Status**: All repositories MUST maintain 100% compliance 24/7.

---

## 1. Mandatory pmo.yaml

Every repository **MUST** contain a `pmo.yaml` file at the root with **24 mandatory labels** across 6 categories:

### 1.1 Organizational Labels (4 required)

```yaml
environment: "production|staging|development|sandbox"
cost_center: "engineering|ai-ml|data|infra"
team: "platform-engineering|ml-team|data-team"
managed_by: "terraform|manual|cloudformation"
```

### 1.2 Lifecycle Labels (5 required)

```yaml
created_by: "email@domain.com"
created_date: "YYYY-MM-DD"
lifecycle_state: "active|maintenance|sunset|archived"
teardown_date: "YYYY-MM-DD|none"
retention_days: "365|3650|7300"
```

### 1.3 Business Labels (4 required)

```yaml
product: "ollama|landing-zone|shared-services"
component: "api|database|frontend|auth|monitoring"
tier: "critical|high|medium|low"
compliance: "sox|hipaa|pci|gdpr|none"
```

### 1.4 Technical Labels (4 required)

```yaml
version: "0.1.0"
stack: "python-3.11-fastapi|golang-1.21|nodejs-20"
backup_strategy: "daily|weekly|monthly|none"
monitoring_enabled: "true|false"
```

### 1.5 Financial Labels (4 required)

```yaml
budget_owner: "email@domain.com"
project_code: "PROJ-YYYY-NNN"
monthly_budget_usd: "500|1000|5000"
chargeback_unit: "team-name|cost-center"
```

### 1.6 Git Labels (3 required)

```yaml
git_repository: "github.com/owner/repo"
git_branch: "main|master|develop"
auto_delete: "true|false"
```

---

## 2. GitHub Labels

All repositories MUST configure **25+ standard labels**:

### Type Labels

- `task` - Work item
- `epic` - Large initiative
- `bug` - Something broken
- `security` - Security vulnerability
- `docs` - Documentation
- `refactor` - Code refactoring
- `perf` - Performance improvement

### Priority Labels

- `priority-p0` - Critical (red)
- `priority-p1` - High (orange)
- `priority-p2` - Medium (yellow)
- `priority-p3` - Low (green)

### Component Labels

- `api` - API changes
- `database` - Database changes
- `frontend` - UI changes
- `auth` - Authentication
- `monitoring` - Observability

### PMO Labels

- `pmo` - PMO governance
- `compliance` - Compliance related
- `cost-tracking` - Cost attribution
- `governance` - Governance policy

### Phase Labels

- `phase-1`, `phase-2`, `phase-3`, `phase-4`

---

## 3. Required Workflows

All repositories MUST include:

1. **pmo-validation.yml** - Validate pmo.yaml on PR
2. **compliance-check.yml** - Daily compliance checks
3. **tests.yml** - Run tests on PR

Optional (recommended): 4. **deploy.yml** - Deployment workflow 5. **security.yml** - Security scanning

---

## 4. GPG Commit Signing

**All commits MUST be signed with GPG** (non-negotiable).

Setup:

```bash
git config --global commit.gpgsign true
git config --global user.signingkey <YOUR_GPG_KEY_ID>
```

Enforcement:

- Pre-commit hooks check for GPG signature
- GitHub branch protection requires signed commits
- CI/CD fails on unsigned commits

---

## 5. Compliance Checks

### Daily Automated Checks

- pmo.yaml exists and valid (24 labels)
- GitHub labels configured (25+ labels)
- Required workflows present
- GPG signing enabled
- No compliance drift detected

### Weekly Audits

- Cross-repository compliance report
- Cost attribution validation
- Budget vs actual analysis
- Team workload distribution

### Monthly Reviews

- Executive compliance dashboard
- Compliance trend analysis
- Cost optimization recommendations
- Governance policy updates

---

## 6. Enforcement Mechanisms

### Automated Enforcement

1. **Pre-commit hooks** - Block commits without pmo.yaml
2. **GitHub Actions** - Fail PRs with violations
3. **Branch protection** - Require compliance checks to pass
4. **Auto-remediation** - Fix drift automatically (low-risk)

### Manual Enforcement

1. **Code review** - Reviewers check compliance
2. **Quarterly audits** - PMO reviews all repos
3. **Executive reviews** - C-level compliance reports

### Consequences of Non-Compliance

- **P0**: Repository access revoked
- **P1**: Deployment blocked
- **P2**: Issue escalated to team lead
- **P3**: Warning notification

---

## 7. Onboarding Process

### New Repository Onboarding (Automated - <5 minutes)

```bash
# One-command onboarding
ollama-pmo onboard my-new-repo --auto

# Manual review and adjust
nano pmo.yaml

# Commit and push
git add pmo.yaml .github/workflows/*.yml
git commit -S -m "chore(pmo): Add PMO governance"
git push origin main
```

### Existing Repository Migration

```bash
# Audit current state
./scripts/pmo/enforce-pmo-governance.sh

# Generate pmo.yaml
ollama-pmo onboard my-existing-repo

# Review and commit
git commit -S -m "chore(pmo): Add PMO compliance"
```

---

## 8. Exception Process

**No exceptions** - Compliance is mandatory.

If temporary deviation is absolutely required:

1. Create escalation issue with justification
2. Get approval from PMO + Engineering Lead
3. Set remediation deadline (max 7 days)
4. Track in compliance dashboard

---

## 9. Success Metrics

- **Compliance Rate**: 100% target (zero tolerance)
- **Drift Detection**: <15 min detection time
- **Auto-Remediation**: 70-80% violations auto-fixed
- **Onboarding Time**: <5 minutes per repository

---

## 10. References

- pmo.yaml schema: `/docs/pmo/pmo-yaml-schema.md`
- Validation script: `/scripts/pmo/validate-pmo-metadata.sh`
- Enforcement script: `/scripts/pmo/enforce-pmo-governance.sh`
- GitHub workflows: `/.github/workflows/pmo-*.yml`

---

**Version**: 2.0.0
**Last Updated**: January 26, 2026
**Owner**: PMO Office
**Enforcement**: Automated (100%)
