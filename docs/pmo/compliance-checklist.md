# ✅ PMO Compliance Checklist

**Purpose**: Verify 100% compliance with PMO governance standards
**Frequency**: Daily (automated), Quarterly (manual review)
**Scope**: All repositories in kushin77 GitHub organization

---

## Quick Compliance Check

```bash
# Run automated compliance check
./scripts/pmo/enforce-pmo-governance.sh

# Expected output:
# ✅ All governance checks passed
```

---

## Category 1: pmo.yaml Metadata (24 Labels)

### Organizational Labels (4 required)

- [ ] **project** - Project name (String, required)
  - Format: `lowercase-with-hyphens`
  - Example: `ollama`, `gcp-landing-zone`

- [ ] **team** - Owning team (String, required)
  - Example: `platform-engineering`, `data-science`

- [ ] **owner** - Primary owner email (Email, required)
  - Format: Valid email address
  - Example: `john.doe@company.com`

- [ ] **department** - Department (String, required)
  - Example: `Engineering`, `Product`, `Operations`

### Lifecycle Labels (5 required)

- [ ] **lifecycle_status** - Current lifecycle status (Enum, required)
  - Values: `active`, `maintenance`, `sunset`, `deprecated`

- [ ] **environment** - Deployment environment (Enum, required)
  - Values: `production`, `staging`, `development`, `sandbox`

- [ ] **tier** - Service tier (Enum, required)
  - Values: `tier-1`, `tier-2`, `tier-3`

- [ ] **criticality** - Business criticality (Enum, required)
  - Values: `critical`, `high`, `medium`, `low`

- [ ] **support_level** - Support availability (Enum, required)
  - Values: `24x7`, `business-hours`, `best-effort`

### Business Labels (4 required)

- [ ] **business_unit** - Business unit (String, required)
  - Example: `Product`, `Sales`, `Marketing`

- [ ] **product** - Product name (String, required)
  - Example: `Ollama AI Platform`

- [ ] **service_category** - Service type (Enum, required)
  - Values: `Application`, `Infrastructure`, `Platform`, `Data`

- [ ] **sla_tier** - SLA tier (Enum, required)
  - Values: `gold`, `silver`, `bronze`

### Technical Labels (4 required)

- [ ] **stack** - Technology stack (String, required)
  - Example: `python`, `nodejs`, `go`, `java`

- [ ] **architecture** - Architecture pattern (Enum, required)
  - Values: `monolith`, `microservices`, `serverless`, `hybrid`

- [ ] **data_classification** - Data sensitivity (Enum, required)
  - Values: `public`, `internal`, `confidential`, `restricted`

- [ ] **compliance_frameworks** - Compliance requirements (Enum[], required)
  - Values: `soc2`, `hipaa`, `gdpr`, `pci-dss`, `none`

### Financial Labels (4 required)

- [ ] **cost_center** - Cost center code (String, required)
  - Format: `CC-XXXXX`
  - Example: `CC-12345`

- [ ] **budget_code** - Budget allocation (String, required)
  - Example: `ENG-2026-Q1`

- [ ] **charge_code** - Charge back code (String, required)
  - Example: `PROJ-001`

- [ ] **approved_budget** - Approved budget (Integer, required)
  - Format: USD amount without decimals
  - Example: `50000`

### Git Labels (3 required)

- [ ] **git_repo** - Repository URL (String, required)
  - Format: Full GitHub URL
  - Example: `https://github.com/kushin77/ollama`

- [ ] **git_branch** - Default branch (String, required)
  - Example: `main`, `develop`

- [ ] **created_by** - Creator email (Email, required)
  - Example: `john.doe@company.com`

---

## Category 2: GitHub Configuration

### Labels (35+ required)

**Type Labels** (7):

- [ ] `type-feature` - New feature
- [ ] `type-bug` - Bug fix
- [ ] `type-docs` - Documentation
- [ ] `type-refactor` - Code refactoring
- [ ] `type-test` - Testing
- [ ] `type-perf` - Performance improvement
- [ ] `type-infra` - Infrastructure changes

**Priority Labels** (4):

- [ ] `priority-p0` - Critical/Urgent
- [ ] `priority-p1` - High priority
- [ ] `priority-p2` - Medium priority
- [ ] `priority-p3` - Low priority

**Component Labels** (7):

- [ ] `component-api` - API layer
- [ ] `component-auth` - Authentication
- [ ] `component-database` - Database
- [ ] `component-docker` - Containerization
- [ ] `component-frontend` - Frontend
- [ ] `component-backend` - Backend
- [ ] `component-tests` - Test infrastructure

**Effort Labels** (5):

- [ ] `effort-xs` - <4 hours
- [ ] `effort-s` - 4-8 hours
- [ ] `effort-m` - 8-16 hours
- [ ] `effort-l` - 16-40 hours
- [ ] `effort-xl` - >40 hours

**PMO Labels** (4):

- [ ] `pmo` - PMO-related
- [ ] `governance` - Governance tasks
- [ ] `compliance` - Compliance requirements
- [ ] `cost-tracking` - Cost attribution

**Phase Labels** (4):

- [ ] `phase-1` - Phase 1
- [ ] `phase-2` - Phase 2
- [ ] `phase-3` - Phase 3
- [ ] `phase-4` - Phase 4

**Status Labels** (4):

- [ ] `in-progress` - Work in progress
- [ ] `blocked` - Blocked by dependencies
- [ ] `waiting-review` - Awaiting review
- [ ] `completed` - Completed

---

## Category 3: GitHub Actions Workflows

### Required Workflows (4 minimum)

- [ ] **pmo-validation.yml** - Validates pmo.yaml on PR
  - Triggers: PR to main/develop, push to main (pmo.yaml changes)
  - Checks: YAML syntax, label validation, enforcement

- [ ] **compliance-check.yml** - Daily compliance monitoring
  - Schedule: Daily at 9am PT (cron: `0 17 * * *`)
  - Actions: Governance checks, drift detection, reporting

- [ ] **monthly-cost-report.yml** - Monthly cost reports
  - Schedule: 1st of month at 9am PT (cron: `0 17 1 * *`)
  - Actions: Cost attribution, budget tracking, issue creation

- [ ] **security-scan.yml** - Security vulnerability scanning
  - Schedule: Daily at 2am PT (cron: `0 10 * * *`)
  - Scans: Dependencies, secrets, code, Docker images

### Workflow Requirements

- [ ] All workflows use latest versions of actions
- [ ] All workflows have proper permissions (least privilege)
- [ ] All workflows upload artifacts for failures
- [ ] All workflows create issues on critical failures
- [ ] All workflows have retry logic for transient failures

---

## Category 4: Git Configuration

### GPG Signing

- [ ] **GPG signing enabled** - `git config commit.gpgsign` = `true`
- [ ] **GPG key configured** - `git config user.signingkey` = valid key ID
- [ ] **All commits signed** - Check recent commits: `git log --show-signature`

### Git Hooks

- [ ] **Pre-commit hook installed** - `.git/hooks/pre-commit` exists and executable
  - Checks: GPG signing, pmo.yaml validation, no credentials

- [ ] **Commit-msg hook installed** - `.git/hooks/commit-msg` exists and executable
  - Checks: Commit message format, GPG signature

### Branch Protection

- [ ] **Main branch protected** - No direct pushes
- [ ] **Require PR reviews** - At least 1 approval required
- [ ] **Require status checks** - All PMO workflows must pass
- [ ] **Require signed commits** - All commits must be GPG-signed
- [ ] **Restrict force pushes** - No `git push --force` allowed

---

## Category 5: Documentation

### Required Documentation (5 minimum)

- [ ] **README.md** - Project overview, quick start, usage
- [ ] **CONTRIBUTING.md** - Contribution guidelines
- [ ] **CHANGELOG.md** - Version history
- [ ] **docs/architecture.md** - System architecture
- [ ] **docs/DEPLOYMENT.md** - Deployment procedures

### Documentation Quality

- [ ] All documentation up-to-date (within last 90 days)
- [ ] All links working (no 404s)
- [ ] All code examples tested and working
- [ ] All images/diagrams render correctly
- [ ] All API documentation matches implementation

---

## Category 6: Code Quality

### Testing

- [ ] **Test coverage ≥90%** - Run `pytest --cov=ollama --cov-report=term-missing`
- [ ] **All tests passing** - Run `pytest tests/ -v`
- [ ] **No flaky tests** - Tests pass consistently (10 consecutive runs)

### Type Safety

- [ ] **Type hints on all functions** - 100% coverage
- [ ] **mypy strict mode passing** - Run `mypy ollama/ --strict`
- [ ] **No `Any` types** - Without explicit `# type: ignore` justification

### Linting

- [ ] **Ruff checks passing** - Run `ruff check ollama/`
- [ ] **Black formatting applied** - Run `black ollama/ tests/ --check`
- [ ] **Import sorting** - Run `isort ollama/ tests/ --check`

### Security

- [ ] **pip-audit clean** - Run `pip-audit`
- [ ] **safety check clean** - Run `safety check`
- [ ] **No hardcoded credentials** - Checked via Gitleaks
- [ ] **No known vulnerabilities** - CodeQL, Trivy scans clean

---

## Category 7: Deployment

### Docker

- [ ] **Dockerfile exists** - `docker/Dockerfile` or `Dockerfile`
- [ ] **Multi-stage builds** - Minimize image size
- [ ] **No root user** - Run as non-root user
- [ ] **Health checks defined** - `HEALTHCHECK` instruction present
- [ ] **Security scan passing** - Trivy vulnerability scan clean

### Kubernetes (if applicable)

- [ ] **Manifests validated** - Run `kubectl apply --dry-run=client`
- [ ] **Resource limits defined** - CPU/memory limits on all containers
- [ ] **Liveness/readiness probes** - Defined for all containers
- [ ] **Secrets not in manifests** - Use Secret Manager or Vault
- [ ] **Network policies** - Defined for all services

### Infrastructure as Code

- [ ] **Terraform code formatted** - Run `terraform fmt -check`
- [ ] **Terraform validated** - Run `terraform validate`
- [ ] **State backend configured** - Remote backend (GCS, S3, etc.)
- [ ] **Variables documented** - All variables have descriptions

---

## Compliance Score Calculation

```
Total Checklist Items: 100
Completed Items: [X]
Compliance Score: ([X] / 100) × 100 = X%

Compliance Tiers:
- Gold (≥95%): Elite 0.01% governance
- Silver (≥85%): Above-average governance
- Bronze (≥75%): Acceptable governance
- Non-Compliant (<75%): Requires immediate action
```

---

## Automated Compliance Check

Run the automated compliance script to calculate your score:

```bash
# Run full compliance check
./scripts/pmo/enforce-pmo-governance.sh

# Sample output:
🔍 PMO Governance Enforcement

✅ pmo.yaml: PASS (24/24 labels)
✅ GitHub Labels: PASS (35/35 configured)
✅ Workflows: PASS (4/4 installed)
✅ Git Hooks: PASS (2/2 installed)
✅ GPG Signing: PASS (enabled)
✅ Branch Protection: PASS (main protected)
✅ Documentation: PASS (5/5 present)
✅ Code Quality: PASS (coverage 94%)
✅ Security: PASS (no vulnerabilities)
✅ Deployment: PASS (Docker + K8s ready)

📊 Compliance Score: 100/100 (100%) - GOLD TIER ✅
```

---

## Remediation Actions

### For Non-Compliant Repositories

**Step 1: Generate pmo.yaml**

```bash
./scripts/pmo/generate-pmo-yaml.sh
```

**Step 2: Configure GitHub labels**

```bash
./scripts/pmo/setup-labels.sh kushin77/your-repo-name
```

**Step 3: Install workflows**

```bash
cp .github/workflows/pmo-*.yml .github/workflows/
git add .github/workflows/
git commit -S -m "infra: add PMO workflows"
git push origin main
```

**Step 4: Install Git hooks**

```bash
cp templates/pmo/hooks/* .git/hooks/
chmod +x .git/hooks/*
```

**Step 5: Enable GPG signing**

```bash
git config commit.gpgsign true
git config user.signingkey YOUR_GPG_KEY_ID
```

**Step 6: Re-validate**

```bash
./scripts/pmo/enforce-pmo-governance.sh
```

---

## Quarterly Review Checklist

**Schedule**: Every quarter (Q1, Q2, Q3, Q4)

- [ ] Review pmo.yaml for accuracy
- [ ] Update cost center and budget codes
- [ ] Verify GitHub labels still relevant
- [ ] Review workflow run history
- [ ] Check for compliance drift
- [ ] Update documentation
- [ ] Run security scans
- [ ] Archive old issues/PRs
- [ ] Update dependencies
- [ ] Review cost reports

---

## Success Criteria

A repository is **100% compliant** when:

✅ All 24 pmo.yaml labels populated
✅ All 35+ GitHub labels configured
✅ All 4 PMO workflows installed and passing
✅ GPG signing enabled and enforced
✅ Git hooks installed and working
✅ Test coverage ≥90%
✅ Type checking passing (mypy strict)
✅ Linting clean (Ruff + Black)
✅ Security scans clean (pip-audit + safety)
✅ Documentation complete and current

**Compliance Score: 100% (Gold Tier)** 🏆

---

**Last Updated**: January 26, 2026
**Maintained By**: Platform Engineering Team
**Version**: 2.0.0
