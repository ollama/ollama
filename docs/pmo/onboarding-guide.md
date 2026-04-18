# 🚀 PMO Onboarding Guide

**For**: New repositories in the kushin77 GitHub organization
**Purpose**: Achieve 100% PMO compliance in <5 minutes
**Outcome**: Elite 0.01% governance standards with zero manual work

---

## Table of Contents

1. [Quick Start (5 Minutes)](#quick-start-5-minutes)
2. [Prerequisites](#prerequisites)
3. [Step-by-Step Onboarding](#step-by-step-onboarding)
4. [Verification](#verification)
5. [Troubleshooting](#troubleshooting)
6. [Manual Onboarding (Fallback)](#manual-onboarding-fallback)

---

## Quick Start (5 Minutes)

### Automated Onboarding (Recommended)

```bash
# 1. Clone the Ollama repository for PMO scripts
git clone https://github.com/kushin77/ollama.git /tmp/ollama-pmo
cd YOUR_REPOSITORY

# 2. Run the automated onboarding script
/tmp/ollama-pmo/scripts/pmo/onboard-repository.sh \
    --repo "$(basename $(pwd))" \
    --team "your-team-name" \
    --cost-center "CC-12345"

# 3. Verify compliance
/tmp/ollama-pmo/scripts/pmo/enforce-pmo-governance.sh

# 4. Done! ✅
```

**Expected Output:**

```
✅ pmo.yaml generated with 24 labels
✅ GitHub labels configured (35+ labels)
✅ Workflows installed (PMO validation + compliance)
✅ Git hooks configured (GPG signing + validation)
✅ All compliance checks passed
🎉 Repository onboarded successfully!
```

---

## Prerequisites

**Required Tools:**

- Git 2.30+ (with GPG signing configured)
- GitHub CLI (`gh`) 2.0+
- Python 3.11+
- `yq` (YAML processor)

**Required Access:**

- Admin access to target repository
- GitHub Personal Access Token with `repo` and `workflow` scopes

**Required Information:**

- Team name (e.g., `platform-engineering`)
- Cost center code (e.g., `CC-12345`)
- Project name (auto-detected or manual)

**Verify Prerequisites:**

```bash
# Check Git
git --version  # Should be >= 2.30

# Check GitHub CLI
gh --version   # Should be >= 2.0

# Check GPG
git config --global user.signingkey  # Should have a GPG key ID

# Check Python
python3 --version  # Should be >= 3.11

# Check yq
yq --version  # Should be installed
```

---

## Step-by-Step Onboarding

### Step 1: Generate pmo.yaml

The `pmo.yaml` file is the cornerstone of PMO governance. It contains 24 mandatory labels across 6 categories.

**Automated Generation (Recommended):**

```bash
cd YOUR_REPOSITORY

# Auto-generate with intelligent defaults
/tmp/ollama-pmo/scripts/pmo/generate-pmo-yaml.sh

# Review generated file
cat pmo.yaml
```

**Manual Creation (If needed):**

```bash
# Copy template
cp /tmp/ollama-pmo/templates/pmo/pmo.yaml.template pmo.yaml

# Edit manually
vi pmo.yaml
```

**What Gets Detected:**

- ✅ Git repository URL (from `.git/config`)
- ✅ Current branch (from `git branch --show-current`)
- ✅ Created by (from `git log --reverse`)
- ✅ Created date (from first commit)
- ✅ Stack (from `pyproject.toml`, `package.json`, `go.mod`)
- ✅ Version (from package files)

**Example Generated pmo.yaml:**

```yaml
# pmo.yaml - PMO Metadata (Auto-generated)
organizational:
  project: "my-awesome-app"
  team: "platform-engineering"
  owner: "john.doe@company.com"
  department: "Engineering"

lifecycle:
  lifecycle_status: "active"
  environment: "production"
  tier: "tier-1"
  criticality: "high"
  support_level: "24x7"

business:
  business_unit: "Product"
  product: "Platform Services"
  service_category: "Infrastructure"
  sla_tier: "gold"

technical:
  stack: "python"
  architecture: "microservices"
  data_classification: "confidential"
  compliance_frameworks: "soc2,hipaa"

financial:
  cost_center: "CC-12345"
  budget_code: "ENG-2026-Q1"
  charge_code: "PLAT-001"
  approved_budget: 50000

git:
  git_repo: "https://github.com/kushin77/my-awesome-app"
  git_branch: "main"
  created_by: "john.doe@company.com"
```

### Step 2: Configure GitHub Labels

GitHub labels enable automated triage, classification, and reporting.

**Run Label Setup:**

```bash
# Configure 35+ standardized labels
/tmp/ollama-pmo/scripts/pmo/setup-labels.sh kushin77/your-repo-name
```

**Labels Created:**

- **Type** (7): `type-feature`, `type-bug`, `type-docs`, `type-refactor`, `type-test`, `type-perf`, `type-infra`
- **Priority** (4): `priority-p0`, `priority-p1`, `priority-p2`, `priority-p3`
- **Component** (7): `component-api`, `component-auth`, `component-database`, `component-docker`, `component-frontend`, `component-backend`, `component-tests`
- **Effort** (5): `effort-xs`, `effort-s`, `effort-m`, `effort-l`, `effort-xl`
- **PMO** (4): `pmo`, `governance`, `compliance`, `cost-tracking`
- **Phase** (4): `phase-1`, `phase-2`, `phase-3`, `phase-4`
- **Status** (4): `in-progress`, `blocked`, `waiting-review`, `completed`

### Step 3: Install PMO Workflows

GitHub Actions workflows automate validation, compliance checks, and reporting.

**Copy Workflows:**

```bash
# Create workflows directory
mkdir -p .github/workflows

# Copy PMO workflows
cp /tmp/ollama-pmo/.github/workflows/pmo-validation.yml .github/workflows/
cp /tmp/ollama-pmo/.github/workflows/compliance-check.yml .github/workflows/
cp /tmp/ollama-pmo/.github/workflows/monthly-cost-report.yml .github/workflows/
cp /tmp/ollama-pmo/.github/workflows/security-scan.yml .github/workflows/

# Verify
ls -la .github/workflows/
```

**Workflows Installed:**

- **pmo-validation.yml** - Validates pmo.yaml on every PR
- **compliance-check.yml** - Daily compliance monitoring (9am PT)
- **monthly-cost-report.yml** - Monthly cost attribution reports
- **security-scan.yml** - Daily security vulnerability scanning

### Step 4: Configure Git Hooks

Git hooks enforce governance at commit time (GPG signing, validation).

**Install Pre-Commit Hook:**

```bash
# Create hooks directory
mkdir -p .git/hooks

# Copy pre-commit hook
cp /tmp/ollama-pmo/templates/pmo/hooks/pre-commit .git/hooks/
chmod +x .git/hooks/pre-commit

# Verify
ls -la .git/hooks/pre-commit
```

**Pre-Commit Hook Checks:**

- ✅ GPG commit signing enabled
- ✅ pmo.yaml exists and valid
- ✅ Mandatory labels populated (min 20/24)
- ✅ No credentials in commit
- ✅ All tests passing (if applicable)

**Install Commit-Msg Hook:**

```bash
# Copy commit-msg hook
cp /tmp/ollama-pmo/templates/pmo/hooks/commit-msg .git/hooks/
chmod +x .git/hooks/commit-msg
```

**Commit-Msg Hook Checks:**

- ✅ Commit message format: `type(scope): description`
- ✅ GPG signature present

### Step 5: Validate Compliance

Run the enforcement script to verify all PMO standards are met.

**Run Enforcement:**

```bash
/tmp/ollama-pmo/scripts/pmo/enforce-pmo-governance.sh
```

**Expected Output:**

```
🔍 PMO Governance Enforcement

Checking pmo.yaml exists... ✅ PASS
Checking mandatory labels... ✅ PASS (24/24 labels populated)
Checking GitHub labels... ✅ PASS (35 labels configured)
Checking workflows... ✅ PASS (4 PMO workflows present)
Checking GPG signing... ✅ PASS (commit signing enabled)
Checking pre-commit hooks... ✅ PASS (hooks installed)

✅ All governance checks passed
```

**If Failures Occur:**

```
❌ FAIL: pmo.yaml missing
→ Run: /tmp/ollama-pmo/scripts/pmo/generate-pmo-yaml.sh

❌ FAIL: GitHub labels not configured
→ Run: /tmp/ollama-pmo/scripts/pmo/setup-labels.sh kushin77/your-repo

❌ FAIL: Workflows missing
→ Copy workflows from /tmp/ollama-pmo/.github/workflows/

❌ FAIL: GPG signing not enabled
→ Run: git config commit.gpgsign true
```

---

## Verification

### Checklist

Use this checklist to verify onboarding completeness:

- [ ] **pmo.yaml exists** - Run `ls -la pmo.yaml`
- [ ] **pmo.yaml valid** - Run `./scripts/pmo/validate-pmo-metadata.sh`
- [ ] **24 labels populated** - Check `yq eval '.organizational, .lifecycle, .business, .technical, .financial, .git' pmo.yaml`
- [ ] **GitHub labels configured** - Run `gh label list | wc -l` (should be 35+)
- [ ] **4 workflows installed** - Run `ls -la .github/workflows/ | grep pmo`
- [ ] **Pre-commit hook installed** - Run `ls -la .git/hooks/pre-commit`
- [ ] **GPG signing enabled** - Run `git config commit.gpgsign` (should output `true`)
- [ ] **All checks passing** - Run `./scripts/pmo/enforce-pmo-governance.sh`

### Automated Verification

```bash
# Run full compliance check
/tmp/ollama-pmo/scripts/pmo/enforce-pmo-governance.sh

# Run metadata validation
/tmp/ollama-pmo/scripts/pmo/validate-pmo-metadata.sh

# Check GitHub labels
gh label list | grep -E 'type-|priority-|component-|effort-|pmo|phase-|status-'

# Test pre-commit hook
echo "test" > test.txt
git add test.txt
git commit -S -m "test: validate hooks" # Should trigger hook validation
```

---

## Troubleshooting

### Common Issues

#### Issue: "pmo.yaml not found"

**Solution:**

```bash
# Generate pmo.yaml
cd YOUR_REPOSITORY
/tmp/ollama-pmo/scripts/pmo/generate-pmo-yaml.sh
```

#### Issue: "GitHub labels not configured"

**Solution:**

```bash
# Install labels (requires admin access)
gh auth login
/tmp/ollama-pmo/scripts/pmo/setup-labels.sh kushin77/your-repo-name
```

#### Issue: "GPG signing not enabled"

**Solution:**

```bash
# Enable GPG signing globally
git config --global commit.gpgsign true
git config --global user.signingkey YOUR_GPG_KEY_ID

# Or per-repository
cd YOUR_REPOSITORY
git config commit.gpgsign true
```

#### Issue: "Pre-commit hook not executing"

**Solution:**

```bash
# Ensure hook is executable
chmod +x .git/hooks/pre-commit

# Test manually
./.git/hooks/pre-commit
```

#### Issue: "Workflows not triggering"

**Solution:**

```bash
# Ensure workflows are in correct location
ls -la .github/workflows/

# Push to trigger
git add .github/workflows/
git commit -S -m "infra: add PMO workflows"
git push origin main

# Check GitHub Actions tab for runs
```

#### Issue: "Validation failing on mandatory labels"

**Solution:**

```bash
# Check which labels are missing
/tmp/ollama-pmo/scripts/pmo/validate-pmo-metadata.sh

# Edit pmo.yaml and fill in missing labels
vi pmo.yaml

# Re-validate
/tmp/ollama-pmo/scripts/pmo/validate-pmo-metadata.sh
```

---

## Manual Onboarding (Fallback)

If automated onboarding fails, follow these manual steps:

### 1. Create pmo.yaml Manually

```bash
# Copy template
cat > pmo.yaml << 'EOF'
organizational:
  project: "your-project-name"
  team: "your-team-name"
  owner: "your.email@company.com"
  department: "Engineering"

lifecycle:
  lifecycle_status: "active"
  environment: "production"
  tier: "tier-2"
  criticality: "medium"
  support_level: "business-hours"

business:
  business_unit: "Product"
  product: "Your Product"
  service_category: "Application"
  sla_tier: "silver"

technical:
  stack: "python"
  architecture: "monolith"
  data_classification: "internal"
  compliance_frameworks: "soc2"

financial:
  cost_center: "CC-00000"
  budget_code: "ENG-2026-Q1"
  charge_code: "PROJ-001"
  approved_budget: 10000

git:
  git_repo: "https://github.com/kushin77/your-repo"
  git_branch: "main"
  created_by: "your.email@company.com"
EOF
```

### 2. Configure GitHub Labels Manually

```bash
# Type labels
gh label create "type-feature" --color "0075ca" --description "New feature"
gh label create "type-bug" --color "d73a4a" --description "Bug fix"
gh label create "type-docs" --color "0075ca" --description "Documentation"
# ... (continue for all 35 labels)
```

### 3. Copy Workflows Manually

```bash
mkdir -p .github/workflows
# Copy each workflow file manually from /tmp/ollama-pmo/.github/workflows/
```

### 4. Install Hooks Manually

```bash
mkdir -p .git/hooks
# Copy pre-commit and commit-msg hooks manually
chmod +x .git/hooks/*
```

---

## Success Metrics

After onboarding, you should achieve:

- ✅ **100% PMO Compliance** - All governance checks passing
- ✅ **<5 Minutes Onboarding Time** - From zero to fully compliant
- ✅ **Zero Manual Work** - Automated enforcement via Git hooks + workflows
- ✅ **Automated Reporting** - Monthly cost reports, compliance dashboards
- ✅ **Real-Time Validation** - PRs validated before merge
- ✅ **Elite Standards** - Top 0.01% governance quality

---

## Next Steps

1. **Commit pmo.yaml** to main branch
2. **Monitor GitHub Actions** for workflow runs
3. **Review monthly cost reports** (1st of each month)
4. **Update pmo.yaml** as project evolves
5. **Onboard remaining repositories** using same process

---

## Support

**Questions?** Open an issue in the Ollama repository:
https://github.com/kushin77/ollama/issues

**Documentation:**

- [PMO Enforcement Mandate](./PMO_ENFORCEMENT_MANDATE.md)
- [pmo.yaml Schema](./pmo-yaml-schema.md)
- [Compliance Checklist](./compliance-checklist.md)

---

**Last Updated**: January 26, 2026
**Maintained By**: Platform Engineering Team
**Version**: 2.0.0
