# Branch Protection & CI/CD Setup

This directory contains infrastructure and automation for branch protection, requiring signed commits, status checks, and Pull Request reviews on the `main` branch.

## Overview

The branch protection configuration enforces **Landing Zone compliance standards**:

✅ **Required Pull Request Reviews** (1 approval minimum)  
✅ **Signed Commits** (GPG signatures enforced)  
✅ **Status Checks** (CI validation before merge)  
✅ **Stale Review Dismissal** (new commits invalidate old approvals)  
✅ **Force Push Prevention** (immutable history)  
✅ **Admin Enforcement** (no exceptions for maintainers)  

---

## Setup Options

### Option 1: GitHub CLI (Recommended - Quick Setup)

**Prerequisites:**
- [GitHub CLI](https://cli.github.com/) installed
- Authenticated: `gh auth login`
- Admin access to the repository

**Steps:**
```bash
cd /home/akushnir/ollama
bash scripts/enable-branch-protection.sh
```

**What it does:**
- Enables branch protection on `main` branch
- Requires 1 PR review
- Enforces signed commits
- Requires "validate-landing-zone" status check
- Blocks force pushes and deletions
- Dismisses stale reviews on new commits

**Verify:**
```bash
gh api repos/kushin77/ollama/branches/main/protection
```

---

### Option 2: Terraform (Infrastructure as Code)

**Prerequisites:**
- [Terraform](https://www.terraform.io/) installed
- GitHub token with `admin:repo_hook` scope
- Workspace configured

**Steps:**
```bash
cd /home/akushnir/ollama/terraform

# Set GitHub token
export GITHUB_TOKEN="ghp_xxxxxxxxxxxxx"

# Initialize Terraform
terraform init

# Plan changes
terraform plan -var="github_token=$GITHUB_TOKEN"

# Apply configuration
terraform apply -var="github_token=$GITHUB_TOKEN"
```

**What it does:**
- Provisions branch protection as code
- Tracks configuration in version control
- Enables disaster recovery and auditing
- Allows easy updates/rollbacks

**Verify:**
```bash
terraform show
```

---

### Option 3: GitHub UI (Manual - Not Recommended)

1. Go to [https://github.com/kushin77/ollama/settings/branches](https://github.com/kushin77/ollama/settings/branches)
2. Click "Add rule"
3. Branch name pattern: `main`
4. Enable:
   - ☑ Require a pull request before merging
   - ☑ Require approvals (1)
   - ☑ Dismiss stale pull request approvals when new commits are pushed
   - ☑ Require code owner reviews
   - ☑ Require status checks to pass before merging
     - Status checks: `validate-landing-zone`
   - ☑ Require branches to be up to date before merging
   - ☑ Require signed commits
   - ☑ Restrict who can push to matching branches
   - ☑ Allow force pushes → None
   - ☑ Allow deletions
5. Click "Create"

---

## CI/CD Status Checks

The `validate-landing-zone` status check runs on every push and PR:

**Workflow:** `.github/workflows/validate-landing-zone.yml`

**Checks:**
1. **Type Safety** → `mypy ollama/ --strict`
2. **Code Quality** → `ruff check ollama/`
3. **Unit & Integration Tests** → `pytest tests/ --cov=ollama`
4. **Security Audit** → `pip-audit` (dependencies)
5. **Folder Structure** → `python scripts/validate_folder_structure.py --strict`
6. **Secret Scanning** → TruffleHog (hardcoded credentials)
7. **CodeQL Analysis** → GitHub's code analysis

---

## Pre-Merge Checklist

Before merging any PR to `main`:

- [ ] All status checks pass (CI green)
- [ ] At least 1 approval from code owner
- [ ] Commit is GPG signed (`git commit -S`)
- [ ] Branch is up to date with `main`
- [ ] Conversation resolution required
- [ ] No stale reviews

---

## Troubleshooting

### "Status check failed: validate-landing-zone"

**Cause:** CI workflow failed (type check, lint, tests, security)

**Fix:**
1. Check workflow logs: https://github.com/kushin77/ollama/actions
2. Fix issues locally:
   ```bash
   mypy ollama/ --strict
   ruff check ollama/ --fix
   pytest tests/
   pip-audit
   ```
3. Push fixes: `git commit -S && git push`

### "Merge blocked: Requires signed commits"

**Cause:** Commits not GPG signed

**Fix:**
```bash
# Configure GPG signing
git config --global user.signingkey <GPG_KEY_ID>
git config --global commit.gpgsign true

# Sign existing commits
git rebase -i <base_branch> --gpg-sign

# Or configure signing per-repo
git config user.signingkey <GPG_KEY_ID>
git config commit.gpgsign true
```

### "Merge blocked: Requires approvals"

**Cause:** PR not approved or approval is stale

**Fix:**
1. Request review from code owner
2. If approval is stale: push a new commit to dismiss it
3. Wait for new approval

### "Force push rejected"

**Cause:** Attempted `git push --force` to `main`

**Fix:**
- Don't force push to `main` (use feature branches)
- For emergency fixes, contact repo admin to temporarily disable protection
- Reset through a regular PR instead

---

## Configuration Files

### Terraform
- **File:** `terraform/branch_protection.tf`
- **Purpose:** Infrastructure as code for branch protection
- **Updates:** Edit and run `terraform apply` to update

### GitHub Actions
- **File:** `.github/workflows/validate-landing-zone.yml`
- **Purpose:** CI/CD pipeline that provides status checks
- **Updates:** Edit and commit to enable new checks

### Shell Script
- **File:** `scripts/enable-branch-protection.sh`
- **Purpose:** One-command branch protection setup
- **Updates:** Edit for new requirements

---

## Best Practices

1. **Always sign commits:**
   ```bash
   git commit -S -m "message"
   ```

2. **Keep branches updated:**
   ```bash
   git pull origin main
   ```

3. **Use descriptive PR titles:**
   ```
   feat(api): add streaming response support
   fix(auth): resolve token expiration race condition
   ```

4. **Require reviews before merging:**
   - Assign reviewers in PR
   - Wait for at least 1 approval
   - Ensure all conversations are resolved

5. **Monitor status checks:**
   - Review CI logs immediately after push
   - Fix failures before requesting review
   - Re-run failed checks if they're flaky

---

## Next Steps

1. **Apply branch protection:**
   ```bash
   bash scripts/enable-branch-protection.sh
   ```

2. **Test the workflow:**
   - Create a feature branch
   - Push a commit
   - Verify CI status check runs
   - Create PR and verify review requirement

3. **Enable on other branches** (optional):
   - `develop` (staging/pre-release)
   - `release/*` (release branches)

4. **Configure branch deletion protection** (if using GitHub CLI):
   ```bash
   gh api -X PUT repos/kushin77/ollama/branches/main/protection \
     -f allow_deletions=false
   ```

---

## References

- [GitHub Branch Protection Docs](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/managing-a-branch-protection-rule)
- [GitHub CLI Docs](https://cli.github.com)
- [Terraform GitHub Provider](https://registry.terraform.io/providers/integrations/github/latest/docs/resources/branch_protection)
- [GCP Landing Zone Compliance](https://github.com/kushin77/GCP-landing-zone)

---

**Status:** ✅ Ready for deployment  
**Maintained by:** GitHub Copilot + kushin77  
**Last Updated:** 2026-01-30
