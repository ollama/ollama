# Post-Merge Activation Checklist

**Objective:** Merge PR #78 and PR #79, then activate branch protection  
**Owner:** kushin77 (Code Owner)  
**Timeline:** ~30 minutes  
**Risks:** Low (PR #78 is test-only; #79 is infra setup, not enforcement yet)

---

## Pre-Merge Verification (5 min)

- [ ] **PR #78 (Test Fixes)**
  - [ ] Read full description
  - [ ] Verify all 24 tests passed locally
  - [ ] Check commits are GPG-signed: `git log --show-signature`
  - [ ] Confirm no breaking changes to public APIs
  - [ ] Review follow-up tasks in PR body

- [ ] **PR #79 (Branch Protection)**
  - [ ] Read full description
  - [ ] Review Terraform configuration
  - [ ] Review GitHub CLI script
  - [ ] Check setup guide for completeness
  - [ ] Confirm no hardcoded secrets in terraform files

- [ ] **CI Status Checks**
  - [ ] Both PRs pass GitHub Actions (validate-landing-zone workflow)
  - [ ] No security warnings or vulnerabilities

---

## Merge Process (5 min)

### 1. Merge PR #78 (Test Fixes)

```bash
cd /home/akushnir/ollama

# Option A: Using GitHub CLI
gh pr merge 78 --merge --delete-branch

# Option B: Using git (if needed)
git checkout main
git pull origin main
git merge --no-ff fix/test-collection-and-shims
git push origin main
```

**Verify:**
```bash
git log --oneline main | head -5
# Should show: "fix(test): enable agent instantiation..." at top
```

### 2. Merge PR #79 (Branch Protection)

```bash
# Option A: Using GitHub CLI
gh pr merge 79 --merge --delete-branch

# Option B: Using git
git pull origin main
git merge --no-ff infra/enable-branch-protection
git push origin main
```

**Verify:**
```bash
git log --oneline main | head -10
# Should show both commits at top in order
```

---

## Post-Merge Verification (5 min)

- [ ] Both PRs merged and deleted
- [ ] Branch `main` has both commits:
  ```bash
  git log --oneline -n 2
  # infra(branch-protection): add automation and documentation
  # fix(test): enable agent instantiation and add compatibility shims
  ```
- [ ] Both commits are GPG-signed:
  ```bash
  git log --show-signature -n 2 | grep -E "Good signature|Bad signature"
  ```
- [ ] Local copy updated:
  ```bash
  git checkout main && git pull origin main
  ```

---

## Branch Protection Activation (10 min)

### Option 1: GitHub CLI (Recommended)

```bash
bash scripts/enable-branch-protection.sh
```

**Output should show:**
```
🔒 Enabling branch protection for kushin77/ollama:main
✅ Repository access verified
📋 Applying branch protection rules...
✅ Branch protection enabled!
```

### Option 2: Terraform

```bash
cd terraform
export GITHUB_TOKEN="your_token_here"
terraform init
terraform plan
terraform apply
```

### Verify Protection Applied

```bash
# Check via GitHub API
gh api repos/kushin77/ollama/branches/main/protection

# Or via GitHub UI
# https://github.com/kushin77/ollama/settings/branches
```

**Expected output includes:**
```json
{
  "enforce_admins": true,
  "require_signed_commits": true,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true
  },
  "required_status_checks": {
    "strict": true,
    "contexts": ["validate-landing-zone"]
  }
}
```

---

## Test Branch Protection (5 min)

1. **Create a test feature branch:**
   ```bash
   git checkout -b test/branch-protection
   echo "test" > test.txt
   git add test.txt
   git commit -S -m "test: verify branch protection"
   git push origin test/branch-protection
   ```

2. **Create a test PR:**
   ```bash
   gh pr create --base main --head test/branch-protection --title "test: verify branch protection" --body "Testing branch protection enforcement"
   ```

3. **Verify protection is active:**
   - [ ] PR shows "❌ Some checks haven't completed yet"
   - [ ] Hovering shows "validate-landing-zone" status check required
   - [ ] "Merge" button is disabled (red)
   - [ ] Requires at least 1 approval to proceed

4. **Test signing requirement:**
   - [ ] Try to force-push: `git push --force` → should be rejected
   - [ ] Try to push unsigned commit → validate-landing-zone workflow will fail

5. **Clean up test branch:**
   ```bash
   gh pr close <PR_NUMBER> --delete-branch
   git branch -D test/branch-protection
   ```

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "Merge button disabled even after approval" | Status checks not complete | Wait for validate-landing-zone workflow to finish or re-run |
| "Error: branch protection not found" | API call failed | Check token permissions and retry |
| "Force push succeeded" | Protection not applied | Run activation script again, verify with API |
| "Terraform state conflict" | Multiple appliers | Use Terraform lock: `terraform state lock` |

---

## Post-Activation Announcements

Once branch protection is active, consider:

1. **Update contributing guide** (if applicable)
   ```markdown
   # Branch Protection Rules
   All PRs to `main` must:
   - [ ] Pass all status checks (CI/CD)
   - [ ] Have ≥1 approval from code owner
   - [ ] Use GPG-signed commits
   - [ ] Resolve all conversations
   ```

2. **Notify team in Slack/Teams:**
   ```
   🔒 Branch protection now active on kushin77/ollama:main

   All merges now require:
   ✅ PR review (1 approval)
   ✅ GPG-signed commits
   ✅ Status checks pass (mypy, ruff, pytest, security, ...)
   ✅ No force pushes
   ✅ No deletions

   See: docs/BRANCH_PROTECTION_SETUP.md for setup and troubleshooting
   ```

3. **Update README** with branch protection badge:
   ```markdown
   [![Branch Protection](https://img.shields.io/badge/branch%20protection-active-green)](docs/BRANCH_PROTECTION_SETUP.md)
   ```

---

## Sign-Off

- [ ] **Code Owner Review:** Approve PR #78 and PR #79
- [ ] **Merge:** Execute both merges
- [ ] **Activation:** Run branch protection setup
- [ ] **Verification:** Confirm protection with test branch
- [ ] **Documentation:** Update contributing guide (if needed)
- [ ] **Announcement:** Notify team of new requirements

**Date Completed:** ________________  
**Completed By:** ________________  
**Verified By:** ________________  

---

## Next Phase: Legacy Module Migration

After branch protection is active, begin phase 3:

1. Identify lowest-risk legacy module
2. Refactor into proper `ollama/` package
3. Add comprehensive tests
4. Create micro-PR for review
5. Merge and close legacy module
6. Repeat for remaining modules

See: `docs/ONBOARDING_COMPLETE.md` for roadmap

---

*Last Updated: 2026-01-30*  
*Estimated Duration: 30 minutes*  
*Maintenance Window: None required (GitHub-side only)*
