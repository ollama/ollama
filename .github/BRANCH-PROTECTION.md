# GitHub Branch Protection & Governance

## Branch Protection Rules

These rules enforce FAANG standards on all pull requests before merge to `main` or `develop`.

### Setup Instructions

**Step 1**: Navigate to Repository Settings

```
GitHub → Repository → Settings → Branches
```

**Step 2**: Add Branch Protection Rule

```
Click "Add rule" → Enter pattern: "main"
```

**Step 3**: Configure Protection Rule

```yaml
Branch name pattern: main

✅ Require a pull request before merging
  ├─ Dismiss stale pull request approvals when new commits are pushed
  ├─ Require review from code owners
  └─ Restrict who can dismiss pull request reviews
     ├─ Allow dismissal by pull request author
     └─ Required number of approvals: 1 (min), 2 (recommended)

✅ Require status checks to pass before merging
  ├─ Require branches to be up to date before merging
  ├─ Require status checks:
  │  ├─ Type Checking (mypy --strict)
  │  ├─ Linting (ruff check)
  │  ├─ Testing (pytest ≥95% coverage)
  │  ├─ Security Audit (pip-audit + bandit)
  │  └─ Standards Validation (validate-standards.py)
  │
✅ Require code reviews (minimum 1 reviewer)
  ├─ Dismiss stale reviews on push
  ├─ Require CODEOWNERS review
  └─ Require latest commit approval

✅ Require signed commits

✅ Include administrators
  └─ Apply rules to administrators also (non-negotiable)

✅ Allow force pushes
  └─ Allows force pushes from everyone (consider restricting)

✅ Allow deletions
  └─ Unchecked (prevent accidental branch deletion)

✅ Require conversation resolution before merging
  └─ All comments must be resolved

✅ Require deployments to succeed before merging
  └─ Required deployments: staging (optional)
```

### CODEOWNERS File

Create `.github/CODEOWNERS`:

```
# Global code owners (default reviewers for all PRs)
* @kushin77 @team-leads

# Core systems
/ollama/api/ @team-api
/ollama/services/ @team-backend
/ollama/repositories/ @team-database
/ollama/models.py @team-ml

# Critical paths
/ollama/exceptions.py @kushin77
/ollama/main.py @kushin77
/scripts/ @team-devops

# Configuration & deployment
/docker/ @team-devops
/k8s/ @team-devops
/.github/ @team-leads
pyproject.toml @kushin77

# Documentation
/docs/ @team-docs
README.md @kushin77
```

### Rule Template (Terraform/IaC)

```hcl
# terraform/github/branch-protection.tf

resource "github_branch_protection" "main" {
  repository_id            = github_repository.ollama.id
  pattern                  = "main"
  require_signed_commits   = true
  required_status_checks {
    strict   = true
    contexts = [
      "Type Checking (mypy --strict)",
      "Linting (ruff check)",
      "Testing (pytest ≥95% coverage)",
      "Security Audit (pip-audit + bandit)",
    ]
  }
  required_pull_request_reviews {
    dismiss_stale_reviews           = true
    restrict_dismissals             = false
    require_code_owner_reviews      = true
    required_approving_review_count = 1
  }
  enforce_admins = true
  allows_deletions = false
  allows_force_pushes = false
  require_conversation_resolution = true
}
```

---

## Pull Request Review Workflow

### Automatic Workflow

1. **Developer Creates PR**

   - All 3 enforcement layers activate automatically
   - Pre-commit hooks run on local machine
   - CI/CD workflows run on GitHub

2. **Status Checks Run** (~2 minutes)

   - ✅ Type checking: `mypy ollama/ --strict`
   - ✅ Linting: `ruff check ollama/`
   - ✅ Testing: `pytest ≥95% coverage`
   - ✅ Security: `pip-audit`, `bandit`
   - ❌ All must pass (blocking)

3. **Human Code Review**

   - Assigned reviewers notified
   - Review with CODE-REVIEW-CHECKLIST
   - PR comments added for feedback

4. **Conversation Resolution**

   - Developer addresses feedback
   - Comments marked as resolved
   - Status checks re-run if changes made

5. **Approval & Merge**
   - Minimum 1 approval required
   - All status checks passing ✅
   - All conversations resolved ✅
   - Can merge to main

### Manual Workflow

**If automatic checks fail:**

```bash
# Developer fixes locally
pytest tests/ --cov=ollama --cov-fail-under=95  # Fix coverage
mypy ollama/ --strict                            # Fix types
ruff check ollama/ --fix                         # Fix linting
git add .
git commit -S -m "fix: address review feedback"
git push origin feature/my-feature

# Checks automatically re-run
# Status shown in GitHub PR
```

---

## Approval Authority

### Required Approvers by Change Type

| Change Type      | Approvers                | Why                           |
| ---------------- | ------------------------ | ----------------------------- |
| Documentation    | 1 (any)                  | Low risk, broad expertise     |
| Tests/CI         | 1 (team-leads)           | Affects all developers        |
| API Changes      | 2 (1 API, 1 leads)       | High impact, public interface |
| Database Changes | 2 (1 database, 1 leads)  | Data consistency risk         |
| Security Changes | 2 (security, leads)      | Must be vetted                |
| Core Modules     | 2 (domain expert, leads) | Critical systems              |
| Configuration    | 1 (ops or leads)         | Infrastructure impact         |

### Reviewer Responsibilities

**Checklist**:

- [ ] Automatic checks all pass ✅
- [ ] Code quality standards met (see CODE-REVIEW-CHECKLIST)
- [ ] Test coverage ≥95%
- [ ] Type safety 100% (mypy --strict)
- [ ] Documentation updated
- [ ] No security vulnerabilities
- [ ] Performance impact acceptable

**Feedback**:

````markdown
# Code Review Comment

**Issue**: Missing type hint on function parameter

**Current**:

```python
def process_data(items):
    return [x.value for x in items]
```
````

**Required**:

```python
def process_data(items: list[DataItem]) -> list[Any]:
    return [x.value for x in items]
```

**Reason**: FAANG standards require 100% type coverage (mypy --strict)

**Reference**: .github/FAANG-ELITE-STANDARDS.md (TIER 1)

```

---

## Merge Strategies

### Squash Merge (Default)
```

Recommended for: Feature branches, bug fixes
Result: Single commit on main
Benefit: Clean history, atomic changes

```

### Rebase Merge
```

Recommended for: Refactoring, documentation
Result: All commits preserved, rebased on main
Benefit: Full context in history

```

### Merge Commit
```

Recommended for: Major releases, merges
Result: Merge commit created
Benefit: Explicit merge point

```

### Policy
```

✅ Use squash merge by default (one feature = one commit)
✅ Use rebase merge for refactoring (preserve commit history)
❌ Avoid merge commits (unnecessary clutter)

````

---

## GitHub Actions Status Required

All of these must pass (green ✅) before merge:

```yaml
Required Checks:
├─ Type Checking (mypy --strict)      ✅ REQUIRED
├─ Linting (ruff check)               ✅ REQUIRED
├─ Testing (pytest ≥95% coverage)     ✅ REQUIRED
├─ Security Audit (pip-audit + bandit) ✅ REQUIRED
└─ Code Review (1+ approvals)         ✅ REQUIRED

Optional Checks:
├─ Performance Regression (if applicable)
├─ Documentation Build (if docs changed)
└─ Deployment Preview (if applicable)
````

---

## Protecting Against Common Issues

### Issue: Stale Branch Blocks Merge

```
Solution: Check "Require branches to be up to date before merging"
Action: GitHub automatically disables merge button if branch out of sync
Fix: Developer clicks "Update branch" button
```

### Issue: Developer Force Pushes and Loses Review

```
Solution: Set "Dismiss stale pull request approvals when new commits are pushed"
Action: Force push invalidates previous approvals
Fix: Request another approval (typically quick second review)
```

### Issue: Unsigned Commits Bypass Protection

```
Solution: Check "Require signed commits"
Action: Blocks merge if any commit lacks GPG signature
Fix: Developer signs all commits locally, force pushes with GPG key
```

### Issue: Tests Pass Locally But Fail in CI

```
Solution: Use same Python version in CI as local (3.11, 3.12 tested)
Action: Run `pytest` with same configuration as CI
Fix: Check GitHub Actions workflow to match local setup
```

### Issue: Code Owner Doesn't Approve

```
Solution: Use CODEOWNERS file to auto-request reviews
Action: GitHub automatically requests CODEOWNERS review on PRs
Fix: Discuss with code owner if approval taking too long
```

---

## Troubleshooting

### Status Check Taking Too Long

```
Expected time: <2 minutes
If longer:
1. Check GitHub Actions logs
2. Look for slow tests (pytest -v --durations=10)
3. Check type checking time (mypy --profile)
4. Profile and optimize
```

### Can't Push to Main

```
Reason: Branch protection prevents direct pushes
Solution: Use pull request instead
Steps:
  1. git checkout -b feature/my-feature
  2. Make changes
  3. git push origin feature/my-feature
  4. Create PR on GitHub
  5. Wait for checks + review + approval
  6. Merge from GitHub
```

### Need to Bypass Protection (Emergency Only)

```
Rare case: Production incident requiring immediate fix
Process:
  1. Contact team leads
  2. Disable protection temporarily (admin action)
  3. Merge emergency fix
  4. Re-enable protection immediately
  5. Document incident in postmortem
```

---

## Governance & Policy

### Decision Rights

```
Code Changes:
├─ Code changes: Team lead or domain expert approval
├─ Security changes: Security team + team lead
├─ API changes: API lead + team lead
└─ Infrastructure: DevOps team

Documentation:
├─ README updates: Any maintainer
├─ API docs: API team
└─ Architecture: Tech leads

Configuration:
├─ CI/CD: DevOps team
├─ Secrets: Security team
└─ Deployments: Ops team
```

### Escalation Path

```
Problem → Resolution:
1. Automatic check fails → Fix code locally
2. Reviewer requests changes → Address feedback
3. Disagreement on approach → Tech lead decision
4. Blocked on approval → Escalate to team lead
5. Emergency override needed → Contact kushin77
```

### Policy Enforcement

```
✅ Automated (GitHub enforces):
  - Status checks must pass
  - Signed commits required
  - Minimum reviews required
  - Conversations resolved
  - Up-to-date branch required

✅ Manual (Reviewers enforce):
  - Code quality standards
  - Test coverage ≥95%
  - Type safety 100%
  - Documentation complete
  - Security best practices
```

---

**Last Updated**: January 14, 2026
**Status**: 🟢 Active
**Maintained By**: @kushin77
