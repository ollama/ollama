# Git Hooks Setup & Configuration

**Status**: ✅ Implemented and Enforced
**Version**: 2.0.0
**Last Updated**: January 26, 2026
**Maintenance**: Critical security infrastructure

## Overview

Git hooks are automated scripts that run at specific points in the git workflow. Ollama uses hooks to enforce:

1. **Security scanning** (gitleaks: prevent secrets from being committed)
2. **Code quality** (type checking, linting, formatting)
3. **Commit message validation** (enforce elite standards)
4. **GPG signing** (immutable, auditable commits)
5. **Folder structure compliance** (5-level depth mandate)

## Quick Start

### Installation (First Time)

```bash
# Clone repository
git clone https://github.com/kushin77/ollama.git
cd ollama

# Install git hooks
bash .githooks/setup.sh

# Verify installation
ls -la .git/hooks/ | grep -E "(pre-commit|commit-msg)"
# Should show: pre-commit, commit-msg-validate, post-commit

# Verify output shows ✅ Git hooks installed successfully!
```

### Setup GPG Signing

```bash
# Check if GPG is configured
git config --list | grep gpg

# If not configured:
git config --global commit.gpgsign true
git config --global user.signingkey YOUR_GPG_KEY_ID

# Test signing
echo "test" > test.txt
git add test.txt
git commit -S -m "test(hooks): verify GPG signing"
git log --show-signature -1
# Should show: Good signature from "Your Name <email@domain.com>"
```

## What Hooks Do

### 1. Pre-Commit Hook (`.githooks/pre-commit`)

Runs **before** each commit with automatic checks:

| Check                 | Tool          | Purpose                               | Fail Action     |
| --------------------- | ------------- | ------------------------------------- | --------------- |
| **Secrets Detection** | gitleaks      | Prevent API keys, tokens, credentials | ❌ BLOCK commit |
| **Folder Structure**  | Custom script | Enforce 5-level depth + naming        | ❌ BLOCK commit |
| **Type Safety**       | mypy --strict | 100% type hint coverage               | ❌ BLOCK commit |
| **Code Linting**      | ruff          | Code quality standards                | ❌ BLOCK commit |
| **Code Formatting**   | black         | Auto-fix, then re-stage               | ✅ Auto-fix     |
| **Security Audit**    | pip-audit     | Check dependencies                    | ⚠️ Warn only    |
| **Unit Tests**        | pytest        | Run unit test suite                   | ❌ BLOCK commit |

**Flow:**

```
git commit
    ↓
[Pre-commit Hook Runs]
    ├─ 🔐 Gitleaks scan → No secrets? ✅
    ├─ 📁 Folder structure → Valid? ✅
    ├─ 📋 Type checking → mypy --strict passes? ✅
    ├─ 🔍 Linting → ruff passes? ✅
    ├─ 💅 Formatting → black fixes, re-stages ✅
    ├─ 🔐 Security audit → pip-audit (warn only)
    ├─ 🧪 Unit tests → pytest passes? ✅
    ↓
All pass → Commit proceeds ✅
Any fail → Commit rejected ❌
```

### 2. Commit Message Hook (`.githooks/commit-msg-validate`)

Validates commit message **format and GPG signing**:

| Check           | Rule                                                          | Example                    |
| --------------- | ------------------------------------------------------------- | -------------------------- |
| **Format**      | `type(scope): description`                                    | `feat(api): add streaming` |
| **Type**        | feat, fix, refactor, perf, test, docs, infra, security, chore | ✅ valid                   |
| **Scope**       | lowercase-with-hyphens, 15 chars max                          | `(cache)`, `(gcp-auth)`    |
| **Description** | Start with capital, max 50 chars                              | "Add caching layer"        |
| **Body**        | Blank line between subject and body                           | Required if body exists    |
| **GPG Signing** | Required for main/develop branches                            | `git commit -S`            |

**Examples:**

✅ **Good:**

```bash
git commit -S -m "feat(api): add streaming response support"
git commit -S -m "fix(auth): resolve token expiration race condition"
git commit -S -m "refactor(services): split inference into modules"
```

❌ **Bad (Rejected):**

```bash
git commit -m "Added new feature"              # ❌ No type
git commit -m "feat: add stuff"                 # ❌ No scope
git commit -m "feat(api): add streaming"        # ❌ Lowercase description
git commit -m "feat(api): Add streaming\nBody" # ❌ No blank line
```

### 3. Post-Commit Hook (`.githooks/post-commit`)

Verifies commit is GPG-signed (warning only, doesn't block):

```bash
# If commit unsigned on non-main branches:
# ⚠️  Warning: Commit is unsigned (GPG signing recommended)
#
# For main/develop branches:
# ✅ Good signature detected
```

## Secret Management

### What Secrets Are Detected

Gitleaks detects (high-entropy strings):

```
API Keys:           sk_live_*, sk_test_*, api_key=*
Tokens:             Bearer token*, access_token=*
AWS Credentials:    AKIAIOSFODNN7EXAMPLE, aws_secret_access_key=*
GCP:                service_account, private_key_id
Azure:              AccountKey=*, connection_string=*
Database:           password=*, mysql://*, postgres://
Private Keys:       -----BEGIN RSA PRIVATE KEY-----
```

### If You Accidentally Commit a Secret

**IMMEDIATE ACTIONS (Within seconds):**

```bash
# 1. Stop - do not push!
git reset --soft HEAD~1

# 2. Remove secret from all files
# (Edit files and remove sensitive data)

# 3. Re-stage and commit
git add .
git commit -S -m "fix(security): remove accidentally exposed secret"

# 4. Force push (ONLY to feature branch, never main/develop)
git push origin feature/branch-name --force

# 5. CRITICAL: Rotate the secret immediately!
# The secret was exposed in git history and must be invalidated
```

**Examples:**

```bash
# ❌ Committed API key in .env
git reset --soft HEAD~1
nano .env  # Remove: API_KEY=sk_live_123abc456
git add .
git commit -S -m "security: remove exposed API key"
git push origin feature/auth --force

# ❌ Committed GCP service account JSON
git reset --soft HEAD~1
rm config/gcp-sa-key.json  # Delete secret file
git add . && git rm config/gcp-sa-key.json
git commit -S -m "security: remove exposed GCP service account key"
git push origin feature/gcp --force
```

## GPG Signing Setup

### Install GPG

```bash
# macOS
brew install gpg

# Ubuntu/Debian
sudo apt-get install gnupg

# Fedora/RHEL
sudo dnf install gnupg

# Windows (WSL)
sudo apt-get install gnupg
```

### Generate or Import GPG Key

**Option 1: Generate New Key**

```bash
# Interactive setup
gpg --gen-key

# Follow prompts:
# - Name: Your Name
# - Email: your-email@example.com
# - Passphrase: Strong password (you'll type this for each commit)

# List your key ID
gpg --list-keys
# Example output:
# pub   rsa4096 2026-01-26 [SC]
#       ABC123DEF456789012345678
# uid           [ultimate] Your Name <your-email@example.com>

# Your Key ID is: ABC123DEF456789012345678 (or last 16 hex digits)
```

**Option 2: Import Existing Key**

```bash
# If you have a private key exported from another machine
gpg --import /path/to/private-key.asc

# Then list to find your key ID
gpg --list-keys
```

### Configure Git

**Global Configuration (All Repositories):**

```bash
git config --global commit.gpgsign true
git config --global user.signingkey ABC123DEF456789012345678
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"

# Verify
git config --list | grep -E "(gpgsign|signingkey)"
```

**Per-Repository Configuration (This Repository Only):**

```bash
cd /home/akushnir/ollama
git config --local commit.gpgsign true
git config --local user.signingkey ABC123DEF456789012345678

# Verify
git config --list | grep -E "(gpgsign|signingkey)"
```

### Test GPG Signing

```bash
# Create a test file
echo "test" > test.txt
git add test.txt

# Commit with signature (you'll be prompted for passphrase)
git commit -S -m "test(hooks): verify GPG setup"

# Verify signature was created
git log --show-signature -1
# Expected: Good signature from "Your Name <your-email@example.com>"

# Clean up test
git reset --soft HEAD~1
rm test.txt
git reset
```

## Troubleshooting

### Issue: Pre-commit Hook Fails

**Problem: `❌ Secrets detected`**

```bash
# Your file contains a potential secret
# Solution:
nano <file>  # Remove the secret
git add <file>
git commit -S -m "fix(security): remove secret"
```

**Problem: `❌ Type checking failed`**

```bash
# Missing type hints
# Solution:
mypy ollama/ --strict  # See detailed errors
# Add type hints to your code
git add .
git commit -S -m "fix(types): add type hints"
```

**Problem: `❌ Linting found issues`**

```bash
# Code style violations
# Solution:
ruff check ollama/ --fix  # Auto-fix what's possible
black ollama/  # Format code
git add .
git commit -S -m "style: fix linting issues"
```

**Problem: `❌ Unit tests failed`**

```bash
# Tests are failing
# Solution:
pytest tests/ -v  # See which tests fail
# Fix the failing tests
git add tests/
git commit -S -m "test: fix failing tests"
```

### Issue: Commit Message Hook Fails

**Problem: `❌ Invalid commit message format`**

```bash
# Your message doesn't match format
# Solution:
git commit --amend -S -m "feat(scope): valid message format"
```

**Problem: `❌ GPG signing required for main branch`**

```bash
# You didn't sign your commit on main/develop
# Solution:
git commit --amend -S  # This will re-open editor, save with -S flag
# Or for feature branches, it's optional (but recommended)
```

### Issue: GPG Not Working

**Problem: `fatal: cannot run gpg: No such file or directory`**

```bash
# GPG not installed
# Solution:
brew install gpg  # macOS
# or
sudo apt-get install gnupg  # Linux
```

**Problem: `error: skipped commit with no changes`**

```bash
# Signing created a new commit, nothing to amend
# Solution: This is normal, your commit is signed
git log --show-signature -1
```

**Problem: Passphrase prompt never appears**

```bash
# GPG agent issue
# Solution:
# Restart GPG agent
gpgconf --kill gpg-agent
gpgconf --launch gpg-agent

# Or configure GPG to use gpinentry
# Edit ~/.gnupg/gpg-agent.conf
# Add: pinentry-program /usr/bin/pinentry-curses
```

### Issue: Hook Installation Failed

**Problem: `.githooks` directory not found**

```bash
# The repository doesn't have .githooks
# Solution:
cd /home/akushnir/ollama
git status  # Verify you're in the right directory
bash .githooks/setup.sh
```

**Problem: Permission denied when running setup**

```bash
# setup.sh is not executable
# Solution:
chmod +x .githooks/setup.sh
bash .githooks/setup.sh
```

## Bypassing Hooks (Emergency Only)

**IMPORTANT**: Never bypass security hooks unless absolutely necessary. Discuss with team first.

```bash
# Bypass ALL hooks (not recommended)
git commit --no-verify

# This skips:
# - Gitleaks scanning
# - Type checking
# - Linting
# - Tests
# - GPG validation

# If you had to use --no-verify:
# 1. Notify the security team
# 2. Explain why
# 3. Schedule a code review immediately
# 4. Consider why the hook was blocking (likely a good reason!)
```

## Hook Maintenance

### Update Hooks to Latest Version

```bash
# Hooks are part of the repository
# They update automatically when you pull:
git pull origin main

# Then reinstall:
bash .githooks/setup.sh
```

### Verify Hooks Are Current

```bash
# Check hook timestamps
ls -la .git/hooks/ | grep -E "(pre-commit|commit-msg)"

# Check source timestamps
ls -la .githooks/ | grep -E "(pre-commit|commit-msg)"

# If .git/hooks are older, reinstall:
bash .githooks/setup.sh
```

### Check Hook Versions

```bash
# View current pre-commit hook
head -20 .git/hooks/pre-commit

# View latest from repository
head -20 .githooks/pre-commit

# If different, run setup again
bash .githooks/setup.sh
```

## Related Documentation

- [CONTRIBUTING.md](./CONTRIBUTING.md) - Full contribution guide
- [Git Commit Standards](./.github/copilot-instructions.md#git-commit-standards) - Commit message standards
- [GCP Landing Zone](./.github/copilot-instructions.md#gcp-landing-zone-compliance-mandate) - Compliance requirements
- [Security Audit](./SECURITY_AUDIT.md) - Security best practices

## Support

### Common Questions

**Q: Do I need to sign every commit?**

- A: Yes, for main/develop branches. Feature branches encouraged but optional.

**Q: Why do tests run in pre-commit?**

- A: To prevent broken code from being committed. Tests should pass locally before pushing.

**Q: Can I disable gitleaks?**

- A: Not recommended, but yes: edit `.githooks/pre-commit` and comment out gitleaks section.

**Q: What if gitleaks has false positives?**

- A: Add to `.gitignore` or `.gitleaksignore` file. Report to security team.

**Q: How do I generate a stronger passphrase?**

- A: Use a 20+ character password with mixed case, numbers, and symbols.

### Escalation

For issues or questions:

1. Check this guide's troubleshooting section
2. Run: `bash .githooks/setup.sh` to reinstall
3. Check git configuration: `git config --list`
4. Post in #engineering Slack channel

---

**Last Updated**: January 26, 2026
**Maintained By**: Security & Infrastructure Team
**Status**: ✅ Actively Maintained
