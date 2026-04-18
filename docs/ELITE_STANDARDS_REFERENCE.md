# Elite Standards Quick Reference

This is a quick reference guide for the elite development standards enforced in the Ollama project.

## ⚡ Quick Commands

### Before Your First Commit

```bash
# 1. Clone and setup
git clone <repo>
cd ollama
python3 -m venv venv
source venv/bin/activate
pip install -r requirements/dev.txt

# 2. Setup git hooks
bash scripts/setup-git-hooks.sh

# 3. Install VS Code extensions
code --install-extension GitHub.copilot
code --install-extension ms-python.python
code --install-extension charliermarsh.ruff

# 4. Configure GPG signing (one-time)
gpg --list-secret-keys
git config user.signingkey <YOUR_KEY_ID>
```

### Making a Commit

```bash
# Format: type(scope): description (≤50 chars, capital letter)
# Always use -S flag for GPG signing

git add .
git commit -S -m "feat(api): add conversation endpoint"
# Or use git config alias.sc 'commit -S' first, then: git sc -m "message"
```

### Pushing Code

```bash
# Branch name must be: {type}/{name-with-hyphens}
git push origin feature/add-conversation-api

# Hooks will validate:
# ✓ Branch name format
# ✓ All tests pass
# ✓ Type checking passes
# ✓ Linting passes
```

---

## 📋 Commit Message Format

**REQUIRED**: `type(scope): description`

### Valid Types
- `feat` - New feature (increases minor version)
- `fix` - Bug fix (increases patch version)
- `refactor` - Code refactoring (no behavior change)
- `perf` - Performance improvement
- `test` - Test additions/modifications
- `docs` - Documentation updates
- `infra` - Infrastructure/CI/CD/Docker
- `security` - Security-related changes
- `chore` - Maintenance/dependency updates

### Valid Scopes (examples)
- `api` - API routes and endpoints
- `auth` - Authentication and authorization
- `models` - ML model integration
- `db` - Database and repositories
- `cache` - Redis and caching
- `docker` - Docker and containerization
- `infra` - Infrastructure
- `security` - Security features

### Examples

✅ **CORRECT**:
```
feat(api): add conversation endpoint with streaming

Add support for server-sent events (SSE) in conversation endpoint.
Clients can now receive partial responses as tokens are generated.

Implements RFC-001
```

✅ **CORRECT**:
```
fix(auth): resolve token expiration race condition

Use atomic compare-and-swap for token updates to prevent duplicates.
```

✅ **CORRECT**:
```
perf(inference): optimize batch processing for 2x throughput
```

❌ **WRONG**:
```
Update code                          # No type
feat: add conversation endpoint      # No scope
Feature(API): add endpoint           # Type not lowercase, scope not lowercase
feat(api): add conversation endpoint to handle conversations better  # Subject > 50 chars
```

---

## 🌿 Branch Naming

**REQUIRED**: `{type}/{descriptive-name}`

### Allowed Types
- `feature/` - New features
- `bugfix/` - Bug fixes
- `refactor/` - Code refactoring
- `infra/` - Infrastructure changes
- `security/` - Security fixes
- `docs/` - Documentation

### Rules
- Lowercase only
- Hyphens for word separation (no underscores)
- Max 40 characters after type
- Descriptive but concise

### Examples

✅ **CORRECT**:
```
feature/add-conversation-api
bugfix/fix-token-refresh-race
refactor/simplify-model-loading
security/add-rate-limiting
docs/update-deployment-guide
```

❌ **WRONG**:
```
Feature/add-conversation-api        # Type not lowercase
feature/AddConversationApi          # camelCase not allowed
feature/add_conversation_api        # underscores not allowed
feature/add-new-awesome-endpoint-that-does-conversations  # too long
```

---

## 🐍 Python Code Standards

### Type Hints - MANDATORY

✅ **CORRECT**:
```python
from typing import Optional, List
from dataclasses import dataclass

@dataclass
class User:
    id: str
    name: str
    email: Optional[str] = None

def get_user(user_id: str) -> Optional[User]:
    """Retrieve user by ID.

    Args:
        user_id: The unique user identifier.

    Returns:
        User object if found, None otherwise.

    Raises:
        ValueError: If user_id is empty.
    """
    if not user_id:
        raise ValueError("user_id cannot be empty")
    # Implementation...
    return None

async def process_users(users: List[User]) -> dict[str, int]:
    """Process multiple users."""
    # Implementation...
    return {}
```

❌ **WRONG**:
```python
def get_user(user_id):  # No type hints!
    """Get user."""
    return None

def process_users(users):  # No types!
    # Implementation...
    pass
```

### Docstrings - REQUIRED

Format: Google-style with Examples

```python
def validate_email(email: str) -> bool:
    """Validate email format using RFC 5322 standard.

    This function checks if the provided email address follows
    the basic RFC 5322 format for valid email addresses.

    Args:
        email: The email address to validate.

    Returns:
        True if email is valid, False otherwise.

    Raises:
        ValueError: If email is None or not a string.

    Example:
        >>> validate_email("user@example.com")
        True
        >>> validate_email("invalid-email")
        False
    """
    if not isinstance(email, str):
        raise ValueError("email must be a string")
    return "@" in email and "." in email.split("@")[1]
```

### Test Coverage - REQUIRED

- **Minimum**: ≥90% overall
- **Critical paths**: 100% coverage
- **File location**: `tests/unit/test_{module_name}.py`

```python
def test_validate_email_valid():
    """Valid email passes validation."""
    assert validate_email("user@example.com") is True

def test_validate_email_invalid():
    """Invalid email fails validation."""
    assert validate_email("invalid-email") is False
    assert validate_email("") is False

def test_validate_email_none_raises():
    """None email raises ValueError."""
    with pytest.raises(ValueError):
        validate_email(None)
```

---

## 🔧 Pre-Commit Checks

Automatically run before commit:

1. **Type Checking**: `mypy ollama/ --strict`
   - 100% type coverage required
   - Shows error codes and line numbers

2. **Linting**: `ruff check ollama/`
   - Code style issues detected
   - Auto-fixed when possible

3. **Formatting**: `black` + `isort`
   - 100 character line length
   - Imports automatically sorted

4. **Security**: `pip-audit` + `bandit`
   - Dependency vulnerabilities scanned
   - Code security issues detected

5. **Code Quality**:
   - No debug `print()` statements
   - No `TODO` in production code
   - No trailing whitespace

### Skip Checks (NOT RECOMMENDED)
```bash
git commit --no-verify  # Only in emergencies!
```

---

## 🧪 Testing

### Run Tests
```bash
pytest tests/ -v --cov=ollama --cov-report=term-missing
```

### Run Type Checking
```bash
mypy ollama/ --strict --show-error-codes
```

### Run Linting
```bash
ruff check ollama/ --fix
```

### Run All Checks
```bash
bash scripts/run-all-checks.sh
# Or use VS Code task: Ctrl+Shift+B → "Run All Checks"
```

---

## 🔐 GPG Signing

### Setup (One-time)

```bash
# List your keys
gpg --list-secret-keys

# Configure git
git config user.signingkey <YOUR_KEY_ID>
git config commit.gpgsign true

# Create alias for easier signing
git config --global alias.sc 'commit -S'
```

### Making Signed Commits

```bash
# Using -S flag
git commit -S -m "feat(api): add endpoint"

# Or using alias (if configured)
git sc -m "feat(api): add endpoint"
```

### Verify Signature
```bash
git log --show-signature -1
```

---

## 🚀 Development Workflow

### 1. Create Feature Branch
```bash
git checkout -b feature/my-feature
```

### 2. Make Changes
```bash
# Edit files, run tests
python3 -m pytest tests/ -v
```

### 3. Commit with Signature
```bash
git add .
git commit -S -m "feat(scope): description"
# Hooks run automatically:
# ✓ Type checking
# ✓ Linting
# ✓ Formatting
# ✓ Security audit
```

### 4. Push to Remote
```bash
git push origin feature/my-feature
# Hooks run:
# ✓ Branch name validation
# ✓ Full test suite
# ✓ Type checking
# ✓ Linting
```

### 5. Create Pull Request
```bash
# GitHub CLI
gh pr create --base main --head feature/my-feature

# Web: Create PR with template from .github/pull_request_template.md
```

### 6. Address Review Comments
```bash
# Make changes
git add .
git commit -S -m "fix(scope): address review comments"
git push origin feature/my-feature
```

### 7. Merge When Approved
```bash
# Via GitHub (recommended for clean history)
# Or locally:
git checkout main
git pull origin main
git merge --no-ff feature/my-feature
git push origin main
```

---

## 📊 Code Quality Standards

| Metric | Requirement | Tools |
|--------|-------------|-------|
| Type Coverage | 100% | mypy --strict |
| Test Coverage | ≥90% (100% critical) | pytest --cov |
| Line Length | ≤100 characters | black --line-length 100 |
| Imports | Sorted | isort --profile black |
| Code Style | Black standard | black |
| Linting | Ruff rules | ruff check |
| Security | No vulnerabilities | pip-audit, bandit |
| Complexity | ≤10 cognitive | flake8-cognitive-complexity |

---

## 🆘 Troubleshooting

### Commit Hook Failed

```bash
# View error
git commit -m "test"

# Fix issues (e.g., type checking)
python3 -m mypy ollama/ --strict

# Retry commit
git add .
git commit -S -m "feat(scope): description"
```

### Branch Name Invalid

```bash
# Current branch
git rev-parse --abbrev-ref HEAD

# Rename
git branch -m old-name feature/new-name

# Push again
git push origin feature/new-name
```

### Tests Failing

```bash
# Run tests locally
pytest tests/ -v --tb=short

# Fix code
# Commit changes
git add .
git commit -S -m "fix(test): correct failing test"
```

### Type Checking Errors

```bash
# See all errors
python3 -m mypy ollama/ --strict

# Fix type hints
# Verify fix
python3 -m mypy ollama/module/file.py --strict
```

---

## 📚 Resources

- **Main Instructions**: [.github/copilot-instructions.md](.github/copilot-instructions.md)
- **Copilot Integration**: [.github/COPILOT_INTEGRATION.md](.github/COPILOT_INTEGRATION.md)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **Git Hooks**: [.githooks/](.githooks/)

---

## Version

- **Elite Standards**: v2.0.0
- **Last Updated**: January 13, 2026

For detailed guidance, see the full copilot-instructions.md document.
