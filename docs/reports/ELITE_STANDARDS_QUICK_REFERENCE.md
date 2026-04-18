# Elite Standards Quick Reference

**Last Updated**: January 13, 2026
**Version**: 1.0.0

## ⚡ TL;DR - Daily Commands

```bash
# Initial setup (one time)
bash scripts/setup-git-hooks.sh

# Create feature branch
git checkout -b feature/short-name

# Work and commit (every 30 min)
git add .
git commit -m "type(scope): description"
# Automatic checks run here!

# Push (every 4 hours max)
git push origin feature/short-name

# Create PR
# - GitHub templates auto-populate
# - Verify checklist ✅
# - Request review
# - Merge after approval
```

## 📋 Commit Message Format

```
type(scope): description

Body paragraph explaining WHAT and WHY.
Not HOW. Max 72 chars per line.

Fixes #123
```

### Valid Types
- `feat` - New feature
- `fix` - Bug fix
- `refactor` - Code refactoring
- `perf` - Performance improvement
- `test` - Test changes
- `docs` - Documentation
- `infra` - Infrastructure/CI
- `security` - Security fix
- `chore` - Maintenance

### Scope Examples
- `api` - API routes
- `auth` - Authentication
- `models` - ML models
- `db` - Database
- `cache` - Redis/caching
- `docker` - Docker/containers
- `k8s` - Kubernetes
- `monitoring` - Logging/metrics

### Examples
```
feat(api): add streaming endpoint
fix(auth): resolve token race condition
refactor(services): split inference module
perf(cache): optimize model loading by 40%
test(auth): add edge case tests
docs(readme): update deployment steps
infra(docker): upgrade to Python 3.12
security(cors): restrict origins to allowlist
```

## 🔧 Code Standards

### Types (MANDATORY)
```python
# ❌ NO
def process(data):
    return data

# ✅ YES
def process(data: dict) -> dict:
    """Process input data."""
    return data
```

### Functions (Max 100 lines)
```python
# ❌ NO: 200 lines, does 5 things
def do_everything(config, data, db, cache, email):
    # Validate, store, cache, email, log...
    pass

# ✅ YES: One responsibility
def validate_data(data: dict) -> ValidData:
    """Validate input data."""
    return ValidData(**data)

def persist_data(data: ValidData) -> StoredData:
    """Store data in database."""
    return db.insert(data)

async def notify_user(data: StoredData) -> None:
    """Send notification to user."""
    await email.send(data.user_id)
```

### Imports Organization
```python
# Stdlib
from typing import Optional
from pathlib import Path
import logging

# Third-party
from fastapi import APIRouter
from pydantic import BaseModel

# Local
from ollama.exceptions import ValidationError
from ollama.repositories.user import UserRepository
```

### Docstrings
```python
def generate_key(prefix: str = "sk") -> str:
    """Generate cryptographically secure API key.

    Args:
        prefix: Key prefix (default: "sk")

    Returns:
        Random key with prefix

    Raises:
        ValueError: If prefix invalid

    Example:
        >>> key = generate_key("token")
        >>> key.startswith("token-")
        True
    """
    return f"{prefix}-{secrets.token_hex(24)}"
```

### Error Handling
```python
# ❌ NO: Bare except
try:
    do_something()
except:
    pass

# ✅ YES: Specific exceptions
try:
    response = model.generate(prompt)
except ModelNotFoundError as e:
    log.error("model_not_found", model=e.model_name)
    raise APIError(code="MODEL_NOT_FOUND", status_code=404)
except TimeoutError as e:
    log.warning("inference_timeout", elapsed=e.elapsed)
    raise APIError(code="TIMEOUT", status_code=504)
```

## 🧪 Testing

### Structure
```
app/services/auth.py           # Implementation
tests/unit/test_auth.py        # Unit tests
tests/integration/test_auth.py # Integration tests
```

### Minimum Coverage
- Overall: ≥90%
- Critical paths: 100%
- New code: Must be tested

### Test Format
```python
class TestGenerateKey:
    """Tests for generate_key function."""

    def test_valid_generation(self) -> None:
        """Key generation produces valid format."""
        key = generate_key(prefix="sk")
        assert key.startswith("sk-")
        assert len(key) == 48

    def test_custom_prefix(self) -> None:
        """Custom prefix is preserved."""
        key = generate_key(prefix="token")
        assert key.startswith("token-")

    def test_invalid_prefix(self) -> None:
        """Invalid prefix raises ValueError."""
        with pytest.raises(ValueError):
            generate_key(prefix="invalid!")
```

## 🔐 Security

### Never Commit
- Passwords, tokens, API keys
- AWS/GCP credentials
- Database connection strings
- Private keys

### Use Instead
- `.env.example` for template
- Environment variables
- Secrets manager
- `.env` (gitignored)

### Code Review
- No hardcoded secrets
- Input validation present
- Output sanitized
- Error messages don't leak info

## 📊 Performance

### Baseline Targets
- API response: <500ms p99
- Model inference: Per-model baseline
- Startup time: <10s
- Memory: <2GB (excluding models)
- DB queries: <100ms p95

### Optimization Checklist
- [ ] Profiled before optimizing
- [ ] Benchmark before/after
- [ ] Tests still pass
- [ ] Documentation updated
- [ ] No new vulnerabilities

## 🎯 Pre-Commit Checklist

Before running `git commit`:

- [ ] Code formatted: `black --line-length=100 .`
- [ ] Linting passes: `ruff check ollama/`
- [ ] Types check: `mypy ollama/ --strict`
- [ ] Tests pass: `pytest tests/ -v`
- [ ] Coverage ≥90%: `pytest --cov=ollama`
- [ ] Security clean: `pip-audit`

**Or just commit and let hooks run!** 🚀

## 🐛 Debugging Hooks

### Pre-commit hook not running?
```bash
# Verify hooks path
git config core.hooksPath
# Should output: .husky

# Reconfigure
bash scripts/setup-git-hooks.sh
```

### Commit message rejected?
Check format:
```bash
✅ feat(api): add endpoint          # Valid
❌ Add endpoint feature             # Invalid
❌ feat(api-routes): add endpoint   # Scope too long
❌ feat(api): add endpoint          # Description too short (>50 chars)
```

### Type errors before commit?
```bash
# Install missing types
python -m pip install types-<package>

# Or add to pyproject.toml:
[tool.mypy]
ignore_missing_imports = false
```

### Tests failing before push?
```bash
# Run locally first
pytest tests/ -v --cov=ollama

# Fix failures
# Then commit
git commit -m "test(auth): fix edge case"

# Then push
git push origin feature/branch-name
```

## 📚 Resources

- [Elite Standards Documentation](./docs/ELITE_STANDARDS_IMPLEMENTATION.md)
- [Copilot Instructions](.github/copilot-instructions.md)
- [PR Template](.github/pull_request_template.md)
- [Git Hooks](.husky/)

## ❓ FAQ

**Q: My commit doesn't follow the format, can I force push?**
A: No. Use `git commit --amend` to fix message.

**Q: I want to commit code I know will fail tests?**
A: The hook won't allow it. Fix tests first.

**Q: Can I disable type checking?**
A: No. Use `# type: ignore` with justification if unavoidable.

**Q: I forgot to run checks before commit?**
A: Hooks run automatically. Fix issues and try again.

**Q: How do I update these standards?**
A: Edit `.github/copilot-instructions.md` and submit PR.

---

**Status**: ✅ Live and Enforced
**Support**: See `.github/copilot-instructions.md`
