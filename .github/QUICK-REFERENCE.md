# FAANG-Elite Quick Reference Guide

**Version**: 3.0.0
**For**: Top 0.01% Master Developers
**Updated**: January 14, 2026

---

## 🚀 Quick Start (5 Minutes)

```bash
# 1. Setup development environment
bash scripts/setup-faang.sh
source venv/bin/activate

# 2. Install pre-commit hooks (mandatory)
pre-commit install

# 3. Run all quality checks
pytest tests/ -v --cov=ollama --cov-fail-under=95
mypy ollama/ --strict
ruff check ollama/
pip-audit

# 4. Create feature branch
git checkout -b feature/your-feature-name

# 5. Start coding (see below)
```

---

## 📋 Development Checklist

### Before Every Commit

- [ ] All tests passing: `pytest tests/`
- [ ] Type checking passes: `mypy ollama/ --strict`
- [ ] Linting passes: `ruff check ollama/`
- [ ] Security audit clean: `pip-audit`
- [ ] Coverage ≥95%: `pytest --cov=ollama`
- [ ] Code formatted: `black ollama/ tests/`
- [ ] Docstrings complete
- [ ] No merge conflicts
- [ ] Commit message formatted: `type(scope): message`
- [ ] Commit signed: `git commit -S`

### Before Every Push

- [ ] All commits atomic (single logical unit)
- [ ] All commits pass checks
- [ ] Branch naming correct: `feature/`, `bugfix/`, `refactor/`
- [ ] No credentials or secrets committed
- [ ] PR description complete
- [ ] No formatting changes mixed with logic

---

## 📂 Folder Structure Reference

```
Source Code             →  Test Mirror
────────────────────────────────────────
ollama/services/        →  tests/unit/services/
ollama/repositories/    →  tests/unit/repositories/
ollama/api/routes/      →  tests/integration/api/
```

**GOLDEN RULES**:

- ✅ One class per file
- ✅ snake_case for files, PascalCase for classes
- ✅ Module docstrings on every file
- ✅ Type hints on all functions
- ✅ Maximum 600 lines per file
- ✅ Maximum cognitive complexity 5

---

## 🔤 Naming Conventions

### Commit Types

```
feat(scope):     New feature
fix(scope):      Bug fix
refactor(scope): Code reorganization
perf(scope):     Performance improvement
test(scope):     Test additions
docs(scope):     Documentation
infra(scope):    CI/CD, Docker, deployment
security(scope): Security-related changes
```

### Scopes

```
api, auth, models, inference, cache, db, config,
docker, k8s, monitoring, testing, types, docs, security
```

### Example Commits

```
feat(api): add conversation history endpoint
fix(auth): resolve token expiration race condition
refactor(services): split inference into modules
perf(cache): optimize model loading 40%
test(inference): add edge case tests
docs(api): document rate limiting
security(auth): add rate limiting
```

---

## 🧪 Testing Standards

### Coverage Targets

```
Critical paths:    100% coverage (non-negotiable)
Overall code:      ≥95% coverage
New features:      ≥95% coverage
Performance tests: ≥50% of total tests
Security tests:    ≥30% of total tests
```

### Test File Template

```python
# tests/unit/services/test_my_service.py
"""Tests for MyService.

Mirrors: ollama/services/my_service.py
"""

import pytest
from unittest.mock import Mock
from ollama.services.my_service import MyService

class TestMyService:
    """Test suite for MyService."""

    @pytest.fixture
    def service(self) -> MyService:
        """Create service for tests."""
        return MyService()

    def test_happy_path(self, service: MyService) -> None:
        """Main functionality works."""
        result = service.do_something()
        assert result is not None

    def test_error_case(self, service: MyService) -> None:
        """Error handling works."""
        with pytest.raises(ValueError):
            service.do_something_invalid()
```

---

## 🔐 Type Safety

### Type Hints (MANDATORY)

```python
# ❌ WRONG
def process(data):
    return data + 1

# ✅ CORRECT
def process(data: int) -> int:
    """Process integer data."""
    return data + 1

# Type narrowing
def handle_optional(value: Optional[str]) -> int:
    """Handle potentially None value."""
    if value is None:
        return 0
    return len(value)  # Type narrowed to str
```

### Common Type Patterns

```python
from typing import Optional, List, Dict, Union, Callable, Any
from dataclasses import dataclass

# Optional (can be None)
value: Optional[str] = None

# List of items
items: List[User] = []

# Dictionary
mapping: Dict[str, int] = {}

# Union (multiple types)
result: Union[str, int] = "value"

# Callable function
handler: Callable[[str], bool] = validate

# Generic dataclass
@dataclass
class Response:
    success: bool
    data: Optional[Dict[str, Any]] = None
```

---

## 💾 Git Workflow

### Daily Workflow

```bash
# 1. Start feature branch
git checkout develop
git pull origin develop
git checkout -b feature/new-feature

# 2. Make atomic commits (every 30-60 minutes)
git add .
# Run ALL checks
pytest tests/ --cov
mypy ollama/ --strict
ruff check ollama/
# If all pass:
git commit -S -m "feat(scope): description"
git push origin feature/new-feature

# 3. After 2+ commits, create PR
# Add comprehensive description
# Request review

# 4. Address review feedback
git add .
git commit -S -m "refactor(scope): address review feedback"
git push origin feature/new-feature

# 5. Merge after approval
# All checks passing
# At least one approval
# No conflicts
```

### Force-Push Policy

- ❌ NEVER on main or develop
- ⚠️ Only on personal feature branches
- ⚠️ Only after explicit approval
- ✅ Alternative: Use rebase locally, then force push (if approved)

---

## 📝 Code Quality Thresholds

| Metric               | Threshold  | Action             |
| -------------------- | ---------- | ------------------ |
| Type Coverage        | 100%       | Fail if not 100%   |
| Test Coverage        | ≥95%       | Fail if lower      |
| Cognitive Complexity | <5         | Refactor if higher |
| Function Length      | <100 lines | Review if longer   |
| File Size            | <600 lines | Split if larger    |
| Test Execution       | <30s       | Optimize if slower |
| Linting Errors       | 0          | Auto-fix with ruff |
| Security Issues      | 0          | Fix immediately    |

---

## 🛠️ Essential Commands

### Development

```bash
# Format code
black ollama/ tests/

# Check types
mypy ollama/ --strict

# Lint code
ruff check ollama/ --fix

# Run tests
pytest tests/ -v
pytest tests/ --cov=ollama
pytest tests/unit/ -x  # Stop on first failure

# Run specific test
pytest tests/unit/test_auth.py::TestAuth::test_login -v

# Security audit
pip-audit
bandit -r ollama/ -ll

# Start dev server
uvicorn ollama.main:app --reload --host 0.0.0.0 --port 8000

# Docker
docker-compose up -d
docker-compose down
docker-compose logs -f api
```

### Git

```bash
# Commit (atomic, signed)
git commit -S -m "type(scope): message"

# Push frequently
git push origin feature-branch

# Rebase on main (before PR)
git fetch origin main
git rebase origin/main
git push -f origin feature-branch  # Only if approved!

# View history
git log --oneline --graph
git show --stat <commit>

# Stash changes (temporary)
git stash
git stash pop
```

### Debugging

```bash
# Add breakpoint
breakpoint()  # Or: import ipdb; ipdb.set_trace()

# Check environment
python -c "import sys; print(sys.executable)"

# Python REPL
python -i -c "from ollama.services import ModelService; ms = ModelService()"

# Profile code
python -m cProfile -o profile.out main.py
snakeviz profile.out
```

---

## 📚 Documentation Links

| Document                                                               | Purpose                 |
| ---------------------------------------------------------------------- | ----------------------- |
| [FAANG-ELITE-STANDARDS.md](.github/FAANG-ELITE-STANDARDS.md)           | Top 0.01% dev standards |
| [FOLDER-STRUCTURE-STANDARDS.md](.github/FOLDER-STRUCTURE-STANDARDS.md) | Project structure guide |
| [copilot-instructions.md](.github/copilot-instructions.md)             | Copilot behavior rules  |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md)                                | System design           |
| [API_DESIGN.md](docs/API_DESIGN.md)                                    | API conventions         |

---

## ✅ Pre-Commit Checklist (Template)

Copy this into your commit message:

```
Type: feat|fix|refactor|perf|test|docs|infra|security
Scope: api|auth|models|inference|...
Coverage: ≥95% ✓
Types: 100% strict ✓
Linting: Clean ✓
Tests: All passing ✓
Security: Audited ✓
Docs: Updated ✓

TICKET #123
Fixes: ...
Relates to: ...
Breaking: No

Testing:
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] No performance regression
- [ ] Tested locally with REAL_IP
```

---

## 🆘 Troubleshooting

### Tests Failing

```bash
# Run with verbose output
pytest tests/ -v -s

# Run specific test
pytest tests/unit/test_auth.py::TestAuth -vv

# Show print statements
pytest -s tests/unit/

# Update snapshots (if using pytest-snapshot)
pytest --snapshot-update
```

### Type Errors

```bash
# Check specific file
mypy ollama/services/auth.py --strict

# Show all errors
mypy ollama/ --strict --show-error-context

# Ignore specific error (last resort)
# Add comment: x = value  # type: ignore[error-code]
```

### Import Errors

```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Verify module structure
python -m py_compile ollama/

# Check __init__.py files
find ollama -name "__init__.py" | head -20
```

### Git Issues

```bash
# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1

# Unstage file
git reset HEAD <file>

# Fix commit message
git commit --amend

# Clean up branches
git branch -d feature-complete
git branch -D feature-abandoned  # Force delete
```

---

## 🎯 Performance Baselines

Expected performance metrics:

```
Metric                  | Target      | Alert Threshold
─────────────────────────────────────────────────────
Unit test execution     | <1ms each   | >100ms
Full test suite         | <30s        | >35s (register regression)
API response (p99)      | <500ms      | >550ms (register regression)
Startup time            | <10s        | >12s (register regression)
Memory footprint        | <2GB        | >2.2GB (register regression)
Model inference (p99)   | Per-model   | ±5% regression (register)
```

---

## 📞 Getting Help

**Questions?** Check:

1. [FAANG-ELITE-STANDARDS.md](.github/FAANG-ELITE-STANDARDS.md) - Most comprehensive
2. [FOLDER-STRUCTURE-STANDARDS.md](.github/FOLDER-STRUCTURE-STANDARDS.md) - Structure questions
3. Code examples in `tests/` directory
4. Review recent PRs for patterns

**Issues?**

1. Create a GitHub issue with full context
2. Include reproduction steps
3. Include relevant error messages
4. Tag with `question`, `bug`, or `enhancement`

---

## 🏆 Top 0.01% Habits

### DO

✅ Commit frequently (every 30-60 minutes)
✅ Write atomic, focused commits
✅ Test all code paths
✅ Use type hints everywhere
✅ Document public APIs
✅ Review your own code first
✅ Keep functions small (<50 lines)
✅ Extract helper functions
✅ Use meaningful variable names
✅ Follow folder structure strictly

### DON'T

❌ Work locally >4 hours without pushing
❌ Mix refactoring with features
❌ Commit unformatted code
❌ Use bare `except` clauses
❌ Use `Any` type without justification
❌ Commit credentials or secrets
❌ Use force push on shared branches
❌ Skip tests to "save time"
❌ Leave TODOs without tickets
❌ Ignore linting warnings

---

**Remember**: Every line of code you write reflects the team's standards.
**Quality is not negotiable at top 0.01% level.**

Last updated: January 14, 2026
