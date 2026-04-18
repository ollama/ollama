# FAANG Elite Standards - Team Onboarding Guide

Welcome to the Ollama project! This guide helps you adopt FAANG-level (Top 0.01% Google/Meta/Amazon/Apple/Netflix equivalent) development standards.

**Time Estimate**: 45-60 minutes for complete setup

---

## Table of Contents

1. [5-Minute Quick Start](#5-minute-quick-start)
2. [Environment Setup](#environment-setup)
3. [Core Standards Overview](#core-standards-overview)
4. [Daily Development Workflow](#daily-development-workflow)
5. [Pre-Commit Enforcement](#pre-commit-enforcement)
6. [Common Questions](#common-questions)
7. [Getting Help](#getting-help)

---

## 5-Minute Quick Start

### Step 1: Clone and Initialize

```bash
# Clone the repository
git clone https://github.com/kushin77/ollama.git
cd ollama

# Run setup script
bash scripts/setup-faang.sh

# Verify all tools work
python -m pytest tests/unit -v
mypy ollama/ --strict
ruff check ollama/
```

### Step 2: Open in VS Code

```bash
# Open project in VS Code
code .

# Accept recommended extensions when prompted
# Extensions: Python, Pylance, Ruff, Black, GitLens
```

### Step 3: First Commit

```bash
# Read the quick reference
cat .github/QUICK-REFERENCE.md

# Make a change
echo '# My first FAANG-standard commit' >> CONTRIBUTING.md

# Commit with standards
git add .
git commit -S -m "docs(contributing): add section header"
git push origin feature/my-feature-branch
```

✅ **You're ready to develop!**

---

## Environment Setup

### Detailed Setup Steps

#### 1. Verify Python Version

```bash
python3 --version
# Should be: Python 3.11+ (3.12+ recommended)

# If not installed, install Python 3.12
sudo apt-get install python3.12 python3.12-venv  # Ubuntu/Debian
brew install python@3.12                          # macOS
```

#### 2. Run Automated Setup

```bash
bash scripts/setup-faang.sh
```

This script:

- ✅ Creates virtual environment
- ✅ Installs dependencies
- ✅ Installs pre-commit hooks
- ✅ Runs initial quality checks
- ✅ Shows setup summary

#### 3. Configure Git for Signing

All commits MUST be signed with GPG.

```bash
# Generate GPG key (if you don't have one)
gpg --full-generate-key
# Answer: (1) RSA, (4096) key size, (0) no expiry, (enter) name/email

# List keys
gpg --list-secret-keys

# Configure git to use GPG
git config --global user.signingkey <KEY_ID>
git config --global commit.gpgsign true

# Verify it works
git commit -S -m "test: gpg signing works" --allow-empty
```

#### 4. VS Code Configuration

The project includes `.vscode/settings-faang.json` with strict enforcement:

```bash
# Copy to active settings
cp .vscode/settings-faang.json .vscode/settings.json

# OR manually merge the settings if you have existing ones
```

Settings include:

- ✅ Type checking: `mypy --strict` on every file save
- ✅ Linting: `ruff` auto-fix on save
- ✅ Formatting: `black` on save
- ✅ Testing: `pytest` integration
- ✅ Git: GPG signing enforcement

---

## Core Standards Overview

### Top 5 FAANG Standards You'll Use Daily

#### 1. Type Safety (100% Required)

Every function parameter and return value must be typed:

```python
# ❌ WRONG: No types
def process_request(request):
    return {"status": "ok"}

# ✅ CORRECT: Full types
from fastapi import Request

async def process_request(request: Request) -> dict[str, str]:
    """Process incoming request and return status."""
    return {"status": "ok"}
```

**Tool**: `mypy ollama/ --strict` (runs automatically on save)

#### 2. Naming Conventions (Strict)

- **Files**: `lowercase_with_underscores.py` (snake_case)
- **Classes**: `PascalCase`
- **Functions**: `snake_case`
- **Constants**: `SCREAMING_SNAKE_CASE`
- **Folders**: `lowercase_underscores` (no hyphens, no spaces)

```python
# ✅ CORRECT file structure
ollama/
├── config/                    # folder: lowercase_underscores
│   ├── settings.py           # file: snake_case
│   └── constants.py          # file: snake_case
├── services/
│   └── inference.py          # One class per file
│       └── class InferenceService:  # class: PascalCase
│           def generate_response(self) -> str:  # method: snake_case
│               MODEL_NAME = "llama3.2"  # constant: SCREAMING_SNAKE_CASE
```

#### 3. One Class Per File (Strict)

Each module should have ONE primary class, with exceptions for helpers:

```python
# ✅ CORRECT: One class per file
# File: services/inference.py
class InferenceService:
    """Manages model inference."""
    pass

# ✅ CORRECT: Multiple related classes allowed (exceptions only)
# File: api/schemas.py (allowed for schemas)
class GenerateRequest(BaseModel):
    pass

class GenerateResponse(BaseModel):
    pass
```

**Validation**: `ruff check ollama/ --select SIM1`

#### 4. Test Coverage (≥95%)

Every new function must have tests in `tests/` mirroring the structure:

```python
# app/services/inference.py
class InferenceService:
    def generate(self, prompt: str) -> str:
        """Generate text response."""
        pass

# tests/unit/services/test_inference.py (EXACT MIRROR)
class TestInferenceService:
    def test_generate_returns_string(self) -> None:
        """Generated response is string type."""
        service = InferenceService()
        result = service.generate("test prompt")
        assert isinstance(result, str)
```

**Validation**: `pytest tests/ --cov=ollama --cov-report=term-missing`

#### 5. Atomic, Signed Commits

Every commit must:

- Be GPG signed (`-S` flag)
- Follow format: `type(scope): description`
- Be reversible without breaking other commits
- Pass all quality checks

```bash
# ✅ CORRECT: Atomic, signed, follows format
git commit -S -m "feat(api): add conversation history endpoint"

# ❌ WRONG: Not signed, multiple unrelated changes
git commit -m "various fixes and improvements"

# ❌ WRONG: Large commits, multiple concerns
git commit -m "refactor everything"
```

---

## Daily Development Workflow

### Morning: Start Development

```bash
# 1. Pull latest changes
git pull origin main

# 2. Create feature branch
git checkout -b feature/my-feature

# 3. Open in VS Code (settings auto-apply)
code .
```

### During Development: Code with Standards

```python
# File: ollama/services/llm.py
"""LLM service for inference.

Handles model loading, token generation, and result formatting.
"""

from typing import ClassVar
from pathlib import Path
import structlog

from ollama.models import LLMModel
from ollama.repositories.model import ModelRepository
from ollama.exceptions import ModelNotFoundError

log = structlog.get_logger(__name__)

class LLMService:
    """Manages LLM inference and response generation."""

    MAX_TOKENS: ClassVar[int] = 2048
    DEFAULT_MODEL: ClassVar[str] = "llama3.2"

    def __init__(self, repo: ModelRepository) -> None:
        """Initialize LLM service with repository."""
        self.repo = repo

    async def generate(
        self,
        prompt: str,
        model_name: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, str]:
        """Generate text response for given prompt.

        Args:
            prompt: Input text to generate response for
            model_name: Model to use (defaults to DEFAULT_MODEL)
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary with 'text' and 'model' keys

        Raises:
            ModelNotFoundError: If model not found
            ValueError: If prompt is empty
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        model = model_name or self.DEFAULT_MODEL
        tokens = max_tokens or self.MAX_TOKENS

        try:
            llm = await self.repo.get_model(model)
        except KeyError as e:
            log.error("model_not_found", model=model)
            raise ModelNotFoundError(f"Model {model} not found") from e

        response = await llm.generate(prompt, tokens)

        return {
            "text": response.text,
            "model": model,
            "tokens_used": response.token_count,
        }
```

### Before Every Commit: Run Checks

VS Code does this automatically, but you can also run manually:

```bash
# Option 1: Run all checks (automated)
# Runs automatically on save in VS Code

# Option 2: Run before commit (manual)
pytest tests/ -v --cov=ollama --cov-report=term-missing
mypy ollama/ --strict
ruff check ollama/
pip-audit

# Option 3: Run via task
cmd/ctrl + shift + p → "Run Task: Run All Checks"
```

### Making Your Commit

```bash
# 1. Stage changes
git add ollama/services/llm.py tests/unit/services/test_llm.py

# 2. Verify changes
git diff --staged

# 3. Commit with GPG signing and proper format
git commit -S -m "feat(services): implement LLM text generation

Add LLMService class to handle model inference with:
- Async text generation with configurable token limits
- Proper error handling for missing models
- Structured logging for observability
- Full type annotations (mypy --strict compatible)

Implements: Issue #234
Coverage: 97% (all code paths tested)
"

# 4. Push immediately
git push origin feature/my-feature
```

### End of Day: Create Pull Request

```bash
# 1. Visit GitHub
# https://github.com/kushin77/ollama

# 2. Create Pull Request
# - Compare: your-branch vs main
# - Title: Same as commit (feat(scope): description)
# - Description: Copy commit body + link issue

# 3. Wait for CI/CD
# Automated checks run:
# - mypy --strict type checking
# - ruff linting
# - pytest tests (≥95% coverage required)
# - pip-audit security scan
# - Black formatting

# 4. Address feedback
git add .
git commit -S -m "refactor(services): improve error handling feedback"
git push origin feature/my-feature

# 5. Merge when approved
# Green checks ✅ = Ready to merge
```

---

## Pre-Commit Enforcement

### What Happens on `git commit`

Pre-commit hooks run automatically:

```
1. ✅ Trailing whitespace removed
2. ✅ File endings fixed
3. ✅ YAML/JSON validated
4. ✅ Large files detected (>500KB blocked)
5. ✅ Private keys detected (blocks if found)
6. ✅ Ruff formatting applied
7. ✅ MyPy type checking (strict mode)
8. ✅ Bandit security scan
9. ✅ Commit message format validated
```

If any check fails:

```bash
# Error example:
# ❌ MyPy found type error at tests/unit/test_auth.py:45
#    Function missing return type annotation

# Fix:
# 1. See the exact error
# 2. Fix in your editor
# 3. `git add <file>`
# 4. `git commit -S -m "..."` (try again)
```

### Skipping Hooks (Not Recommended)

Only in emergencies:

```bash
# Skip ALL hooks (dangerous!)
git commit --no-verify -S -m "emergency fix"

# Better: Fix the issue
# 1. Understand why it failed
# 2. Fix the problem
# 3. Commit normally

# Or: Use bypass with justification (requires approval)
git commit --no-verify -S -m "chore(skip-hooks): reason for bypass"
```

---

## Common Questions

### Q1: "My function is complex - can I skip the type hints?"

**A**: No. FAANG standards mandate 100% type coverage. Break the function into smaller pieces:

```python
# ❌ Complex function (breaks type requirement)
def complex_operation(data):
    # 100+ lines mixing concerns
    return result

# ✅ Broken into typed functions
def validate_input(data: dict[str, str]) -> bool:
    """Validate input data."""
    return True

def process_data(validated: dict[str, str]) -> ProcessResult:
    """Process validated data."""
    return ProcessResult()

async def complex_operation(data: dict[str, str]) -> ProcessResult:
    """Orchestrate validation and processing."""
    if validate_input(data):
        return await process_data(data)
    raise ValueError("Invalid input")
```

### Q2: "Can I commit without GPG signing?"

**A**: No. GPG signing is MANDATORY for all commits. This ensures:

- Commit authenticity (proves you wrote it)
- Non-repudiation (can't deny authorship)
- Compliance (required for enterprise)

Set it up:

```bash
# One-time setup (5 minutes)
gpg --full-generate-key
git config --global user.signingkey <KEY_ID>
git config --global commit.gpgsign true
```

### Q3: "My test coverage is 94% - can I commit?"

**A**: No. Minimum is 95%. Add tests until you reach 95%:

```bash
# Check coverage
pytest tests/ --cov=ollama --cov-report=term-missing

# See which lines need testing
# Coverage shows exact line numbers missing tests

# Add tests for those lines
# Commit when ≥95%
```

### Q4: "Can I put multiple classes in one file?"

**A**: Only in exceptions. Otherwise, NO:

```python
# ❌ WRONG: Two unrelated classes
# File: services/auth.py
class LoginService:
    pass
class TokenValidator:
    pass

# ✅ CORRECT: Split into two files
# File: services/auth.py
class LoginService:
    pass

# File: services/token.py
class TokenValidator:
    pass

# ✅ CORRECT: Multiple related schemas allowed
# File: api/schemas.py (EXCEPTION)
class LoginRequest(BaseModel):
    pass
class LoginResponse(BaseModel):
    pass
```

### Q5: "How do I handle exceptions?"

**A**: Create custom exception hierarchy:

```python
# ✅ CORRECT: Custom exception hierarchy
# File: exceptions.py
class OllamaException(Exception):
    """Base exception for Ollama errors."""
    pass

class ModelNotFoundError(OllamaException):
    """Model not found."""
    pass

class AuthenticationError(OllamaException):
    """Authentication failed."""
    pass

# Usage in your code:
try:
    model = load_model(name)
except KeyError as e:
    raise ModelNotFoundError(f"Model '{name}' not found") from e
```

---

## Getting Help

### Documentation

| Need                       | Resource                                                               |
| -------------------------- | ---------------------------------------------------------------------- |
| **Standards Overview**     | [FAANG-ELITE-STANDARDS.md](.github/FAANG-ELITE-STANDARDS.md)           |
| **Folder Structure**       | [FOLDER-STRUCTURE-STANDARDS.md](.github/FOLDER-STRUCTURE-STANDARDS.md) |
| **Daily Quick Reference**  | [QUICK-REFERENCE.md](.github/QUICK-REFERENCE.md)                       |
| **Implementation Details** | [IMPLEMENTATION-SUMMARY.md](.github/IMPLEMENTATION-SUMMARY.md)         |
| **Navigation Hub**         | [MASTER-INDEX.md](.github/MASTER-INDEX.md)                             |

### Common Issues

```bash
# Issue: Pre-commit hooks failing on mypy
# Solution:
mypy ollama/ --strict
# Fix all errors shown

# Issue: Can't commit - says GPG not configured
# Solution:
gpg --full-generate-key
git config --global user.signingkey <KEY_ID>

# Issue: Tests failing on coverage
# Solution:
pytest tests/ --cov=ollama --cov-report=term-missing
# Add tests for uncovered lines (red lines in output)

# Issue: Can't push - says pre-commit failed
# Solution:
ruff check ollama/ --fix
mypy ollama/ --strict
pytest tests/
# Then retry
```

### Contact

- 🔧 **Tech Issues**: Open GitHub issue
- 💬 **Questions**: Post in #engineering Slack
- 📖 **Documentation**: See MASTER-INDEX.md
- 👥 **Code Review**: Tag @kushin77/ollama-team on PR

---

## Verification Checklist

After setup, verify everything works:

```bash
# ✅ Python version
python3 --version  # Should be 3.11+

# ✅ Virtual environment
source venv/bin/activate

# ✅ Pre-commit hooks installed
pre-commit run --all-files

# ✅ Type checking works
mypy ollama/ --strict

# ✅ Tests pass
pytest tests/ -v

# ✅ Linting passes
ruff check ollama/

# ✅ Security audit passes
pip-audit

# ✅ Git signing configured
git config --global user.signingkey
# Should show a key ID (not empty)
```

When ALL above show ✅, you're ready!

---

## Welcome to FAANG-Level Development! 🚀

You're now part of a team that develops at the Top 0.01% standard, equivalent to Google L5-L6, Meta E5-E6, Amazon SDE-III+.

**Key Mindset**:

- Type safety is not optional - it's mandatory
- Tests prove your code works - aim for >95% coverage
- Commits are forever - make them atomic and signed
- Documentation is code - keep it current
- Standards are non-negotiable - they protect product quality

**Next Steps**:

1. Read [QUICK-REFERENCE.md](.github/QUICK-REFERENCE.md) (5 min)
2. Create your first feature branch
3. Make first commit with standards
4. Create your first PR
5. Ask questions anytime - we're here to help!

**Remember**: These standards aren't bureaucracy - they're guardrails that let you move fast without breaking things. Embrace them. Master them. Excel.

---

**Last Updated**: January 14, 2026
**Standard Version**: FAANG Elite v2.0
**Status**: 🟢 Production Ready
