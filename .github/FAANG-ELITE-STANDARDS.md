# FAANG-Elite Master Development Standards (Top 0.01%)

**Version**: 3.0.0-FAANG
**Effective**: January 14, 2026
**Standard Level**: Top 0.01% Master Developer (Meta/Google/Amazon/Apple/Netflix level)

**Platform**: [https://elevatediq.ai/ollama](https://elevatediq.ai/ollama)

---

## Executive Standard

This document defines **non-negotiable** development standards at the **top 0.01%** tier, equivalent to senior engineers at FAANG companies. Every line of code, every commit, every folder structure decision reflects this elite commitment.

### The Zero-Tolerance Principle

✅ **ACCEPTABLE**: Production-ready code with perfect hygiene
❌ **NOT ACCEPTABLE**: Everything else

---

## TIER 1: Code Quality Absolutism

### 1.1 Type Safety: 100% Strict Mode

```python
# MANDATORY: All Python code uses Pylance strict mode
# Command: mypy ollama/ --strict --warn-unused-ignores

# ❌ WRONG - Any of these are auto-rejected
def process(data):                           # Missing type hint
    return data + 1                          # Type ambiguity

x = some_function()                          # No return type annotation
result = unknown_dict["key"]                 # No type guard

# ✅ CORRECT - All types explicit
from typing import Optional, Union, TypeVar, Generic
from dataclasses import dataclass

T = TypeVar('T', bound='Comparable')

@dataclass
class ProcessResult(Generic[T]):
    success: bool
    data: Optional[T]
    error: Optional[str]

def process(data: int) -> ProcessResult[int]:
    """Process integer data."""
    if data < 0:
        return ProcessResult(False, None, "Negative input")
    return ProcessResult(True, data + 1, None)

# Type narrowing (required for strict mode)
def handle_optional(value: Optional[str]) -> int:
    if value is None:
        return 0
    return len(value)  # Type narrowed to str
```

### 1.2 Cognitive Complexity: Maximum 5 (FAANG Standard)

- Each function: Max complexity 5 (measured with `radon cc --min B`)
- Average complexity: < 3
- No nested loops (max 1 level deep)
- No nested conditionals (max 2 levels deep)

```python
# ❌ WRONG - Complexity 8 (auto-rejected)
def process_users(users, filters, config):
    result = []
    for user in users:
        if user.active:
            if filters:
                for f in filters:
                    if f(user):
                        if config.validate:
                            if validate_user(user):
                                result.append(user)
    return result

# ✅ CORRECT - Complexity 2 (extract functions)
def should_include_user(user: User, filters: List[Callable]) -> bool:
    """Filter predicate extraction."""
    return user.active and all(f(user) for f in filters)

def validate_and_include(user: User, config: Config) -> bool:
    """Validation extraction."""
    return validate_user(user) if config.validate else True

def process_users(
    users: List[User],
    filters: List[Callable],
    config: Config
) -> List[User]:
    """Clean orchestration function."""
    return [
        u for u in users
        if should_include_user(u, filters) and validate_and_include(u, config)
    ]
```

### 1.3 Function Purity: 85% Pure Functions

- Pure functions (no side effects): ≥85% of codebase
- Impure functions: Clearly named with verb prefix (`send_`, `persist_`, `mutate_`)
- Immutability first: Use `dataclasses.replace()`, `copy.deepcopy()`
- No module-level state mutation

### 1.4 Test Coverage Mandate

- **Critical paths**: 100% coverage (enforce with coverage minimum)
- **Overall**: ≥95% coverage (top 0.01% standard)
- **Performance tests**: ≥50% of test count
- **Security tests**: ≥30% of test count
- **Mutation testing**: ≥80% mutation score

```python
# pytest configuration (mandatory)
[tool.pytest.ini_options]
testpaths = ["tests"]
minversion = "7.0"
addopts = [
    "-v",
    "--strict-markers",
    "--strict-config",
    "--tb=short",
    "--cov=ollama",
    "--cov=config",
    "--cov-fail-under=95",  # MANDATORY: 95% minimum
    "--cov-report=term-missing:skip-covered",
    "--cov-report=html:htmlcov",
    "--cov-report=xml",
    "--maxfail=1",           # Stop on first failure (fast feedback)
]
```

### 1.5 Error Handling: Explicit Errors Only

```python
# ❌ WRONG - Any bare except or generic Exception
try:
    operation()
except Exception:
    pass

# ✅ CORRECT - Specific exception hierarchy
class OllamaError(Exception):
    """Base exception for all Ollama operations."""
    pass

class ModelError(OllamaError):
    """Model-specific errors."""
    pass

class InferenceError(ModelError):
    """Inference operation errors."""
    pass

class ModelNotFoundError(ModelError):
    """Raised when model doesn't exist."""
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        super().__init__(f"Model {model_name} not found")

# Usage with context
try:
    model = load_model(model_id)
except ModelNotFoundError as e:
    logger.error("model_load_failed", model=e.model_name, error=str(e))
    raise APIException(
        code="MODEL_NOT_FOUND",
        message=f"Model {e.model_name} not available",
        status_code=404
    )
```

---

## TIER 2: Folder Structure Absolutism

### 2.1 The Directory Hierarchy (Strict Enforcement)

```
ollama/                                    # Root package
├── __init__.py                            # Package marker (empty or minimal)
├── main.py                                # FastAPI app instantiation only
├── config/                                # Configuration (constants, env)
│   ├── __init__.py
│   ├── settings.py                        # Environment-based settings
│   ├── constants.py                       # Application constants
│   ├── development.yaml                   # Dev environment config
│   ├── production.yaml                    # Prod environment config
│   └── schemas/                           # Configuration schemas
│       └── __init__.py
│
├── api/                                   # HTTP API layer
│   ├── __init__.py
│   ├── router.py                          # Main router (imports all routes)
│   ├── routes/                            # Route handlers (one file = one resource)
│   │   ├── __init__.py
│   │   ├── health.py                      # GET /health
│   │   ├── models.py                      # Model management endpoints
│   │   ├── inference.py                   # Inference endpoints
│   │   ├── conversation.py                # Conversation endpoints
│   │   └── admin.py                       # Admin endpoints
│   ├── schemas/                           # Pydantic models (request/response)
│   │   ├── __init__.py
│   │   ├── health.py
│   │   ├── models.py
│   │   ├── inference.py
│   │   ├── error.py
│   │   └── common.py                      # Shared schemas
│   ├── dependencies.py                    # Dependency injection (auth, db)
│   └── exceptions.py                      # HTTP exceptions (app-specific)
│
├── services/                              # Business logic layer
│   ├── __init__.py
│   ├── base_service.py                    # BaseService abstract class
│   ├── model_service.py                   # Model loading, caching, lifecycle
│   ├── inference_service.py               # Inference orchestration
│   ├── conversation_service.py            # Conversation management
│   ├── auth_service.py                    # Authentication & API keys
│   ├── cache_service.py                   # Cache management
│   └── monitoring_service.py              # Observability
│
├── repositories/                          # Data access layer
│   ├── __init__.py
│   ├── base_repository.py                 # BaseRepository abstract class
│   ├── user_repository.py
│   ├── api_key_repository.py
│   ├── conversation_repository.py
│   └── model_cache_repository.py
│
├── models/                                # SQLAlchemy ORM models
│   ├── __init__.py
│   ├── base.py                            # Base model with timestamps
│   ├── user.py                            # User model
│   ├── api_key.py                         # API key model
│   ├── conversation.py                    # Conversation model
│   └── message.py                         # Message model
│
├── middleware/                            # Request/response processing
│   ├── __init__.py
│   ├── auth_middleware.py
│   ├── logging_middleware.py
│   ├── correlation_id_middleware.py
│   ├── rate_limit_middleware.py
│   └── error_handler_middleware.py
│
├── exceptions.py                          # Custom exception hierarchy
├── types.py                               # Type definitions & aliases
└── monitoring/                            # Observability
    ├── __init__.py
    ├── logger.py                          # Structured logging setup
    ├── metrics.py                         # Prometheus metrics
    ├── tracing.py                         # Jaeger tracing setup
    └── health_checks.py                   # Health check definitions

tests/                                     # Test mirror (exact structure match)
├── __init__.py
├── conftest.py                            # Pytest fixtures (database, cache, auth)
├── fixtures/                              # Reusable fixtures
│   ├── __init__.py
│   ├── models.py                          # Mock models for testing
│   ├── db.py                              # Database fixtures
│   └── auth.py                            # Auth fixtures
├── unit/                                  # Unit tests (fast, isolated)
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_exceptions.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── test_model_service.py          # Mirrors ollama/services/model_service.py
│   │   ├── test_inference_service.py
│   │   └── test_cache_service.py
│   └── repositories/
│       ├── __init__.py
│       ├── test_user_repository.py
│       └── test_api_key_repository.py
├── integration/                           # Integration tests (with services)
│   ├── __init__.py
│   ├── test_api_models_endpoint.py
│   ├── test_inference_endpoint.py
│   └── test_database_transactions.py
├── e2e/                                   # End-to-end tests (full stack)
│   ├── __init__.py
│   └── test_full_workflow.py
├── performance/                           # Performance benchmarks
│   ├── __init__.py
│   ├── test_inference_latency.py
│   └── test_api_throughput.py
└── security/                              # Security tests
    ├── __init__.py
    ├── test_auth_bypass.py
    └── test_rate_limiting.py

config/                                    # Configuration files (NOT CODE)
├── development.yaml
├── production.yaml
└── schemas/

docker/                                    # Container definitions
├── Dockerfile                             # Multi-stage production image
├── Dockerfile.dev                         # Development image
├── postgres/
├── redis/
└── nginx/

k8s/                                       # Kubernetes manifests
├── base/
├── overlays/
└── helm/

monitoring/                                # Observability config
├── prometheus.yml
├── alerts.yml
└── grafana/

scripts/                                   # Automation (executable only)
├── setup.sh                               # Initial setup
├── health_check.sh                        # Health verification
└── migrate.sh                             # Database migrations

alembic/                                   # Database migrations
├── env.py
├── script.py.mako
└── versions/

docs/                                      # Documentation (Markdown)
├── README.md
├── ARCHITECTURE.md
├── API_DESIGN.md
├── DEPLOYMENT.md
├── FAANG_STANDARDS.md
└── TROUBLESHOOTING.md

.github/                                   # GitHub configuration
├── copilot-instructions.md
├── FAANG-ELITE-STANDARDS.md
└── workflows/                             # CI/CD
    ├── test.yml
    ├── lint.yml
    └── security.yml

.vscode/                                   # VS Code configuration
├── settings.json                          # Strict enforcer settings
├── extensions.json                        # Required extensions
├── launch.json                            # Debug configurations
└── tasks.json                             # Automation tasks
```

### 2.2 File Organization Rules (Strict)

**RULE 1: One Class Per File** (except enums, constants)

```python
# ❌ WRONG - Multiple classes in one file
# services/models.py
class ModelLoader: ...
class ModelCache: ...
class ModelMetrics: ...

# ✅ CORRECT - Each class in separate file
# services/model_loader.py
class ModelLoader: ...

# services/model_cache.py
class ModelCache: ...

# services/model_metrics.py
class ModelMetrics: ...
```

**RULE 2: Logical Module Grouping**

- Group related functionality in packages
- Each package has `__init__.py` with clear exports
- No circular imports (enforce with `import-graph` tool)

**RULE 3: Naming Consistency**

- Module files: `snake_case.py`
- Classes: `PascalCase`
- Functions: `snake_case`
- Constants: `SCREAMING_SNAKE_CASE`
- Private (internal): `_leading_underscore`
- Protected (internal to package): `__double_leading_underscore`

### 2.3 Module Header Requirements

**EVERY Python file MUST have:**

```python
"""Module-level docstring (one sentence summary).

Detailed description of module purpose, responsibilities, and how it
fits into the system architecture. Include usage examples.

Example:
    >>> from ollama.services.auth import TokenManager
    >>> manager = TokenManager()
    >>> token = manager.create_token(user_id="123")
    >>> print(token)
    eyJhbGc...

Note:
    This module is core to authentication. Modifications require
    security review.
"""

from __future__ import annotations

import logging
from typing import Optional, Any
from pathlib import Path
from abc import ABC, abstractmethod

from fastapi import HTTPException
import structlog

from ollama.config import settings
from ollama.exceptions import AuthError

log = structlog.get_logger(__name__)
```

---

## TIER 3: Git Hygiene - Master Level

### 3.1 Commit Excellence

**Commit Message Format (STRICT)**

```
type(scope): imperative description [max 50 chars]

Body (wrapped at 72 chars): Explain WHAT and WHY, not HOW.
- Use bullet points for multiple reasons
- Reference GitHub issues: Fixes #123
- Include impact metrics if applicable

Fixes #123
Relates to #456
Breaking: This changes the API contract
Performance: 40% latency improvement
```

**Type Classification (Mandatory)**:

- `feat`: Feature addition (bumps minor version)
- `fix`: Bug fix (bumps patch version)
- `refactor`: Code reorganization (no behavior change)
- `perf`: Performance improvement
- `test`: Test additions/modifications
- `docs`: Documentation updates
- `infra`: Infrastructure/CI/CD/Docker
- `security`: Security-focused changes
- `chore`: Maintenance/dependencies
- `revert`: Revert previous commit

**Scope Classification (Mandatory)**:

- `api`: HTTP API layer
- `auth`: Authentication/authorization
- `models`: ML model management
- `inference`: Inference operations
- `cache`: Redis/caching layer
- `db`: Database/repositories
- `config`: Configuration management
- `docker`: Docker/containerization
- `k8s`: Kubernetes
- `monitoring`: Observability/metrics
- `testing`: Test infrastructure
- `types`: Type safety/mypy
- `docs`: Documentation
- `security`: Security-related

### 3.2 Atomic Commits Mandate

**RULE: Each commit must**:

1. ✅ Be reversible independently
2. ✅ Pass all tests (zero test failures)
3. ✅ Pass type checking (mypy --strict)
4. ✅ Pass linting (ruff check)
5. ✅ Pass security audit (pip-audit)
6. ✅ Be signed with GPG (`git commit -S`)
7. ✅ Touch 1-15 files (≤500 lines changed)
8. ✅ Address single logical unit

```bash
# ❌ WRONG - Monolithic commit
git commit -S -m "feat: massive refactor" # 200 files, 50000 lines

# ✅ CORRECT - Atomic commits
git commit -S -m "feat(api): add health check endpoint"     # 3 files
git commit -S -m "test(api): add health check tests"       # 2 files
git commit -S -m "docs(api): document health endpoint"     # 1 file
```

### 3.3 Push Frequency (Absolute)

- **Minimum**: Every 2 hours of development
- **Maximum**: Never hold >5 commits locally
- **Workflow**:

  ```bash
  # Every 30-60 minutes
  git add .                    # Stage changes
  pytest tests/ -v --cov      # Run all checks (MANDATORY)
  mypy ollama/ --strict       # Type check
  ruff check ollama/          # Lint
  git commit -S -m "type(scope): message"  # Atomic commit
  git push origin feature-branch           # Push immediately
  ```

### 3.4 Branch Naming (Strict Enforcement)

**Format**: `{type}/{lowercase-kebab-case}`

```
feature/add-conversation-history
bugfix/fix-token-refresh-race
refactor/simplify-model-loader
security/add-rate-limiting
infra/add-kubernetes-manifests
perf/optimize-inference-pipeline
docs/update-deployment-guide
```

### 3.5 Pre-commit Hooks (Mandatory)

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-merge-conflict
      - id: no-commit-to-branch
        args: [--branch, main, --branch, develop]

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.14
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: [--strict, --warn-unused-ignores]

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: [-c, .bandit]

  - repo: https://github.com/Lucas-C/pre-commit-hooks
    rev: v1.5.4
    hooks:
      - id: forbid-crlf
      - id: forbid-tabs
```

---

## TIER 4: Code Documentation - Absolute Standard

### 4.1 Docstring Format (Google Style - Mandatory)

```python
def generate_response(
    prompt: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> InferenceResponse:
    """Generate text completion for given prompt.

    Orchestrates model loading, inference, and response formatting.
    Implements exponential backoff on transient failures.

    Args:
        prompt: User input text (max 50000 characters).
        model: Model identifier (e.g., 'llama2:7b').
        temperature: Sampling temperature in range [0.0, 2.0].
            Lower values → deterministic output.
            Higher values → creative output.
            Default: 0.7 (balanced).
        max_tokens: Maximum tokens in response. If None, uses model default.

    Returns:
        InferenceResponse with generated text, tokens used, and metadata.

    Raises:
        ModelNotFoundError: If model doesn't exist in registry.
        InferenceTimeoutError: If generation exceeds 5 minute timeout.
        ValidationError: If prompt exceeds max length.

    Example:
        >>> response = await generate_response(
        ...     prompt="What is machine learning?",
        ...     model="llama2:7b",
        ...     temperature=0.5
        ... )
        >>> print(response.text[:100])
        "Machine learning is a subset of artificial intelligence..."

    Performance:
        - Cold start (model not cached): ~2000ms
        - Warm start (model cached): ~500ms
        - Per token: ~50ms average
        - Batch processing: 4x throughput improvement

    Note:
        This function uses connection pooling from Redis cache.
        Model is locked during inference to prevent concurrent loads.
    """
```

### 4.2 Inline Comments (Sparse, Purposeful)

```python
# ❌ WRONG - Over-commenting obvious code
def add_numbers(a: int, b: int) -> int:
    # Add a and b
    result = a + b  # Store in result
    return result   # Return the result

# ✅ CORRECT - Comments explain WHY, not WHAT
def calculate_batch_size(model_memory_mb: int, available_memory_mb: int) -> int:
    """Calculate optimal batch size for given memory constraints."""
    # Use 70% of available memory to leave headroom for OS and caches
    usable_memory = int(available_memory_mb * 0.7)

    # Ensure minimum batch size of 1 to avoid division errors
    batch_size = max(1, usable_memory // model_memory_mb)

    return batch_size
```

### 4.3 Type Hints in All Return Statements

```python
# ❌ WRONG - Missing type info
def fetch_user(user_id):
    if not user_id:
        return None
    user_data = db.query(User).get(user_id)
    return user_data

# ✅ CORRECT - Complete type annotation
def fetch_user(user_id: UUID) -> Optional[User]:
    """Fetch user by ID or return None if not found."""
    if not user_id:
        return None
    return db.query(User).filter(User.id == user_id).first()
```

---

## TIER 5: Testing Excellence

### 5.1 Test Pyramid (FAANG-Level)

```
            /\
           /  \       E2E Tests (5%)
          /----\      - Full stack workflows
         /      \     - Real database
        /        \
       /          \
      /____________\   Integration Tests (25%)
     /              \  - Service interactions
    /                \ - Mock external APIs
   /                  \
  /____________________\ Unit Tests (70%)
                       - Fast, isolated
                       - Mock everything
                       - <100ms per test
```

### 5.2 Test File Structure

```python
# tests/unit/services/test_model_service.py
"""Tests for ModelService.

Mirrors: ollama/services/model_service.py
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4

from ollama.services.model_service import ModelService
from ollama.exceptions import ModelNotFoundError
from tests.fixtures import get_mock_model


class TestModelService:
    """Test suite for ModelService class."""

    @pytest.fixture
    def service(self) -> ModelService:
        """Create service instance for testing."""
        return ModelService(
            model_registry=Mock(),
            cache=Mock()
        )

    def test_load_model_success(self, service: ModelService) -> None:
        """Loading existing model returns model instance."""
        model_id = "llama2:7b"
        mock_model = get_mock_model(model_id)

        service.model_registry.get.return_value = mock_model

        result = service.load_model(model_id)

        assert result.id == model_id
        service.model_registry.get.assert_called_once_with(model_id)

    def test_load_model_not_found(self, service: ModelService) -> None:
        """Loading non-existent model raises ModelNotFoundError."""
        model_id = "nonexistent:1b"
        service.model_registry.get.side_effect = KeyError(model_id)

        with pytest.raises(ModelNotFoundError):
            service.load_model(model_id)

    @pytest.mark.asyncio
    async def test_generate_response_success(self, service: ModelService) -> None:
        """Generating response returns valid InferenceResponse."""
        prompt = "What is AI?"
        model = get_mock_model("llama2:7b")
        service.model_registry.get.return_value = model

        response = await service.generate(
            model_id=model.id,
            prompt=prompt
        )

        assert response.success
        assert response.text is not None
        assert response.tokens_used > 0

    @pytest.mark.asyncio
    async def test_generate_response_timeout(
        self,
        service: ModelService
    ) -> None:
        """Generating response with timeout raises InferenceTimeoutError."""
        model = get_mock_model("llama2:7b")
        service.model_registry.get.return_value = model
        model.generate = AsyncMock(side_effect=asyncio.TimeoutError())

        with pytest.raises(InferenceTimeoutError):
            await service.generate(
                model_id=model.id,
                prompt="Test"
            )
```

### 5.3 Test Markers (Strict Organization)

```python
# pytest.ini configuration
[pytest]
markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests (with services)
    e2e: End-to-end tests (full stack)
    slow: Slow tests (>1 second)
    performance: Performance benchmarks
    security: Security tests
    db: Tests requiring database
    cache: Tests requiring Redis cache
    requires_model: Tests requiring ML model
```

---

## TIER 6: Security & Secrets Management

### 6.1 Credentials: Never Committed

```bash
# ✅ CORRECT - Use environment variables
export POSTGRES_PASSWORD="$(openssl rand -base64 32)"
export REDIS_PASSWORD="$(openssl rand -base64 32)"
export API_SECRET_KEY="$(openssl rand -base64 64)"

# ✅ CORRECT - Use .env template for developers
# .env.example
POSTGRES_PASSWORD=change-me-in-development
REDIS_PASSWORD=change-me-in-development
API_SECRET_KEY=change-me-in-development

# ❌ WRONG - Never commit
# .env (gitignored)
POSTGRES_PASSWORD=super-secret-dev-password
REDIS_PASSWORD=another-secret
API_SECRET_KEY=development-key-123
```

### 6.2 Secret Scanning Enforcement

```yaml
# .github/workflows/secret-scan.yml
name: Secret Scanning

on: [push, pull_request]

jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Run Truffles Secret Scanner
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
          base: ${{ github.event.repository.default_branch }}
          head: HEAD
```

---

## TIER 7: CI/CD Pipeline Enforcement

### 7.1 Required Checks (Zero Tolerance)

All of these MUST pass before merge:

```yaml
# .github/workflows/checks.yml
name: All Checks

on:
  push:
    branches: ["main", "develop"]
  pull_request:
    branches: ["main", "develop"]

jobs:
  type-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
      - run: pip install -e ".[dev]"
      - run: mypy ollama/ --strict # MANDATORY

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
      - run: pip install -e ".[dev]"
      - run: ruff check ollama/ tests/ # MANDATORY

  tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15-alpine
        env:
          POSTGRES_PASSWORD: testpass
      redis:
        image: redis:7-alpine
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v --cov=ollama --cov-fail-under=95
      - run: coverage-badge -o coverage.svg

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
      - run: pip install -e ".[dev]"
      - run: pip-audit # MANDATORY
      - run: bandit -r ollama/ -ll # MANDATORY

  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
      - run: pip install -e ".[dev]"
      - run: pytest tests/ --cov=ollama --cov-report=xml
      - uses: codecov/codecov-action@v3
        with:
          fail_ci_if_error: true # MANDATORY
```

---

## TIER 8: Performance Standards (FAANG-Level)

### 8.1 Performance Baselines (Non-Negotiable)

| Metric                  | Baseline       | Tolerance | Enforcement                 |
| ----------------------- | -------------- | --------- | --------------------------- |
| Unit test execution     | <1ms each      | N/A       | Fail if >100ms              |
| Full test suite         | <30s           | ±5%       | Fail if >31.5s              |
| API response time (p99) | <500ms         | ±10%      | Alert if >550ms             |
| Model inference latency | Model-specific | ±5%       | Benchmark on merge          |
| Memory footprint        | <2GB baseline  | ±10%      | Track with regression tests |
| Startup time            | <10s           | ±20%      | Fail if >12s                |
| Cache hit rate          | >85%           | ±5%       | Alert if <80%               |

### 8.2 Benchmarking (Mandatory)

```python
# tests/performance/test_inference_latency.py
import pytest
import statistics
from ollama.services.inference import InferenceService

class TestInferencePerformance:
    """Performance benchmarks for inference operations."""

    @pytest.mark.performance
    async def test_inference_latency_baseline(
        self,
        inference_service: InferenceService
    ) -> None:
        """Inference latency must stay within baseline."""
        latencies: list[float] = []

        for _ in range(100):
            start = time.perf_counter()
            await inference_service.generate(
                model="llama2:7b",
                prompt="Test prompt"
            )
            latencies.append(time.perf_counter() - start)

        p99 = statistics.quantiles(latencies, n=100)[98]

        # MANDATORY: p99 latency within tolerance
        assert p99 < 0.550, f"p99 latency {p99}s exceeds 550ms baseline"

        print(f"p99 latency: {p99*1000:.1f}ms")
        print(f"p50 latency: {statistics.median(latencies)*1000:.1f}ms")
        print(f"std dev: {statistics.stdev(latencies)*1000:.1f}ms")
```

---

## TIER 9: Code Review Standards

### 9.1 Mandatory Review Checklist

Before ANY merge:

- [ ] **Type Safety**: All new code passes `mypy --strict`
- [ ] **Test Coverage**: New code has ≥95% coverage
- [ ] **Performance**: No regression in benchmarks
- [ ] **Security**: No credentials, passwords, or keys exposed
- [ ] **Documentation**: Docstrings and comments are complete
- [ ] **Linting**: All lint errors fixed
- [ ] **Commits**: Atomic, signed, properly formatted
- [ ] **Folder Structure**: Follows exact standards
- [ ] **Dependencies**: No new external dependencies without approval
- [ ] **Breaking Changes**: Documented if present

### 9.2 Code Review Comments (Template-Driven)

```markdown
## Code Review: [PR #123]

### ✅ Approved (with minor comments)

#### Type Safety ✅

- All functions have type hints
- No `Any` types without justification

#### Test Coverage ✅

- New code has 96% coverage
- Edge cases tested

#### Performance ✅

- No regression vs baseline
- Optimized hot paths

#### Security ✅

- No credentials exposed
- Input validation present

#### Documentation ✅

- Docstrings complete
- Examples provided

#### Suggestions for improvement:

1. Consider extracting `_validate_input` to reduce complexity
2. Add performance benchmark for large batch sizes

**Verdict**: APPROVED ✅
```

---

## TIER 10: Developer Environment Setup

### 10.1 VS Code Configuration (Mandatory)

All developers MUST use standardized settings:

```json
{
  "python.linting.mypyEnabled": true,
  "python.linting.mypyArgs": ["--strict", "--warn-unused-ignores"],
  "python.analysis.typeCheckingMode": "strict",
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll": "explicit",
    "source.fixAll.ruff": "explicit"
  },
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.rulers": [100],
    "editor.wordWrapColumn": 100
  }
}
```

### 10.2 Git Configuration (Mandatory)

```bash
# Configure GPG signing (required for all commits)
git config --global user.signingkey <YOUR_GPG_KEY>
git config --global commit.gpgSign true

# Configure pre-commit hooks
pip install pre-commit
pre-commit install

# Enforce push frequency check
git config --global alias.push-check '!git push --force-if-includes'
```

---

## Exception Handling

**NO EXCEPTIONS** to these standards. If you believe an exception is warranted:

1. Create a GitHub issue with full justification
2. Get approval from code review team AND tech lead
3. Document exception in EXCEPTIONS.md with expiration date
4. Schedule removal of exception (max 1 sprint)

---

## Enforcement & Automation

### Automated Checks

- ✅ Type checking: mypy (on every commit)
- ✅ Linting: ruff (on every commit)
- ✅ Testing: pytest (on every push)
- ✅ Security: pip-audit (on every push)
- ✅ Coverage: Must maintain ≥95% (enforced in CI)
- ✅ Performance: Benchmarks run on every merge (alerting if regression)
- ✅ Folder structure: Enforced via pre-commit hooks
- ✅ Git hygiene: Enforced via pre-commit hooks (no force push, commit message format, signing)

### Manual Checks

- Code review with checklist
- Security review for sensitive changes
- Performance review for latency-sensitive code
- Architecture review for structural changes

---

## Success Metrics

Your code meets **Top 0.01% FAANG standards** if:

✅ Type coverage: 100% (mypy --strict passes)
✅ Test coverage: ≥95% (all new code tested)
✅ Cognitive complexity: All functions <5
✅ Performance: Zero regressions
✅ Security: Zero vulnerabilities
✅ Git commits: Atomic, signed, formatted perfectly
✅ Folder structure: Exact match to standards
✅ Documentation: Complete, examples included
✅ Code review: Approved by peer + tech lead
✅ CI/CD: All checks passing

---

**Version**: 3.0.0-FAANG
**Last Updated**: January 14, 2026
**Maintained By**: Elite Engineering Team
**Status**: ACTIVE - NO EXCEPTIONS
