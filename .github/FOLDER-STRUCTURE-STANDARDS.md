# FAANG-Level Folder Structure & File Organization Standards

**Version**: 3.0.0
**Effective**: January 14, 2026
**Enforcement Level**: Mandatory (Automated)

## Executive Summary

This document defines the **exact, non-negotiable** folder structure for the Ollama project. Deviations require explicit written approval and trigger automated CI/CD rejections.

---

## Root Directory Structure

```
ollama/                                    # Root: Project container
├── README.md                              # Project overview (50-200 lines)
├── LICENSE                                # MIT or Apache 2.0
├── .github/                               # GitHub configuration
│   ├── copilot-instructions.md           # This file
│   ├── FAANG-ELITE-STANDARDS.md          # Elite development standards
│   ├── FOLDER-STRUCTURE.md               # Folder structure guide (this)
│   └── workflows/                        # CI/CD pipelines
│       ├── tests.yml                     # Unit/integration tests
│       ├── lint.yml                      # Type check, linting, formatting
│       ├── security.yml                  # Security audits
│       └── deploy.yml                    # Deployment pipeline
├── .vscode/                               # VS Code configuration
│   ├── settings.json                     # Base settings (version controlled)
│   ├── settings-faang.json               # FAANG-level strict settings
│   ├── extensions.json                   # Recommended extensions
│   ├── launch.json                       # Debug configurations
│   └── tasks.json                        # Automation tasks
├── .pre-commit-config.yaml               # Pre-commit hooks (MANDATORY)
├── pyproject.toml                        # Python project metadata
├── setup.py                              # Package setup
├── requirements/                         # Dependency management
│   ├── base.txt                          # Core dependencies
│   ├── dev.txt                           # Development tools
│   └── prod.txt                          # Production dependencies
│
├── ollama/                               # Main package (source code)
│   ├── __init__.py                       # Package initialization (minimal)
│   ├── main.py                           # FastAPI app instantiation
│   ├── config/                           # Configuration layer
│   ├── api/                              # HTTP API layer
│   ├── services/                         # Business logic layer
│   ├── repositories/                     # Data access layer
│   ├── models/                           # SQLAlchemy ORM models
│   ├── middleware/                       # HTTP middleware
│   ├── exceptions.py                     # Exception hierarchy
│   ├── types.py                          # Type definitions
│   └── monitoring/                       # Observability
│
├── tests/                                # Test suite (mirror structure)
│   ├── __init__.py
│   ├── conftest.py                       # Pytest configuration & fixtures
│   ├── fixtures/                         # Reusable test fixtures
│   ├── unit/                             # Unit tests (fast, isolated)
│   ├── integration/                      # Integration tests
│   ├── e2e/                              # End-to-end tests
│   ├── performance/                      # Performance benchmarks
│   └── security/                         # Security tests
│
├── config/                               # Configuration files (NOT code)
│   ├── development.yaml
│   ├── production.yaml
│   └── schemas/
│
├── docker/                               # Container configuration
│   ├── Dockerfile                        # Production image
│   ├── Dockerfile.dev                    # Development image
│   ├── postgres/
│   ├── redis/
│   └── nginx/
│
├── k8s/                                  # Kubernetes manifests
│   ├── base/
│   ├── overlays/
│   └── helm/
│
├── monitoring/                           # Observability configuration
│   ├── prometheus.yml
│   ├── alerts.yml
│   └── grafana/
│
├── scripts/                              # Automation scripts (executable only)
│   ├── setup.sh
│   ├── health_check.sh
│   └── migrate.sh
│
├── alembic/                              # Database migrations
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
│
├── docs/                                 # Documentation (Markdown)
│   ├── README.md
│   ├── ARCHITECTURE.md
│   ├── API_DESIGN.md
│   ├── DEPLOYMENT.md
│   ├── TROUBLESHOOTING.md
│   └── images/
│
├── htmlcov/                              # Coverage reports (gitignored)
├── .env.example                          # Environment template
├── .gitignore                            # Git ignore rules
└── venv/                                 # Virtual environment (gitignored)
```

---

## TIER 1: Package Structure (ollama/)

### 1.1 Main Package Layout

```
ollama/
├── __init__.py                          # MINIMAL: Only __version__ string
├── main.py                              # FastAPI instantiation ONLY
│
├── config/                              # Configuration (read-only)
│   ├── __init__.py                      # from . import settings
│   ├── settings.py                      # Environment-based settings class
│   ├── constants.py                     # Application constants
│   ├── development.yaml
│   ├── production.yaml
│   └── schemas/
│       └── __init__.py
│
├── api/                                 # HTTP API layer
│   ├── __init__.py                      # from .router import router
│   ├── router.py                        # Main APIRouter (imports all routes)
│   ├── routes/
│   │   ├── __init__.py                  # Empty or: from . import health, models, ...
│   │   ├── health.py                    # Health check endpoint
│   │   ├── models.py                    # Model management endpoints
│   │   ├── inference.py                 # Inference endpoints
│   │   ├── conversation.py              # Conversation endpoints
│   │   └── admin.py                     # Admin endpoints
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── health.py                    # Health response schemas
│   │   ├── models.py                    # Model schemas
│   │   ├── inference.py                 # Inference schemas
│   │   ├── error.py                     # Error response schemas
│   │   └── common.py                    # Shared schemas
│   ├── dependencies.py                  # Dependency injection
│   └── exceptions.py                    # HTTP-specific exceptions
│
├── services/                            # Business logic
│   ├── __init__.py                      # Empty (no direct imports)
│   ├── base_service.py                  # Abstract BaseService class
│   ├── model_service.py                 # Model loading & lifecycle
│   ├── inference_service.py             # Inference orchestration
│   ├── conversation_service.py          # Conversation management
│   ├── auth_service.py                  # Authentication & API keys
│   ├── cache_service.py                 # Cache management
│   └── monitoring_service.py            # Observability
│
├── repositories/                        # Data access layer
│   ├── __init__.py                      # Empty (no direct imports)
│   ├── base_repository.py               # Abstract BaseRepository
│   ├── user_repository.py               # User data access
│   ├── api_key_repository.py            # API key data access
│   ├── conversation_repository.py       # Conversation data access
│   └── model_cache_repository.py        # Model cache data access
│
├── models/                              # SQLAlchemy ORM models
│   ├── __init__.py                      # from . import User, APIKey, ...
│   ├── base.py                          # Base model with timestamps
│   ├── user.py                          # User model
│   ├── api_key.py                       # API key model
│   ├── conversation.py                  # Conversation model
│   └── message.py                       # Message model
│
├── middleware/                          # HTTP middleware
│   ├── __init__.py
│   ├── auth_middleware.py               # API key authentication
│   ├── logging_middleware.py            # Request/response logging
│   ├── correlation_id_middleware.py     # Correlation ID tracking
│   ├── rate_limit_middleware.py         # Rate limiting
│   └── error_handler_middleware.py      # Error handling
│
├── exceptions.py                        # Custom exception hierarchy
├── types.py                             # Type aliases & definitions
└── monitoring/                          # Observability
    ├── __init__.py
    ├── logger.py                        # Structured logging
    ├── metrics.py                       # Prometheus metrics
    ├── tracing.py                       # Jaeger tracing
    └── health_checks.py                 # Health check definitions
```

### 1.2 Module Header Template (REQUIRED)

Every `.py` file MUST start with:

```python
"""Module-level docstring (one sentence).

Detailed description of module purpose, key responsibilities, and
architecture role. Include usage examples for public APIs.

Example:
    >>> from ollama.services.auth import TokenManager
    >>> manager = TokenManager()
    >>> token = manager.create_token(user_id="user-123")
    >>> print(token[:20] + "...")
    eyJhbGc...

Note:
    Any important warnings or implementation notes.
"""

from __future__ import annotations

import logging
from typing import Optional, Any
from abc import ABC, abstractmethod

from fastapi import HTTPException
import structlog

from ollama.config import settings
from ollama.exceptions import OllamaError

log = structlog.get_logger(__name__)
```

---

## TIER 2: Test Structure (tests/)

### 2.1 Test Mirror Requirement

**MANDATE**: Tests MUST mirror the source structure exactly.

```
Source Structure          →  Test Structure
─────────────────────────────────────────────────
ollama/config/            →  tests/unit/config/
ollama/services/          →  tests/unit/services/
ollama/repositories/      →  tests/unit/repositories/
ollama/api/routes/        →  tests/integration/api/
```

### 2.2 Test Directory Layout

```
tests/
├── __init__.py                          # Empty
├── conftest.py                          # Pytest configuration
├── fixtures/
│   ├── __init__.py
│   ├── models.py                        # Mock models
│   ├── db.py                            # Database fixtures
│   └── auth.py                          # Auth fixtures
│
├── unit/                                # Fast, isolated tests
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── test_settings.py             # Tests for settings.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── test_model_service.py        # Mirrors services/model_service.py
│   │   ├── test_inference_service.py
│   │   └── test_auth_service.py
│   └── repositories/
│       ├── __init__.py
│       ├── test_user_repository.py
│       └── test_api_key_repository.py
│
├── integration/                         # With services (slower)
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── test_health_endpoint.py
│   │   ├── test_models_endpoint.py
│   │   └── test_inference_endpoint.py
│   └── db/
│       ├── __init__.py
│       └── test_transactions.py
│
├── e2e/                                 # Full stack tests
│   ├── __init__.py
│   └── test_complete_workflow.py
│
├── performance/                         # Benchmarks
│   ├── __init__.py
│   ├── test_inference_latency.py
│   └── test_api_throughput.py
│
└── security/                            # Security validation
    ├── __init__.py
    ├── test_auth_bypass.py
    └── test_rate_limiting.py
```

### 2.3 Test File Naming Convention

```
Source Class              →  Test File
────────────────────────────────────────────
ModelService             →  test_model_service.py
ModelCacheRepository     →  test_model_cache_repository.py
TokenManager             →  test_token_manager.py
```

**Test Class Naming**:

```python
class TestModelService:           # Test class for ModelService
    def test_load_model_success(self): ...
    def test_load_model_not_found(self): ...
    async def test_generate_async(self): ...
```

---

## TIER 3: Configuration & Assets

### 3.1 config/ Directory (Non-Code)

```
config/                                 # Static configuration files
├── development.yaml                    # Dev environment
├── production.yaml                     # Production environment
├── staging.yaml                        # Staging environment
└── schemas/                            # Configuration schemas
    └── __init__.py
```

### 3.2 Docker Configuration

```
docker/
├── Dockerfile                          # Multi-stage production image
├── Dockerfile.dev                      # Development image
├── Dockerfile.minimal                  # Minimal test image
├── postgres/
│   └── init.sql                        # Database initialization
├── redis/
│   └── redis.conf                      # Redis configuration
└── nginx/
    └── nginx.conf                      # Reverse proxy configuration
```

### 3.3 Documentation

```
docs/
├── README.md                           # Documentation index
├── ARCHITECTURE.md                     # System architecture
├── API_DESIGN.md                       # API design principles
├── DEPLOYMENT.md                       # Deployment procedures
├── FAANG_STANDARDS.md                 # Elite standards guide
├── TROUBLESHOOTING.md                 # Common issues
└── images/                             # Documentation images
```

---

## TIER 4: Naming Conventions (Strict)

### 4.1 File Naming Rules

| Type              | Format                 | Example                    |
| ----------------- | ---------------------- | -------------------------- |
| Module            | `snake_case.py`        | `model_service.py`         |
| Class             | `PascalCase`           | `class ModelService:`      |
| Function          | `snake_case`           | `def generate_response():` |
| Constant          | `SCREAMING_SNAKE_CASE` | `MAX_RETRIES = 3`          |
| Private Function  | `_snake_case`          | `def _validate_input():`   |
| Internal Variable | `__double_underscore`  | `__instance: Optional[T]`  |

### 4.2 Directory Naming Rules

- **Plural for collections**: `services/`, `repositories/`, `models/`
- **Singular for layered architecture**: `middleware/`, `monitoring/`
- **Avoid abbreviations**: `config/` not `cfg/`, `services/` not `svcs/`
- **Lowercase only**: NO `Services/` or `SERVICES/`
- **Hyphens for multi-word**: `api-routes/` not `apiRoutes/` or `api_routes/`

### 4.3 Import Organization (STRICT)

Every Python file MUST organize imports in this exact order:

```python
"""Module docstring."""

from __future__ import annotations

# Standard library (alphabetical)
import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

# Third-party (alphabetical)
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, validator
from sqlalchemy import Column, String, DateTime
import structlog

# Local imports (absolute, then relative)
from ollama.config import settings
from ollama.exceptions import OllamaError
from . import constants
from .models import User

log = structlog.get_logger(__name__)
```

---

## TIER 5: One Class Per File Rule

### 5.1 MANDATE: Absolute

```
❌ WRONG (Multiple classes in one file):
# services/models.py
class ModelLoader: ...
class ModelCache: ...
class ModelMetrics: ...

✅ CORRECT (One class per file):
# services/model_loader.py
class ModelLoader: ...

# services/model_cache.py
class ModelCache: ...

# services/model_metrics.py
class ModelMetrics: ...
```

### 5.2 Exception: Constants & Enums

```python
# models/enums.py (ALLOWED: Multiple enums in one file)
from enum import Enum

class ModelStatus(Enum):
    LOADING = "loading"
    READY = "ready"

class CacheStatus(Enum):
    MISS = "miss"
    HIT = "hit"

# models/constants.py (ALLOWED: Multiple constants)
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30
BATCH_SIZE = 32
```

---

## TIER 6: **init**.py Files

### 6.1 Package Initialization Rules

```python
# ✅ CORRECT: Minimal __init__.py
# ollama/__init__.py
__version__ = "0.1.0"

# ✅ CORRECT: Explicit exports
# ollama/api/__init__.py
from .router import router
__all__ = ["router"]

# ✅ CORRECT: Service package (no direct imports)
# ollama/services/__init__.py
# Empty: Services are imported by specific module name

# ❌ WRONG: Wildcard imports
# from .models import *  # FORBIDDEN

# ❌ WRONG: Complex initialization logic
# if __name__ == "__main__":  # WRONG: Init should not have side effects
```

### 6.2 When to Add **init**.py

- **Always** for packages (directories with modules)
- **Never** for simple utility folders (unless they're packages)
- Keep minimal: Usually just version or explicit exports

---

## TIER 7: File Size Guidelines

| File Type    | Ideal Size    | Maximum   | Enforcement                |
| ------------ | ------------- | --------- | -------------------------- |
| Module (.py) | 200-400 lines | 600 lines | Soft limit, review if over |
| Class        | 100-300 lines | 500 lines | Extract methods if over    |
| Function     | 20-50 lines   | 100 lines | Refactor if over           |
| Test file    | 300-500 lines | 800 lines | Split into focused tests   |
| Docstring    | 5-15 lines    | 30 lines  | Be concise                 |

---

## TIER 8: Folder Structure Enforcement

### 8.1 Pre-commit Hook

```python
# scripts/check_folder_structure.py
#!/usr/bin/env python3
"""Validate folder structure against standards."""

import sys
from pathlib import Path

FORBIDDEN_DIRS = {
    "Utils", "Utility", "utils_old", "old_code", "backup",
    "temp", "tmp", "test_", "tests_old", "__old__"
}

REQUIRED_STRUCTURE = {
    "ollama/config": "Configuration layer",
    "ollama/api": "HTTP API layer",
    "ollama/services": "Business logic",
    "ollama/repositories": "Data access",
    "tests/unit": "Unit tests",
    "tests/integration": "Integration tests",
    "docs": "Documentation"
}

def check_structure():
    """Validate folder structure."""
    root = Path(".")

    # Check required directories exist
    for required_dir in REQUIRED_STRUCTURE.keys():
        if not (root / required_dir).exists():
            print(f"ERROR: Missing required directory: {required_dir}")
            return False

    # Check no forbidden directories exist
    for forbidden in FORBIDDEN_DIRS:
        if (root / forbidden).exists():
            print(f"ERROR: Forbidden directory found: {forbidden}")
            return False

    # Check one class per file (except special cases)
    for py_file in (root / "ollama").rglob("*.py"):
        if py_file.stem == "__init__":
            continue
        # Implementation to check class count

    print("✅ Folder structure is valid")
    return True

if __name__ == "__main__":
    sys.exit(0 if check_structure() else 1)
```

### 8.2 CI/CD Validation

```yaml
# .github/workflows/folder-structure.yml
name: Validate Folder Structure

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Check folder structure
        run: python scripts/check_folder_structure.py
```

---

## TIER 9: Git Hygiene

### 9.1 File Commit Guidelines

**DO commit**:

- Source code (`.py`)
- Documentation (`.md`)
- Configuration (`.yaml`, `.toml`)
- Tests
- GitHub workflows
- Dockerfile

**DON'T commit**:

- `__pycache__/`
- `.pytest_cache/`
- `.mypy_cache/`
- `.ruff_cache/`
- `venv/` or `.venv/`
- `.env` (use `.env.example`)
- `*.pyc`, `*.pyo`
- `htmlcov/`
- `dist/`, `build/`
- IDE-specific files (`.idea/`, `.DS_Store`)

### 9.2 .gitignore Template

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
.venv/
ENV/
env/

# Testing
.pytest_cache/
.coverage
htmlcov/
.mypy_cache/
.ruff_cache/

# IDE
.vscode/settings.json  # Ignore personal settings
.idea/
*.swp
*.swo
.DS_Store

# Environment
.env
.env.local
.env.*.local

# Build
dist/
build/
*.egg-info/
```

---

## TIER 10: Enforcement Checklist

Use this checklist on every commit:

- [ ] All files in correct directories
- [ ] No forbidden directories present
- [ ] File naming follows snake_case
- [ ] Module docstrings present
- [ ] One class per file (except constants/enums)
- [ ] **init**.py files minimal
- [ ] No circular imports
- [ ] Test files mirror source structure
- [ ] File sizes within guidelines
- [ ] No gitignored files committed

---

## Quick Reference: Moving Files

```bash
# If you need to reorganize, use git mv
git mv old_path/file.py new_path/file.py

# Verify
git status

# Commit with appropriate message
git commit -S -m "refactor(structure): reorganize module location"
```

---

**Version**: 3.0.0
**Last Updated**: January 14, 2026
**Status**: MANDATORY STANDARD - NO EXCEPTIONS
