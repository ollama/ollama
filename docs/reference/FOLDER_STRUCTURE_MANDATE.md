# Folder Structure Mandate - 5 Levels Deep

**CRITICAL MANDATE**: Filesystem structure is non-negotiable. All code must conform to this 5-level hierarchy with clear separation of concerns at each level.

## Quick Reference

| Level | Purpose        | Example                          | Max Dirs | Max Files | Rules                                                  |
| ----- | -------------- | -------------------------------- | -------- | --------- | ------------------------------------------------------ |
| 1     | Root           | `/ollama/`                       | 12       | 5         | Top-level dirs only, no arbitrary subdirs              |
| 2     | Package        | `ollama/`                        | 12       | 8         | Domain-oriented dirs + single-responsibility modules   |
| 3     | Domain         | `ollama/api/`                    | 4        | 1         | Only `__init__.py`, functional containers below        |
| 4     | Functional     | `ollama/api/routes/`             | 20       | 20        | One resource per file, max 500 lines                   |
| 5     | Implementation | `ollama/api/routes/inference.py` | 0        | 1         | Actual code: constants → schemas → classes → functions |

## Level 1: Root (Project Root)

**Purpose**: Top-level project organization

**Location**: `/home/akushnir/ollama/`

**Structure**:

```
/home/akushnir/ollama/
├── ollama/              # Main application package
├── tests/               # Test suite
├── docs/                # Documentation
├── config/              # Configuration files
├── docker/              # Docker assets
├── k8s/                 # Kubernetes manifests
├── scripts/             # Automation scripts
├── alembic/             # Database migrations
├── .github/             # GitHub configurations
├── .vscode/             # VSCode settings
├── .githooks/           # Git hooks
└── pyproject.toml       # Project config
```

**Rules**:

- ✅ **MANDATE**: Maximum 12 top-level directories
- ✅ **MANDATE**: Maximum 5 top-level .py files (setup.py, main.py, etc.)
- ✅ **MANDATE**: Only configuration files at root (pyproject.toml, setup.py, Dockerfile)
- ❌ **FORBIDDEN**: Arbitrary directories (src/, lib/, app/, common/)
- ❌ **FORBIDDEN**: Python code files except specific ones (setup.py, main.py)
- ❌ **FORBIDDEN**: Mixing test files with application files

**Allowed Top-Level Directories**:

- `ollama/` - Main application
- `tests/` - Test suite
- `docs/` - Documentation
- `config/` - Configuration
- `docker/` - Docker resources
- `k8s/` - Kubernetes
- `scripts/` - Automation
- `alembic/` - DB migrations
- `.github/` - GitHub configs
- `.vscode/` - VSCode configs
- `.githooks/` - Git hooks
- `monitoring/` - Observability

**Validation**:

```bash
# Check directory count
ls -d */ | wc -l  # Should be ≤ 12

# Check for forbidden directories
ls -d */ | grep -E "(src|lib|app|common|core)"  # Should be empty
```

---

## Level 2: Application Package (ollama/)

**Purpose**: Domain-oriented organization within main package

**Location**: `ollama/`

**Structure**:

```
ollama/
├── api/                 # HTTP API domain
├── auth/                # Authentication domain
├── config/              # Configuration domain
├── exceptions/          # Exception hierarchy
├── middleware/          # Middleware domain
├── models/              # ORM models domain
├── monitoring/          # Observability domain
├── repositories/        # Data access domain
├── services/            # Business logic domain
├── utils/               # Cross-cutting utilities
├── main.py              # FastAPI entry point
├── config.py            # Settings loader
├── metrics.py           # Metrics registry
├── client.py            # HTTP client
├── auth_manager.py      # Auth utilities
└── __init__.py
```

**Rules**:

- ✅ **MANDATE**: Maximum 12 subdirectories
- ✅ **MANDATE**: Maximum 8 .py files at this level
- ✅ **MANDATE**: Each subdirectory = one domain responsibility
- ✅ **MANDATE**: Single-responsibility .py files (main.py, config.py, client.py)
- ✅ **MANDATE**: No duplicate names (e.g., can't have both `api.py` AND `api/`)
- ✅ **MANDATE**: Directory names lowercase, plural for collections
- ❌ **FORBIDDEN**: Arbitrary files (utils.py without utils/ domain)
- ❌ **FORBIDDEN**: Mixed responsibilities (api_and_auth/, models_and_services/)

**Domain Categories**:

- `api/` - HTTP routing and endpoint definitions
- `auth/` - Authentication and authorization
- `config/` - Configuration management
- `exceptions/` - Custom exception hierarchy
- `middleware/` - Request/response processing
- `models/` - SQLAlchemy ORM models
- `monitoring/` - Logging, metrics, tracing
- `repositories/` - Data access layer
- `services/` - Business logic
- `utils/` - Cross-cutting utilities

**Single-Responsibility Modules** (allowed at Level 2):

- `main.py` - FastAPI app setup and lifespan
- `config.py` - Settings loader and environment config
- `metrics.py` - Prometheus metrics registry
- `client.py` - External HTTP client
- `auth_manager.py` - Password hashing and auth utilities

**Validation**:

```bash
# Check directory count
ls -d ollama/*/ | wc -l  # Should be ≤ 12

# Check file count
ls -1 ollama/*.py | wc -l  # Should be ≤ 8

# Check for mixed names
ls ollama/ | sort | uniq -d  # Should be empty
```

---

## Level 3: Domain Container (ollama/api/)

**Purpose**: Clear boundary for domain - only functional containers allowed

**Location**: `ollama/{domain}/`

**Structure Examples**:

### Example 1: ollama/api/

```
ollama/api/
├── routes/              # Route handlers container
├── schemas/             # Pydantic models container
├── dependencies/        # FastAPI dependencies container
└── __init__.py
```

### Example 2: ollama/services/

```
ollama/services/
├── inference/           # Inference service container
│   ├── generator.py     # Text generation (Level 5)
│   ├── embeddings.py    # Embeddings (Level 5)
│   └── __init__.py
├── ollama_client_main.py # Ollama client (Level 4)
├── cache_manager.py     # Caching (Level 4)
└── __init__.py
```

### Example 3: ollama/repositories/

```
ollama/repositories/
├── user.py              # User repository (Level 4)
├── conversation.py      # Conversation repository (Level 4)
└── __init__.py
```

**Rules**:

- ✅ **MANDATE**: Only `__init__.py` file allowed at Level 3
- ✅ **MANDATE**: Maximum 4 subdirectories per domain
- ✅ **MANDATE**: Each subdirectory is a functional container
- ✅ **MANDATE**: Clear domain boundary - Level 3 is ONLY a container
- ❌ **FORBIDDEN**: Python code files except `__init__.py`
- ❌ **FORBIDDEN**: Business logic at Level 3
- ❌ **FORBIDDEN**: More than 4 subdirectories
- ❌ **FORBIDDEN**: Generic containers (utils/, common/, helpers/)

**Functional Container Types** (Level 4):

- `routes/` - API route handlers
- `schemas/` - Request/response schemas
- `dependencies/` - FastAPI dependency injection
- `generators/` - Generation logic
- `embeddings/` - Embedding logic
- `completion/` - Completion logic

**Validation**:

```bash
# Check that only __init__.py exists
ls -1 ollama/api/*.py  # Should ONLY show __init__.py

# Count subdirectories
ls -d ollama/api/*/ | wc -l  # Should be ≤ 4
```

---

## Level 4: Functional Container (ollama/api/routes/)

**Purpose**: Specialization - actual implementation containers

**Location**: `ollama/{domain}/{functional_container}/`

**Structure**:

```
ollama/api/routes/
├── inference.py         # Inference endpoint
├── chat.py              # Chat endpoint
├── documents.py         # Document operations
├── embeddings.py        # Embeddings endpoint
├── models.py            # Model management
└── __init__.py
```

**File Naming Convention**:

- `inference.py` - Handles `/inference` endpoints
- `chat.py` - Handles `/chat` endpoints
- `documents.py` - Handles `/documents` endpoints
- `user.py` - Handles `/users` endpoints or User repository operations
- `conversation.py` - Handles conversation operations

**Rules**:

- ✅ **MANDATE**: Maximum 20 .py files per Level 4 directory
- ✅ **MANDATE**: One resource/responsibility per file
- ✅ **MANDATE**: Files named by resource (lowercase_with_underscores.py)
- ✅ **MANDATE**: File size: 100-500 lines (split if larger)
- ✅ **MANDATE**: File names follow pattern: `^[a-z][a-z0-9_]*\.py$`
- ❌ **FORBIDDEN**: Nested directories (no Level 5+ subdirectories)
- ❌ **FORBIDDEN**: Generic names (main.py, utils.py, helpers.py, common.py)
- ❌ **FORBIDDEN**: CamelCase files (MyService.py)
- ❌ **FORBIDDEN**: More than 500 lines per file

**File Structure Template**:

```python
"""Module description in one sentence.

Detailed description (2-3 sentences) explaining what this module does,
its responsibilities, and how it fits into the system.
"""

# ✅ Step 1: Imports (organized)
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# ✅ Step 2: Constants
DEFAULT_TIMEOUT = 30
MAX_TOKENS = 2048

# ✅ Step 3: Schemas
class GenerateRequest(BaseModel):
    """Request schema for generation endpoint."""
    prompt: str
    model: str

# ✅ Step 4: Route handlers
router = APIRouter()

@router.post("/generate")
async def generate(request: GenerateRequest) -> dict:
    """Generate endpoint."""
    pass

# ✅ Step 5: Helper functions
def _validate_model(model: str) -> bool:
    """Internal validation helper."""
    pass
```

**Validation**:

```bash
# Count files
ls -1 ollama/api/routes/*.py | wc -l  # Should be ≤ 20

# Check file sizes
wc -l ollama/api/routes/*.py | sort -n  # All should be < 500

# Check naming
ls ollama/api/routes/*.py | grep -E "[A-Z]|_test\.py|main\.py|utils\.py"  # Should be empty
```

---

## Level 5: Implementation Details (ollama/api/routes/inference.py)

**Purpose**: Actual code implementation - constants, schemas, classes, functions

**Location**: `ollama/{domain}/{container}/{resource}.py`

**Structure**:

```python
"""Inference endpoint handlers.

Provides HTTP endpoints for text generation, chat completion,
and other inference operations via Ollama.
"""

# Step 1: Imports
from typing import Optional, AsyncGenerator
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import structlog

from ollama.services.inference import InferenceEngine
from ollama.exceptions import ModelNotFoundError

# Step 2: Constants
DEFAULT_TIMEOUT = 30
MAX_TOKENS = 2048
SUPPORTED_MODELS = ["llama3.2", "mistral", "neural-chat"]

# Step 3: Schemas (Pydantic models)
class GenerateRequest(BaseModel):
    """Request for text generation."""
    prompt: str = Field(..., min_length=1, max_length=10000)
    model: str = Field(..., regex="^[a-z0-9._-]+$")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

class GenerateResponse(BaseModel):
    """Response from generation endpoint."""
    text: str
    model: str
    tokens_used: int

# Step 4: Classes
class InferenceHandler:
    """Manages inference request handling."""

    def __init__(self, engine: InferenceEngine) -> None:
        """Initialize with inference engine."""
        self.engine = engine

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Execute generation request."""
        pass

# Step 5: Public functions
router = APIRouter()

@router.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """Generate text completion.

    Args:
        request: Generation request with prompt and model

    Returns:
        Generated text with metadata

    Raises:
        ModelNotFoundError: If model not found
    """
    pass

# Step 6: Private helper functions
def _validate_model(model: str) -> bool:
    """Check if model is supported."""
    return model in SUPPORTED_MODELS

log = structlog.get_logger(__name__)
```

**Rules**:

- ✅ **MANDATE**: File size maximum 500 lines
- ✅ **MANDATE**: Maximum 10 functions per file
- ✅ **MANDATE**: Maximum 1 public class per file
- ✅ **MANDATE**: Organization: Constants → Schemas → Classes → Public Functions → Private Functions
- ✅ **MANDATE**: All functions have type hints
- ✅ **MANDATE**: All classes have docstrings
- ✅ **MANDATE**: Helper functions prefixed with `_`
- ✅ **MANDATE**: Module docstring at top
- ❌ **FORBIDDEN**: Nested directories
- ❌ **FORBIDDEN**: Multiple public classes
- ❌ **FORBIDDEN**: Generic code (utils, helpers, common logic at this level)
- ❌ **FORBIDDEN**: Untyped functions

**Validation**:

```bash
# Check line count
wc -l ollama/api/routes/inference.py  # Should be < 500

# Check for multiple classes
grep "^class [A-Z]" ollama/api/routes/inference.py | wc -l  # Should be ≤ 1

# Check for type hints (via mypy)
mypy ollama/api/routes/inference.py --strict  # Must pass
```

---

## Test Structure Mirroring

**MANDATE**: `tests/` directory structure EXACTLY mirrors `ollama/` structure.

### Mirroring Rules:

**Rule 1: File Location Mirroring**

```
ollama/api/routes/inference.py
    ↓
tests/unit/api/routes/test_inference.py

ollama/services/inference/generator.py
    ↓
tests/unit/services/inference/test_generator.py

ollama/repositories/user.py
    ↓
tests/unit/repositories/test_user.py
```

**Rule 2: Directory Structure**

```
tests/
├── unit/                   # Fast, isolated tests
│   ├── api/               # Mirror ollama/api/
│   │   ├── routes/        # Mirror ollama/api/routes/
│   │   │   ├── test_inference.py
│   │   │   ├── test_chat.py
│   │   │   └── __init__.py
│   │   ├── schemas/       # Mirror ollama/api/schemas/
│   │   │   ├── test_inference.py
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── services/          # Mirror ollama/services/
│   │   ├── inference/     # Mirror ollama/services/inference/
│   │   │   ├── test_generator.py
│   │   │   └── __init__.py
│   │   ├── test_ollama_client_main.py
│   │   └── __init__.py
│   ├── repositories/      # Mirror ollama/repositories/
│   │   ├── test_user.py
│   │   └── __init__.py
│   ├── conftest.py        # Root fixtures
│   └── __init__.py
├── integration/           # Tests with service interactions
│   ├── test_api_flow.py
│   ├── conftest.py
│   └── __init__.py
├── fixtures/              # Shared test fixtures
│   ├── models.py
│   ├── auth.py
│   └── __init__.py
└── __init__.py
```

**Rule 3: Test File Naming**

- Test file: `test_{resource}.py`
- Test class: `Test{Resource}` (e.g., `TestInferenceEndpoint`)
- Test method: `test_{action}_{scenario}` (e.g., `test_generate_success`)

**Rule 4: conftest.py Placement**

- Root: `tests/conftest.py` - Global fixtures
- Unit: `tests/unit/conftest.py` - Unit test fixtures
- Integration: `tests/integration/conftest.py` - Integration test fixtures
- Subdirectories: `tests/unit/api/conftest.py` - Domain-specific fixtures

---

## Naming Conventions

### Modules (.py files)

- **Pattern**: `^[a-z][a-z0-9_]*\.py$`
- **Examples**:
  - ✅ `inference.py`
  - ✅ `oauth_handler.py`
  - ✅ `vector_database.py`
  - ❌ `InferenceEndpoint.py`
  - ❌ `Inference.py`
  - ❌ `inference_endpoint.py` (too specific at Level 4)
- **Rule**: MANDATE - lowercase_with_underscores.py (snake_case)

### Classes

- **Pattern**: `^[A-Z][a-zA-Z0-9]*$`
- **Examples**:
  - ✅ `InferenceEngine`
  - ✅ `OAuthHandler`
  - ✅ `VectorDatabase`
  - ❌ `inference_engine`
  - ❌ `Inference_Engine`
  - ❌ `INFERENCEENGINE`
- **Rule**: MANDATE - PascalCase (CapWords)

### Functions

- **Pattern**: `^[a-z_][a-z0-9_]*$`
- **Examples**:
  - ✅ `generate_response()`
  - ✅ `verify_token()`
  - ✅ `get_user_by_id()`
  - ✅ `_internal_helper()` (private)
  - ❌ `GenerateResponse()`
  - ❌ `generate-response()`
  - ❌ `generateResponse()`
- **Rule**: MANDATE - snake_case for public, \_snake_case for private

### Constants

- **Pattern**: `^[A-Z][A-Z0-9_]*$`
- **Examples**:
  - ✅ `DEFAULT_TIMEOUT`
  - ✅ `MAX_RETRIES`
  - ✅ `API_VERSION`
  - ✅ `SUPPORTED_MODELS`
  - ❌ `default_timeout`
  - ❌ `DefaultTimeout`
  - ❌ `DEFAULT_timeout`
- **Rule**: MANDATE - SCREAMING_SNAKE_CASE

### Directories

- **Pattern**: `^[a-z][a-z0-9_]*$`
- **Examples**:
  - ✅ `api/` (specific domain)
  - ✅ `repositories/` (plural, collection)
  - ✅ `services/` (plural, collection)
  - ✅ `middleware/` (singular, middleware)
  - ✅ `inference/` (singular, domain)
  - ❌ `API/` (uppercase)
  - ❌ `Api/` (mixed case)
  - ❌ `api_routes/` (descriptive, use routes/ under api/)
  - ❌ `common/` (forbidden generic name)
- **Rule**: MANDATE - lowercase, plural for collections

---

## Forbidden Patterns

### At ALL Levels:

```
❌ src/          (generic top-level container)
❌ lib/          (generic library container)
❌ app/          (ambiguous application folder)
❌ application/  (verbose version of app/)
❌ common/       (violates domain separation)
❌ core/         (ambiguous core functionality)
❌ utils/        (only at ollama/ package level)
❌ helpers/      (generic helper utilities)
❌ utils.py      (in isolation - use utils/ domain)
❌ CamelCase/    (directories must be lowercase)
❌ UPPERCASE/    (directories must be lowercase)
```

### At Level 3 (Domain):

```
❌ *.py files except __init__.py
❌ Nested directories beyond 4 per domain
❌ Logic implementation (moves to Level 4)
❌ Generic containers (utils/, helpers/, common/)
```

### At Level 4+ (Implementation):

```
❌ Nested directories (no Level 6)
❌ Files > 500 lines
❌ > 20 files per directory
❌ > 10 functions per file
❌ > 1 public class per file
❌ Untyped functions
❌ Generic names (main.py, utils.py, helpers.py)
```

---

## Validation Checklist

### Pre-Commit Validation:

- [ ] No directories nested beyond 5 levels
- [ ] All .py files follow naming convention
- [ ] Max 12 directories at each level
- [ ] No duplicate file+directory names
- [ ] test/ structure mirrors ollama/ structure
- [ ] All files < 500 lines
- [ ] No forbidden directory patterns

### Code Review Validation:

- [ ] Level 3 contains only `__init__.py`
- [ ] Each directory has single responsibility
- [ ] No circular dependencies between domains
- [ ] Test files co-located with modules
- [ ] Type hints on all functions (mypy strict)
- [ ] Docstrings on all public classes/functions

### VSCode Integration:

- [ ] folder-structure.json validation active
- [ ] File exclusions hiding generated files
- [ ] Workspace settings enforcing structure
- [ ] Pre-commit hook running on commit

---

## Examples of Conforming vs Non-Conforming

### ✅ CONFORMING:

```
ollama/
├── api/
│   ├── routes/
│   │   ├── inference.py          (single resource)
│   │   └── __init__.py
│   ├── schemas/
│   │   ├── inference.py          (request/response)
│   │   └── __init__.py
│   └── __init__.py
├── services/
│   ├── inference/
│   │   ├── generator.py          (text generation)
│   │   ├── embeddings.py         (embedding generation)
│   │   └── __init__.py
│   ├── ollama_client_main.py    (Ollama client)
│   └── __init__.py
└── main.py                       (FastAPI entry point)

tests/unit/
├── api/
│   ├── routes/
│   │   ├── test_inference.py     (mirrors ollama/api/routes/inference.py)
│   │   └── __init__.py
│   └── __init__.py
└── services/
    ├── inference/
    │   ├── test_generator.py     (mirrors generator.py)
    │   └── __init__.py
    └── __init__.py
```

### ❌ NON-CONFORMING:

```
ollama/
├── api_routes/                   (WRONG: mixed concern at Level 2)
│   └── generate.py
├── services_and_models/          (WRONG: mixed responsibilities)
│   └── inference_model.py
├── utils/                        (WRONG: too generic at Level 2)
│   ├── helpers.py
│   ├── common.py
│   └── decorators.py
├── src/                          (WRONG: forbidden top-level)
│   └── main.py
└── lib/                          (WRONG: forbidden top-level)
    └── inference.py

tests/
├── test_api_generate.py          (WRONG: not mirrored structure)
└── test_inference.py             (WRONG: not organized by domain)
```

---

## Quick Start: Adding New Feature

### Step 1: Determine Domain

```
Is this HTTP routing? → ollama/api/
Is this business logic? → ollama/services/
Is this data access? → ollama/repositories/
Is this authentication? → ollama/auth/
```

### Step 2: Check Level 4 Container

```
Example: Adding /embeddings endpoint
→ Domain: api/
→ Container: routes/
→ File: ollama/api/routes/embeddings.py ✅
```

### Step 3: Check Level 5 Structure

```
File should follow:
1. Module docstring
2. Imports
3. Constants
4. Schemas (Pydantic)
5. Classes
6. Route handlers (public functions)
7. Helpers (private functions with _prefix)
```

### Step 4: Create Test File

```
New file: ollama/api/routes/embeddings.py
Test file: tests/unit/api/routes/test_embeddings.py

Mirror the directory structure exactly!
```

### Step 5: Validate

```bash
# Check structure
mypy ollama/api/routes/embeddings.py --strict
ruff check ollama/api/routes/embeddings.py
pytest tests/unit/api/routes/test_embeddings.py -v
```

---

## Enforcement Tools

### Pre-Commit Hook

**Location**: `.githooks/pre-commit`

Validates on every commit:

- Directory depth (max 5)
- Naming conventions
- Mirrored test structure
- File size limits

### VSCode Settings

**Location**: `.vscode/settings.json`

- File explorer excludes generated files
- Warnings for non-conforming patterns
- Linting integration (ruff, mypy)

### Folder Structure Reference

**Location**: `.vscode/folder-structure.json`

- Schema validation
- Quick reference for each level
- Automated checks

---

## When to Break the Rules

**NEVER break these rules without explicit approval:**

- ✅ Maximum 5 levels deep
- ✅ Naming conventions
- ✅ Single responsibility per directory
- ✅ Test file mirroring

**Can deviate with justification:**

- Number of files (> 20) if unavoidable
- File size (> 500 lines) with refactoring plan
- Directory count (> 12) if architectural necessity

**Always document deviations**:

```python
# DEVIATION: This file exceeds 500 lines because...
# Refactoring plan: [specific steps to reduce size]
# Approved by: [team lead]
# Ticket: [issue number]
```

---

## Reference Links

- [copilot-instructions.md](../.github/copilot-instructions.md) - Full guidelines
- [folder-structure.json](./.vscode/folder-structure.json) - Schema validation
- [pyproject.toml](./pyproject.toml) - Project configuration
- [.editorconfig](./.editorconfig) - Editor settings

---

**Last Updated**: January 14, 2026
**Mandate Version**: 1.0.0 (5-Level Deep, FAANG Elite Standards)
