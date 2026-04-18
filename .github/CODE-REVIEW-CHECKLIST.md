# Code Review Checklist - FAANG Elite Standards

This checklist ensures all code meets Top 0.01% (FAANG-level) standards before merge.

**Purpose**: Enforce consistency, quality, and security across all pull requests.

---

## Automatic Checks (Pre-Review)

These checks run automatically in CI/CD. **ALL MUST PASS** before human review:

- [ ] ✅ MyPy type checking: `mypy ollama/ --strict` passes
- [ ] ✅ Ruff linting: `ruff check ollama/` passes
- [ ] ✅ Code formatting: `black ollama/` compliant
- [ ] ✅ Test coverage: ≥95% (shown in PR comment)
- [ ] ✅ Security audit: `pip-audit` + `bandit` pass
- [ ] ✅ Commit signatures: All commits GPG signed
- [ ] ✅ Tests passing: All 3 Python versions (3.9, 3.10, 3.11)

**If any automatic check fails**: PR is blocked. Fix and retry. No exceptions.

---

## Code Quality Review

### Type Safety

- [ ] Every function has parameter type hints
- [ ] Every function has return type annotation
- [ ] No `Any` without `# type: ignore` with justification
- [ ] No implicit `Optional` (use `X | None`)
- [ ] Type hints match actual implementation
- [ ] Docstring types match parameter types

**Example**:

```python
# ❌ REVIEW: Missing types
def process_request(request, data):
    return {"status": "ok"}

# ✅ ACCEPT: Full types
def process_request(request: Request, data: dict) -> dict[str, str]:
    """Process request with data."""
    return {"status": "ok"}
```

### Naming Conventions

- [ ] Files use `lowercase_with_underscores.py`
- [ ] Classes use `PascalCase`
- [ ] Functions use `snake_case`
- [ ] Constants use `SCREAMING_SNAKE_CASE`
- [ ] Folder names are lowercase with underscores
- [ ] No single-letter variables (except loop `i`, `j`, `k`)
- [ ] Names are descriptive (>3 chars, <30 chars)

**Example**:

```python
# ❌ REVIEW: Single letter variable, camelCase function
def getUserData(u):
    return u.data

# ✅ ACCEPT: Clear names
def get_user_data(user: User) -> dict[str, Any]:
    """Get user data."""
    return user.data
```

### Structure & Organization

- [ ] **One class per file** (exceptions: schemas.py, types.py, exceptions.py)
- [ ] File is <600 lines
- [ ] Functions are <50 lines (ideal) or <100 lines (acceptable)
- [ ] Module has docstring explaining purpose
- [ ] Imports organized: stdlib → third-party → local
- [ ] Imports are sorted alphabetically
- [ ] No circular imports
- [ ] Related functionality grouped together

**Example**:

```python
# ❌ REVIEW: Multiple unrelated classes
# File: services/auth.py
class LoginService:
    pass
class SessionManager:
    pass

# ✅ ACCEPT: Proper separation
# File: services/auth.py
class LoginService:
    pass

# File: services/session.py
class SessionManager:
    pass
```

### Error Handling

- [ ] Custom exception hierarchy used (not built-ins)
- [ ] Exceptions provide context (message + data)
- [ ] No silent failures (`except: pass`)
- [ ] No bare `except` clauses
- [ ] Specific exception types caught (not `Exception`)
- [ ] Errors logged with context
- [ ] Error responses include error codes

**Example**:

```python
# ❌ REVIEW: Silent failure
try:
    model = load_model(name)
except:
    pass

# ✅ ACCEPT: Explicit handling
try:
    model = load_model(name)
except ModelNotFoundError as e:
    log.error("model_not_found", model_name=name)
    raise APIError(
        code="MODEL_NOT_FOUND",
        message=f"Model {name} not available"
    ) from e
```

### Complexity

- [ ] Cyclomatic complexity ≤10 per function
- [ ] Cognitive complexity ≤10 per function
- [ ] Nested depth ≤4 levels
- [ ] Function has single responsibility
- [ ] Parameter count ≤4 (use dataclass for more)
- [ ] Function parameters are validated

**Measurement**:

```bash
# Check complexity
flake8 ollama/ --select=C901  # Cyclomatic
flake8 ollama/ --select=CCR   # Cognitive
```

---

## Testing Review

### Coverage

- [ ] Overall coverage ≥95%
- [ ] All code paths tested (including error paths)
- [ ] Happy path tests exist
- [ ] Edge case tests exist
- [ ] Error condition tests exist
- [ ] Test file mirrors module structure
- [ ] Test class mirrors main class name
- [ ] Test method names descriptive

**Example**:

```python
# Module: ollama/services/llm.py
class LLMService:
    async def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Empty prompt")
        return llm.generate(prompt)

# Tests: tests/unit/services/test_llm.py (MIRRORS structure)
class TestLLMService:
    @pytest.mark.asyncio
    async def test_generate_returns_string(self) -> None:
        """Happy path: Returns string response."""
        service = LLMService()
        result = await service.generate("test")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_generate_raises_on_empty_prompt(self) -> None:
        """Error path: Raises on empty input."""
        service = LLMService()
        with pytest.raises(ValueError, match="Empty prompt"):
            await service.generate("")
```

### Test Quality

- [ ] Tests are isolated (no dependencies on each other)
- [ ] Tests don't mock external dependencies (use fixtures)
- [ ] Assertions are specific (not just checking return type)
- [ ] Test names describe what's being tested
- [ ] Fixtures properly cleaned up
- [ ] No hardcoded data (use factories/fixtures)
- [ ] Tests run in <100ms (unit tests)

---

## Documentation Review

### Code Documentation

- [ ] Module has docstring
- [ ] Public classes have docstrings
- [ ] Public functions have docstrings
- [ ] Docstrings include description
- [ ] Docstrings include parameters (type + meaning)
- [ ] Docstrings include return value
- [ ] Docstrings include exceptions raised
- [ ] Docstrings include usage example (if complex)

**Format** (Google-style):

```python
class MyService:
    """Service for processing requests.

    Handles validation, transformation, and persistence of request data.
    """

    def process(
        self,
        request: Request,
        options: dict[str, str] | None = None,
    ) -> Response:
        """Process incoming request.

        Args:
            request: HTTP request object
            options: Optional processing options

        Returns:
            Response object with results

        Raises:
            ValidationError: If request invalid
            ProcessingError: If processing fails

        Example:
            >>> service = MyService()
            >>> response = service.process(request)
            >>> print(response.status)
            'success'
        """
```

### API Documentation

- [ ] Endpoint documented in code comment
- [ ] Response format documented
- [ ] Error responses documented
- [ ] Rate limits documented (if applicable)
- [ ] Authentication requirements documented
- [ ] Example requests/responses provided

---

## Security Review

### Authentication & Authorization

- [ ] API keys required for all endpoints
- [ ] No hardcoded credentials
- [ ] Credentials loaded from env vars
- [ ] Credentials never logged
- [ ] Session tokens validated
- [ ] CORS properly configured

### Input Validation

- [ ] User input validated with Pydantic
- [ ] Request size limits enforced
- [ ] Rate limits enforced
- [ ] No SQL injection vulnerabilities
- [ ] No XSS vulnerabilities
- [ ] Sensitive data not logged

### Data Protection

- [ ] Sensitive data encrypted at rest
- [ ] HTTPS enforced for public endpoints
- [ ] No PII in error messages
- [ ] Audit logging for sensitive operations

---

## Performance Review

### Efficiency

- [ ] No N+1 query problems
- [ ] Database queries optimized (have indexes)
- [ ] No unnecessary loops/recursion
- [ ] No memory leaks (proper cleanup)
- [ ] Async/await used appropriately
- [ ] Connection pooling used

### Scalability

- [ ] Code handles large datasets
- [ ] No hardcoded limits that could break
- [ ] Pagination used where needed
- [ ] Caching used for expensive operations

---

## Git & PR Review

### Commits

- [ ] All commits GPG signed
- [ ] Commit messages follow format: `type(scope): description`
- [ ] Commits are atomic (one concern per commit)
- [ ] Commits are reversible
- [ ] No merge commits (rebase instead)
- [ ] No massive commits (>500 lines)

### PR Metadata

- [ ] PR title matches commit format
- [ ] PR description explains WHAT and WHY
- [ ] Related issue linked (Fixes #123)
- [ ] No merge conflicts
- [ ] Target branch is correct (main/develop)
- [ ] Feature branch deleted after merge

**Example PR**:

```
Title: feat(api): add streaming response support

Description:
## What
Add server-sent events (SSE) support to inference endpoint for streaming
partial responses as tokens are generated.

## Why
Improves perceived latency by 40% and enables real-time token display
in frontends.

## Testing
- Added unit tests for SSE encoder
- Added integration tests with streaming client
- Manual testing with curl and WebSocket client
- Coverage: 97%

## Related
Fixes #234
Implements RFC-001
```

---

## Folder Structure Review

- [ ] Code in `ollama/` (not `src/` or other)
- [ ] Tests in `tests/` mirroring `ollama/`
- [ ] Required directories exist:
  - [ ] `ollama/config/`
  - [ ] `ollama/api/routes/`
  - [ ] `ollama/api/schemas/`
  - [ ] `ollama/services/`
  - [ ] `ollama/repositories/`
  - [ ] `ollama/models.py`
  - [ ] `ollama/middleware/`
  - [ ] `ollama/monitoring/`
  - [ ] `tests/unit/`
  - [ ] `tests/integration/`
- [ ] No forbidden directories (Utils, old_code, etc.)

---

## Final Checklist Before Approval

| Check                         | Status   |
| ----------------------------- | -------- |
| ✅ All CI/CD checks pass      | Required |
| ✅ Code quality standards met | Required |
| ✅ Test coverage ≥95%         | Required |
| ✅ Documentation complete     | Required |
| ✅ Security review passed     | Required |
| ✅ Performance acceptable     | Required |
| ✅ Commits properly signed    | Required |
| ✅ Folder structure correct   | Required |

**Result**:

- ✅ **APPROVE** if ALL checks pass
- ❌ **REQUEST CHANGES** if any check fails
- ⏸️ **COMMENT** if clarification needed before review

---

## Reviewer Commands

```bash
# Review locally
git checkout -b review/feature-name
git pull origin feature-name

# Run full test suite
pytest tests/ -v --cov=ollama --cov-fail-under=95

# Check types
mypy ollama/ --strict

# Verify standards compliance
python scripts/validate-standards.py --verbose

# Check commit signatures
git log --oneline --no-decorate feature-name..main | while read commit; do
  git verify-commit $(echo "$commit" | awk '{print $1}') && echo "✅ $commit" || echo "❌ $commit"
done
```

---

## Feedback Templates

### Constructive Feedback

````markdown
### Type Safety Issue

Found missing type hints in function:

```python
def process_data(items):  # ❌ No types
    return [x.value for x in items]
```
````

**Fix**: Add parameter and return types

```python
def process_data(items: list[DataItem]) -> list[Any]:  # ✅ Types added
    return [x.value for x in items]
```

This ensures mypy --strict passes.

````

### Coverage Issue

```markdown
### Test Coverage Gap

Lines 45-50 in `ollama/services/auth.py` not covered:

```python
45: if not token:
46:     raise InvalidTokenError()  # ❌ Not tested
````

**Fix**: Add test case

```python
def test_raises_on_empty_token(self) -> None:
    """InvalidTokenError raised for empty token."""
    with pytest.raises(InvalidTokenError):
        authenticate("")
```

This brings coverage to 95%+.

```

---

**Remember**: Code review is mentoring, not gatekeeping. Review with kindness and clarity.

**Last Updated**: January 14, 2026
**Status**: 🟢 In Use
```
