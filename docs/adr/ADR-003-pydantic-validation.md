# ADR-003: Pydantic for All Schema Validation

**Status**: Accepted
**Date**: 2026-01-26
**Author**: @api-team

---

## Context

### Problem

API requests need validation. Currently using:

- FastAPI auto-validation (fragmented)
- Manual validation functions (inconsistent)
- Type hints only (no runtime validation)

Need unified approach that:

- Validates all inputs (API, internal functions)
- Provides clear error messages
- Works with async code
- Supports complex nested schemas

### Constraints

- Must support FastAPI (current framework)
- Must work with Python 3.11+
- Must have excellent type hint support (mypy compatible)

---

## Decision

**Chosen**: Pydantic V2 for all schema validation

Pydantic provides:

1. **Unified Validation**: Single approach across all layers
2. **Type Safety**: Full mypy integration with strict mode
3. **Performance**: V2 is 2-5x faster than alternatives
4. **Error Messages**: Clear, actionable validation errors
5. **Async Support**: Native support for async validation

---

## Consequences

### Positive

1. **Type Safety**: mypy can catch >90% of errors before runtime
2. **Consistent Errors**: All validation errors have same format
3. **Performance**: V2 is faster than manual validation + type checking
4. **Documentation**: Pydantic generates OpenAPI schemas automatically

### Negative

1. **Learning Curve**: Team must learn Pydantic patterns
2. **Runtime Overhead**: Validation takes CPU time (minimal though)
3. **Strict Mode**: May catch edge cases team didn't expect (good long-term, painful short-term)

---

## Implementation

### Approach

```python
# Before: Mixed validation strategies
def get_user(user_id: int) -> User:
    if not user_id or user_id < 0:  # Manual validation
        raise ValueError("Invalid user_id")
    # ...

# After: Unified Pydantic validation
from pydantic import BaseModel, Field, validator

class GetUserRequest(BaseModel):
    user_id: int = Field(..., gt=0)  # Pydantic validation

@app.get("/users/{user_id}")
async def get_user(request: GetUserRequest) -> UserResponse:
    # No manual validation needed
    # Pydantic already validated request.user_id > 0
    # ...
```

### Rollout

1. **Phase 1**: All API endpoints use Pydantic schemas
2. **Phase 2**: All internal functions use Pydantic models
3. **Phase 3**: Legacy validation code removed

### Success Criteria

- ✅ 100% of API endpoints validated by Pydantic
- ✅ mypy --strict passes on all code
- ✅ No runtime validation errors (caught at schema validation)
- ✅ OpenAPI docs generated automatically

---

## Related Decisions

- ADR-001: Cloud Run orchestration (uses Pydantic for API validation)

---

**Created**: 2026-01-26
**Status**: Production (Active since 2025-11-01)
