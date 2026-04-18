# PHASE 2.2: Middleware Refactoring - Next Action

**Status**: Ready to Execute
**Duration**: 1-1.5 hours
**Expected Result**: 2 more errors fixed (13 → 11)
**Complexity**: MEDIUM

---

## Objective

Split middleware classes from multi-class files into individual modules:

- `ollama/middleware/rate_limit.py` → 4 classes → 4 separate files
- `ollama/middleware/cache.py` → 4 classes → 4 separate files

---

## Files to Create

### From `rate_limit.py` (currently 4 classes):

1. `ollama/middleware/rate_limiter.py` - RateLimiter
2. `ollama/middleware/rate_limit_middleware.py` - RateLimitMiddleware
3. `ollama/middleware/endpoint_rate_limiter.py` - EndpointRateLimiter
4. `ollama/middleware/redis_rate_limiter.py` - RedisRateLimiter

### From `cache.py` (currently 4 classes):

5. `ollama/middleware/cache_key.py` - CacheKey
6. `ollama/middleware/caching_middleware.py` - CachingMiddleware
7. `ollama/middleware/cache_stats.py` - CacheStats
8. (Note: RateLimiter class name collision - verify which file)

---

## Step-by-Step Execution

### Step 1: Audit Current Files

```bash
# Check current structure
head -50 ollama/middleware/rate_limit.py
head -50 ollama/middleware/cache.py

# See what imports them
grep -r "from.*middleware" ollama/ --include="*.py" | head -20
grep -r "import.*rate_limit\|import.*cache" ollama/ --include="*.py" | head -20
```

### Step 2: Create Individual Modules

For each class in rate_limit.py and cache.py:

1. Create new file: `ollama/middleware/{class_name_snake_case}.py`
2. Copy class definition and its docstring
3. Copy required imports for that class
4. Add module docstring explaining the class
5. Verify type hints present (mypy compliance)

### Step 3: Update Middleware **init**.py

Add to `ollama/middleware/__init__.py`:

```python
"""Middleware components for FastAPI application.

Contains cross-cutting concerns like rate limiting, caching, and monitoring.
"""

from ollama.middleware.cache_key import CacheKey
from ollama.middleware.cache_stats import CacheStats
from ollama.middleware.caching_middleware import CachingMiddleware
from ollama.middleware.endpoint_rate_limiter import EndpointRateLimiter
from ollama.middleware.rate_limit_middleware import RateLimitMiddleware
from ollama.middleware.rate_limiter import RateLimiter
from ollama.middleware.redis_rate_limiter import RedisRateLimiter

__all__ = [
    "CacheKey",
    "CacheStats",
    "CachingMiddleware",
    "EndpointRateLimiter",
    "RateLimitMiddleware",
    "RateLimiter",
    "RedisRateLimiter",
]
```

### Step 4: Test Each Module

After creating module, verify import works:

```python
# Quick test in Python REPL
from ollama.middleware.rate_limiter import RateLimiter
# Should NOT raise ImportError
```

### Step 5: Update Imports Throughout Codebase

Find all files that import from middleware:

```bash
grep -r "from ollama.middleware" ollama/ --include="*.py"
grep -r "from .middleware" ollama/ --include="*.py"
```

Update imports to use new module paths if needed.

### Step 6: Run Validation

```bash
# After each module created
python3 scripts/validate-standards.py

# Should see errors decreasing
# Expected: 13 → 12 → 11 (as you split files)
```

### Step 7: Commit

```bash
git add -A
git commit -m "refactor(middleware): split rate limiting and caching to individual modules

Moved 8 middleware classes to separate files:
- RateLimiter, RateLimitMiddleware, EndpointRateLimiter, RedisRateLimiter
- CacheKey, CachingMiddleware, CacheStats, and one name-conflict class

Updated middleware/__init__.py to re-export all classes.
No functional changes, backward compatible.

Validation: 13 errors → 11 errors (2 fixed)"

git push
```

---

## Before You Start

### Verify Current Setup

```bash
# Confirm validation tool works
python3 scripts/validate-standards.py

# Should show 13 errors
```

### Review Existing Pattern

Look at what we already did with auth:

```bash
# Check how we created auth modules
ls -la ollama/auth/
cat ollama/auth/__init__.py

# Copy this pattern for middleware
```

### Understand Dependencies

```bash
# See what imports middleware
grep -r "RateLimiter\|CachingMiddleware" ollama/ --include="*.py" | wc -l

# See where they're used
grep -r "from ollama.middleware import\|from ollama import middleware" ollama/ --include="*.py"
```

---

## Success Criteria

✅ Each class in its own file
✅ Middleware **init**.py has all re-exports
✅ `python3 scripts/validate-standards.py` shows 11 errors
✅ No import errors
✅ All tests still pass
✅ Git commit clean and atomic

---

## If You Get Stuck

### ImportError?

```bash
# Verify file exists
ls ollama/middleware/rate_limiter.py

# Check __init__.py includes it
grep "rate_limiter\|RateLimiter" ollama/middleware/__init__.py

# Try direct import
python3 -c "from ollama.middleware.rate_limiter import RateLimiter; print('OK')"
```

### Class Not Found?

```bash
# Verify class is in the file
grep "class RateLimiter" ollama/middleware/rate_limiter.py

# Check for syntax errors
python3 -m py_compile ollama/middleware/rate_limiter.py
```

### Validation Still Shows 13 Errors?

```bash
# Check __init__.py is correct
python3 scripts/validate-standards.py --verbose

# Look for missing re-exports in middleware/__init__.py
grep "__all__" ollama/middleware/__init__.py
```

---

## Next Phase After This

Once Phase 2.2 is complete (11 errors):

1. **Phase 2.3**: Split services (services/models.py → 5 files, services/ollama_client.py → 3 files)

   - Expected: 11 → 9 errors

2. **Phase 3**: Complex refactoring (models.py, routes/)
   - Expected: 9 → 0 errors

---

## Estimated Time Breakdown

- **Audit & planning**: 10 minutes
- **Create 8 new files**: 20 minutes
- **Update **init**.py**: 5 minutes
- **Test & import verification**: 10 minutes
- **Update imports in codebase**: 10 minutes
- **Run validation**: 5 minutes
- **Commit & push**: 5 minutes

**Total: 65 minutes (~1 hour)**

Ready to execute! 🚀
