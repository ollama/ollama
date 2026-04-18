# FAANG Audit Remediation: Immediate Action Items

**Status**: BLOCKING production commit
**Priority**: 🔴 CRITICAL
**Target**: Fix all Phase 1 violations within 2 hours

---

## Phase 1: BLOCKING ISSUES (Fix NOW)

### Issue #1: Fix Test Infrastructure (26 test errors)

**Affected files**:

- `tests/integration/test_api_smoke.py` (14 errors)
- `tests/unit/test_auth.py` (12 errors)

**Root cause**: Import/attribute errors from test fixtures after recent refactoring

**Action plan**:

1. Diagnose first error: `pytest tests/unit/test_auth.py::TestFirebaseAuth::test_hash_password -v`
2. Review fixture definitions in `tests/fixtures/`
3. Fix broken imports in test files
4. Re-run: Verify all 26 errors eliminated

**Expected time**: 30 minutes

---

### Issue #2: Fix Test Failures (9 failing tests)

**Affected tests**:

- `tests/unit/test_metrics.py` (5 failures)
- `tests/unit/test_ollama_client.py` (3 failures)
- (1 more)

**Root cause**: Mock assertions not matching actual behavior after service refactoring

**Action plan**:

1. Run each test individually with `-vvv` flag
2. Compare expected vs actual in mocks
3. Update mock/assertion setup
4. Verify pass

**Expected time**: 30 minutes

---

### Issue #3: Upgrade transformers Dependency (16 CVEs)

**Current**: `transformers==4.35.2` (16 known CVEs)
**Target**: `transformers>=4.53.0` (all CVEs fixed)

**Commands**:

```bash
# Option A: Direct edit
sed -i 's/"transformers>=4.35.2"/"transformers>=4.53.0"/' pyproject.toml

# Option B: Use pip
cd /home/akushnir/ollama
/home/akushnir/ollama/venv/bin/pip install --upgrade transformers

# Verify
/home/akushnir/ollama/venv/bin/pip-audit | grep transformers
```

**Expected time**: 10 minutes (includes verification)

---

### Issue #4: Fix Linting Errors (15 violations)

#### Part A: Fix Exception Chaining (9 violations of B904)

**Files affected**:

- `ollama/api/routes/inference.py` (7 violations at lines 57, 83, 151, 154, 180, 205, 230, 286)
- `ollama/auth/firebase_auth.py` (2 violations at lines 62, 252, 256)

**Pattern to fix**:

```python
# ❌ BEFORE
except SomeException as e:
    raise HTTPException(detail="...")

# ✅ AFTER
except SomeException as e:
    raise HTTPException(detail="...") from e
```

**Commands**:

```bash
# Review file
cat /home/akushnir/ollama/ollama/api/routes/inference.py | grep -n "raise"

# Apply fixes manually or with script
```

**Expected time**: 20 minutes

#### Part B: Fix Import Sorting (1 violation in inference.py:7)

**Command**:

```bash
/home/akushnir/ollama/venv/bin/python -m ruff check ollama/api/routes/inference.py --fix
```

**Expected time**: 2 minutes

#### Part C: Reduce Complexity (3 functions > complexity 10)

**Functions**:

1. `ollama/main.py:77` - `lifespan()` (complexity 11)
2. `ollama/auth/firebase_auth.py:65` - `get_current_user()` (complexity 13)
3. `ollama/api/server.py:25` - `create_app()` (complexity 12)

**Strategy**: Extract helper functions

**Example**:

```python
# ❌ BEFORE (complexity 13)
async def get_current_user(token: str) -> User:
    if not token:
        raise ...
    try:
        payload = decode_token(token)
    except:
        raise ...
    if not payload.get("sub"):
        raise ...
    user = get_user(payload["sub"])
    if not user:
        raise ...
    if user.disabled:
        raise ...
    if user.role not in allowed_roles:
        raise ...
    ...

# ✅ AFTER (complexity 5, extracted helpers at complexity 2-3)
def _validate_token_structure(token: str) -> dict:
    """Validate and decode token."""
    if not token:
        raise InvalidTokenError()
    payload = decode_token(token)
    if not payload.get("sub"):
        raise InvalidTokenError()
    return payload

def _validate_user(user: User) -> None:
    """Validate user status and permissions."""
    if not user:
        raise UserNotFoundError()
    if user.disabled:
        raise UserDisabledError()
    if user.role not in ALLOWED_ROLES:
        raise InsufficientPermissionsError()

async def get_current_user(token: str) -> User:
    payload = _validate_token_structure(token)
    user = get_user(payload["sub"])
    _validate_user(user)
    return user
```

**Expected time**: 30 minutes

---

### Summary: Phase 1 Timeline

| Task                                  | Time      | Status  |
| ------------------------------------- | --------- | ------- |
| Fix test errors (26)                  | 30 min    | ⏳ TODO |
| Fix test failures (9)                 | 30 min    | ⏳ TODO |
| Upgrade transformers                  | 10 min    | ⏳ TODO |
| Fix linting (import sort)             | 2 min     | ⏳ TODO |
| Fix exception chaining (9 violations) | 20 min    | ⏳ TODO |
| Reduce complexity (3 functions)       | 30 min    | ⏳ TODO |
| **TOTAL**                             | **2 hrs** | ⏳ TODO |

---

## Phase 2: COVERAGE CLOSURE (After Phase 1)

**Target**: 39.71% → 90%+ coverage

### Priority 1: CRITICAL COVERAGE (0%)

#### `ollama/services/ollama_model_manager.py` - 53 lines (0% coverage)

**Missing tests for**:

- Model manager initialization
- Model loading/caching
- Error handling

**Test template**:

```python
# tests/unit/test_ollama_model_manager.py
class TestOllamaModelManager:
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Manager initializes correctly."""
        manager = OllamaModelManager(base_url="http://ollama:11434")
        await manager.initialize()
        assert manager is not None

    @pytest.mark.asyncio
    async def test_load_model(self):
        """Model loads and caches correctly."""
        manager = OllamaModelManager(...)
        model = await manager.load_model("llama2")
        assert model is not None

    @pytest.mark.asyncio
    async def test_load_model_not_found(self):
        """Non-existent model raises error."""
        manager = OllamaModelManager(...)
        with pytest.raises(ModelNotFoundError):
            await manager.load_model("nonexistent")
```

**Expected lines to cover**: 53
**Estimated effort**: 1 hour

---

### Priority 2: HIGH COVERAGE GAPS

#### `ollama/services/cache.py` - 78 lines (19.4% coverage)

**Missing tests for**:

- Cache hits/misses
- TTL expiration
- LRU eviction
- Concurrent access

**Target lines**: 60+
**Estimated effort**: 45 minutes

#### `ollama/services/vector.py` - 57 lines (24.5% coverage)

**Missing tests for**:

- Vector search
- Batch operations
- Embedding generation
- Error handling

**Target lines**: 45+
**Estimated effort**: 45 minutes

---

### Priority 3: MEDIUM COVERAGE GAPS

| File                                 | Lines | Current | Target | Effort |
| ------------------------------------ | ----- | ------- | ------ | ------ |
| `repositories/usage_repository.py`   | 52    | 23.8%   | 95%+   | 40 min |
| `repositories/message_repository.py` | 29    | 39.2%   | 95%+   | 30 min |
| `services/database.py`               | 31    | 28.9%   | 95%+   | 25 min |
| `services/ollama_client_main.py`     | 48    | 34.2%   | 95%+   | 40 min |

---

### Phase 2 Timeline

| Task                                      | Effort         | Status  |
| ----------------------------------------- | -------------- | ------- |
| Test `ollama_model_manager.py` (0% → 95%) | 1 hour         | ⏳ TODO |
| Test `cache.py` (19.4% → 95%)             | 45 min         | ⏳ TODO |
| Test `vector.py` (24.5% → 95%)            | 45 min         | ⏳ TODO |
| Test medium-gap files (4 files)           | 2 hours        | ⏳ TODO |
| **TOTAL**                                 | **~4.5 hours** | ⏳ TODO |

**Expected final coverage**: 90%+

---

## Phase 3: DOCUMENTATION FIXES (After Phase 1-2 complete)

**Scope**: Fix 60+ markdown linting errors

### Issues to fix:

1. **MD040** (15 errors): Add language tags to code blocks

   - Find: `\`\`\`` → Replace: ` ``` bash ` (with language)

2. **MD036** (8 errors): Use headings, not emphasis

   - Find: `**Heading**` → Replace: `### Heading`

3. **MD034** (8 errors): Wrap bare URLs in links

   - Find: `https://example.com` → Replace: `[Link](https://example.com)`

4. **MD051** (2 errors): Fix invalid link fragments
5. **MD024** (2 errors): Rename duplicate headings
6. **MD031** (1 error): Add blank line before code fence

### Automated fix:

```bash
# Install markdownlint-cli (if not present)
npm install -g markdownlint-cli

# Auto-fix (limited options)
markdownlint --fix *.md
```

**Expected time**: 30 minutes

---

## FULL EXECUTION CHECKLIST

```bash
#!/bin/bash
set -e  # Exit on first error

echo "🔴 PHASE 1: BLOCKING ISSUES"

echo "📋 Step 1: Diagnose test errors"
cd /home/akushnir/ollama
/home/akushnir/ollama/venv/bin/pytest tests/unit/test_auth.py::TestFirebaseAuth::test_hash_password -v
# TODO: Fix import errors

echo "📋 Step 2: Fix test failures"
/home/akushnir/ollama/venv/bin/pytest tests/unit/test_metrics.py -v
# TODO: Update mocks

echo "📋 Step 3: Upgrade transformers"
sed -i 's/"transformers>=4.35.2"/"transformers>=4.53.0"/' pyproject.toml
/home/akushnir/ollama/venv/bin/pip install --upgrade transformers

echo "📋 Step 4: Fix linting errors"
# Fix imports
/home/akushnir/ollama/venv/bin/python -m ruff check ollama/api/routes/inference.py --fix

# Fix exception chaining (manual edits to files listed above)
# Then fix complexity (manual refactoring)

echo "✅ VERIFY PHASE 1"
/home/akushnir/ollama/venv/bin/python -m ruff check ollama/
/home/akushnir/ollama/venv/bin/pip-audit
/home/akushnir/ollama/venv/bin/pytest tests/ -x
echo "✅ All Phase 1 checks passed!"

echo "🟡 PHASE 2: COVERAGE CLOSURE"
# Add unit tests for critical paths
# Expected: coverage 39.71% → 90%+

echo "🟡 PHASE 3: DOCUMENTATION"
# Fix markdown linting errors

echo "✅ READY FOR COMMIT"
git add .
git commit -S -m "fix(quality): resolve critical violations from FAANG audit

- Fix 26 test infrastructure errors in integration/unit tests
- Fix 9 test failures in metrics and ollama_client modules
- Upgrade transformers from 4.35.2 to 4.53.0 (fixes 16 CVEs)
- Fix 15 linting violations (9 B904 exception chaining, 3 complexity, 1 import)
- Reduce cognitive complexity in 3 functions (lifespan, get_current_user, create_app)

Test Coverage: 39.71% → [Phase 2 target: 90%+]
Linting: 15 errors → 0
Security: 16 vulnerabilities → 0
All FAANG mandate compliance checks now passing.

See: FAANG_BRUTALITY_AUDIT_JAN14_2025.md for full audit."
git push origin main
```

---

## NO BULLSHIT SUMMARY

### Status: **🔴 NOT READY FOR PRODUCTION**

**Blocking**:

- ❌ 26 test errors (fixture/import issues)
- ❌ 9 test failures (mock/logic issues)
- ❌ 16 security vulnerabilities (transformers)
- ❌ 15 linting violations (code quality)

### What needs to happen:

1. **NOW** (2 hours): Fix Phase 1 blocking issues
2. **THEN** (4.5 hours): Increase test coverage to 90%+
3. **THEN** (30 min): Fix documentation
4. **THEN**: **READY FOR PRODUCTION COMMIT**

### Git state:

- ✅ Commits are atomic and well-formatted
- ✅ Branch naming is correct
- ⚠️ Working directory has 55 modified files + 21 untracked files
- ⚠️ Need to stage and commit all changes

### Next immediate step:

```bash
cd /home/akushnir/ollama

# Run the full checks to see current state
/home/akushnir/ollama/venv/bin/python -m pytest tests/ --tb=short -q
/home/akushnir/ollama/venv/bin/python -m ruff check ollama/
/home/akushnir/ollama/venv/bin/pip-audit
```

---

**This document is the remediation roadmap. Follow Phase 1 first before moving to Phase 2.**
