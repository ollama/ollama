# Quick Reference - 5-Level Folder Structure Mandate

## ✅ Current Status

- **Validator**: PASSING ✅
- **Pre-commit Hook**: ENABLED ✅
- **Structure**: COMPLIANT ✅

---

## When Adding New Code

### Decision Tree

```
Is this a new domain?
├─ YES → Create Level 3: ollama/{domain}/ + add Level 4 containers
└─ NO
   ├─ Adding to existing domain?
   │  ├─ YES → Create Level 4: ollama/{domain}/{container}/
   │  └─ NO
   │     ├─ Just a utility?
   │     │  └─ Add to existing Level 5 file (don't create new container)
   │     └─ Creating new functional area?
   │        └─ Create Level 4 container in appropriate domain
   └─ Exceeding 500 lines?
      ├─ YES → Split into Level 5 files OR new Level 4 container
      └─ NO → Keep in current Level 5 file
```

---

## Required Checklist

Before committing ANY code:

```bash
# 1. Run validator
python3 scripts/validate_folder_structure.py --strict

# 2. Check this list:
□ No Python files at Level 3 (domain root - only __init__.py)
□ All directories ≤ 5 levels deep
□ All directory names: lowercase_with_underscores
□ All files: lowercase_with_underscores.py
□ All Level 3+ directories have __init__.py with docstring
□ No Level 5 files > 500 lines
□ Max 1 class per Level 5 file
□ All imports organized (stdlib, third-party, local)

# 3. Try to commit (pre-commit hook validates)
git commit -m "feat(scope): description"
```

---

## Examples

### ✅ CORRECT: Adding Inference Feature

```
Create: ollama/services/inference/websocket_handler.py

Why?
- It's a functional container (inference/) in existing domain (services/)
- Level 4 container, Level 5 file
- Hierarchy: ollama → services → inference → websocket_handler.py (4 levels)
```

### ✅ CORRECT: New Schema

```
Create: ollama/api/schemas/streaming_response.py

Why?
- Stays in existing schemas container
- Max 32 files allowed in schemas (special limit)
- Follow naming: lowercase_with_underscores.py
```

### ✅ CORRECT: New Utility

```
Add to: ollama/utils/validators.py

Why?
- Just a helper function
- Use existing Level 5 file
- Don't create new Level 4 container for single function
```

### ❌ WRONG: Too Deep

```
Create: ollama/api/routes/inference/v1/handlers/generate.py

Why?
- 6 levels deep: ollama/api/routes/inference/v1/handlers/generate.py
- MAX is 5 levels
- Fix: Use ollama/api/routes/generate.py instead
```

### ❌ WRONG: File at Level 3

```
Create: ollama/services/cache.py

Why?
- Python file at Level 3 (domain root)
- Only __init__.py allowed at Level 3
- Fix: Create container: ollama/services/cache/cache.py
```

### ❌ WRONG: Bad Naming

```
Create: ollama/Services/Cache.py

Why?
- Not snake_case (has uppercase)
- Inconsistent with codebase
- Fix: Use ollama/services/cache.py
```

---

## Limits Per Level

| Level         | Max Dirs | Max Files | Notes                                                |
| ------------- | -------- | --------- | ---------------------------------------------------- |
| 1 (root)      | 10       | N/A       | Excluded: .git, .venv, cache, logs, etc              |
| 2 (ollama/)   | 12       | 5         | Allowed: main.py, config.py, \_models_compat.py, etc |
| 3 (domain)    | 4        | -         | Only **init**.py allowed                             |
| 4 (container) | -        | 20        | 40 for schemas (special)                             |
| 5 (files)     | 0        | -         | LEAF LEVEL - no subdirs                              |

---

## Commands

### Run Validator

```bash
# Quick check
python3 scripts/validate_folder_structure.py

# Strict mode (what pre-commit uses)
python3 scripts/validate_folder_structure.py --strict

# Verbose output
python3 scripts/validate_folder_structure.py --strict --verbose
```

### Install Pre-commit Hook

```bash
# Make hook executable
chmod +x .githooks/pre-commit

# Configure git to use custom hooks directory
git config core.hooksPath .githooks

# Verify
git config core.hooksPath
# Output: .githooks
```

### Run All Pre-commit Checks Manually

```bash
# Run all checks that pre-commit runs
./.githooks/pre-commit
```

---

## Fixing Violations

### Too Many Files in Container

```bash
# Problem: ollama/api/schemas/ has 40 files

# Solution 1: Increase limit (only for schemas)
# Already done - schemas limit is 40, others are 20

# Solution 2: Create new container
mkdir -p ollama/api/schemas_legacy/
mv ollama/api/schemas/old_*.py ollama/api/schemas_legacy/
```

### File Too Deep

```bash
# Problem: ollama/api/routes/inference/v1/handlers/generate.py (6 levels)

# Solution: Flatten structure
mv ollama/api/routes/inference/v1/handlers/generate.py ollama/api/routes/generate.py

# Or: Use file naming to indicate versions
# ollama/api/routes/generate_v1.py
# ollama/api/routes/generate_v2.py
```

### Missing **init**.py

```bash
# Problem: ollama/services/inference/ has no __init__.py

# Solution: Add with docstring
cat > ollama/services/inference/__init__.py << 'EOF'
"""Inference service module.

Handles AI model inference operations including text generation,
embeddings, and completions.
"""

__all__ = []
EOF
```

---

## Troubleshooting

### Pre-commit Hook Not Running

```bash
# Check hook is executable
ls -l .githooks/pre-commit
# Should show: -rwxr-xr-x (has x flag)

# Make executable if needed
chmod +x .githooks/pre-commit

# Check git config
git config core.hooksPath
# Should show: .githooks
```

### Validator Shows Old Violations

```bash
# Clear cache
rm -rf .pytest_cache __pycache__ .mypy_cache .ruff_cache

# Run validator again
python3 scripts/validate_folder_structure.py --strict
```

### Can't Commit Because of Structure

```bash
# Fix violations first (see "Fixing Violations" section above)

# Once fixed, run validator to confirm
python3 scripts/validate_folder_structure.py --strict

# Should see: ✅ All checks passed!

# Then try commit again
git commit -m "feat: add feature"
```

---

## Reference Links

- **Full Documentation**: `.github/copilot-instructions.md` (search "Folder Structure Enforcement")
- **Validator Script**: `scripts/validate_folder_structure.py`
- **VS Code Settings**: `.vscode/settings.json` (search "FOLDER STRUCTURE")
- **Pre-commit Hook**: `.githooks/pre-commit`
- **Implementation Guide**: `FOLDER_STRUCTURE_ENFORCEMENT_FINAL.md`

---

## Quick Examples By Domain

### API Module (`ollama/api/`)

```
✅ CORRECT:
└── ollama/api/
    ├── routes/
    │   ├── health.py
    │   ├── generate.py
    │   ├── chat.py
    │   └── __init__.py
    ├── schemas/
    │   ├── health_response.py
    │   ├── generate_request.py
    │   └── __init__.py
    ├── dependencies/
    │   ├── auth.py
    │   └── __init__.py
    └── __init__.py
```

### Services Module (`ollama/services/`)

```
✅ CORRECT:
└── ollama/services/
    ├── inference/
    │   ├── ollama_client.py
    │   ├── generate_request.py
    │   └── __init__.py
    ├── models/
    │   ├── model.py
    │   ├── ollama_model_manager.py
    │   └── __init__.py
    ├── cache/
    │   ├── cache.py
    │   └── __init__.py
    ├── persistence/
    │   ├── database.py
    │   ├── chat_message.py
    │   └── __init__.py
    └── __init__.py
```

---

**Last Updated**: January 14, 2026
**Status**: ✅ COMPLIANT
