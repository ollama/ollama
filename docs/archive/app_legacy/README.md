# Legacy App Directory - ARCHIVED

**Archived**: January 13, 2026
**Status**: ARCHIVED (No longer part of active codebase)
**Reason**: Experimental code not integrated into main `ollama/` package

---

## What's Here

This directory contains experimental prototype code from the legacy `app/` directory that was located at the root of the repository. The code was not integrated into the main `ollama/` package and contained imports from non-existent modules.

### Files Included

The following Python files were archived:

- `batch.py` - Batch processing experiments
- `finetune.py` - Model fine-tuning prototypes
- `streaming.py` - Streaming response experiments
- `performance.py` - Performance testing code

---

## Status

**Integration**: ❌ Not integrated with main codebase
**Dependencies**: ⚠️ Imports from non-existent modules (`app.core`, `app.schemas`)
**Tests**: ❌ No corresponding tests
**Documentation**: ❌ Not documented in main API
**Maintenance**: ⏸️ No recent updates

---

## If You Need This Code

If this experimental code needs to be revived:

1. **Review the code quality**:
   ```bash
   # Inspect files for any usable patterns
   cat batch.py
   cat finetune.py
   cat streaming.py
   cat performance.py
   ```

2. **Understand the intent**:
   - What problem was it trying to solve?
   - Is this functionality now elsewhere?
   - Are there better approaches now?

3. **Migrate patterns to `ollama/` package**:
   - Move implementations to appropriate `ollama/services/` or `ollama/api/` subdirectories
   - Update imports to use `ollama.` prefix
   - Fix any dependency issues
   - Add proper type hints

4. **Add proper tests and documentation**:
   - Create corresponding test files in `tests/unit/` and `tests/integration/`
   - Update API documentation in `PUBLIC_API.md`
   - Add examples to docstrings

5. **Re-integrate with main codebase**:
   - Create feature branch: `git checkout -b feature/revive-legacy-app`
   - Make necessary changes
   - Create PR for review
   - Ensure all tests pass and coverage maintained

---

## Archival Decision

**Reason for Archival**:
- Code was not actively used in production
- Imports referenced non-existent modules
- No integration with main application
- No associated tests or documentation
- Better to preserve history and clean up codebase

**Decision Made**: January 13, 2026
**Reference**: [INCOMPLETE_TASKS_CONSOLIDATED.md#task-3-decide-on-legacy-app-directory](../../INCOMPLETE_TASKS_CONSOLIDATED.md)

---

## Related Documentation

- [COPILOT_COMPLIANCE_REPORT.md](../../../COPILOT_COMPLIANCE_REPORT.md) - Detailed compliance audit
- [DEEP_SCAN_COMPLETION_SUMMARY.md](../../../DEEP_SCAN_COMPLETION_SUMMARY.md) - Folder structure verification
- [Ollama Architecture](../../../docs/architecture.md) - Current system design

---

## Recovery

If you need to recover these files:

```bash
# Check git history
git log --oneline -- app/

# Restore a specific commit
git show <commit-hash>:app/batch.py

# Or restore all from specific commit
git checkout <commit-hash> -- app/
```

---

**Questions?** See [INCOMPLETE_TASKS_CONSOLIDATED.md](../../INCOMPLETE_TASKS_CONSOLIDATED.md) for task context.
