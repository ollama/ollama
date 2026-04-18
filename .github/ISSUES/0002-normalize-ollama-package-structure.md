---
title: "Normalize `ollama/` package structure to Landing Zone Level-2 rules"
labels: [refactor, infra, pmo-compliance]
assignees: [kushin77]
---

## Summary

The `ollama/` package currently violates the Landing Zone filesystem rules (15 subdirectories; max 12) and contains Python files at Level 2 that should be Level 4 containers. We must reorganize modules to meet the Level 2/3/4 structure described in the onboarding guide.

## Goals

- Ensure `ollama/` has <=12 subdirectories at Level 2
- Remove Python files at Level 3 (only `__init__.py` allowed)
- Ensure Level 4 containers contain implementation files (Level 5 leaf files)

## Proposed Changes (example)

1. Move `ollama/exceptions.py` → `ollama/exceptions/__init__.py` and convert to package
2. Create missing `__init__.py` files in Level-3 domains (e.g., `api/`, `services/`, `models/`)
3. Group small single-file modules into logical Level-4 containers where needed
4. Run `mypy`, `ruff`, and `pytest` to catch import path or API regressions

## Acceptance criteria

- `scripts/validate_folder_structure.py --strict` reports no `ollama/` subdirectory count error
- `mypy ollama/ --strict` passes (or regresses minimally with planned follow-ups)
- Unit and integration tests pass

## Implementation steps

1. Draft a set of file moves in a branch and run a dry-run to detect import breakage.
2. Prepare PR with automated refactor (use `git mv` to retain history), update imports.
3. Run CI and fix any failing tests.

Notes: This is an invasive change touching package layout — recommend incremental PRs per domain to simplify review.

---
Status: Closed
Resolution: One-shot reorganization applied: root cleanup, `_legacy` grouping, exceptions packaged, `repositories` flattened, and Level‑2 shims converted to packages to meet Landing Zone layout. Remaining action: CI validation and follow-up cleanup if reviewers request. (2026-01-30)
