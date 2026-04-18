---
title: "Add PMO package and missing `__init__.py` (compatibility shim)"
labels: [infra, pmo-compliance]
assignees: [kushin77]
---

## Summary

Validator reports a missing `pmo/` package `__init__.py`. There are PMO scripts under `scripts/pmo/` and a top-level `pmo.yaml`. Add a thin `pmo/` package or compatibility `__init__.py` to satisfy validators and make PMO modules importable.

## Goals

- Create `pmo/__init__.py` with docstring and minimal exports
- Avoid changing semantics; keep file non-invasive

## Proposed `__init__.py` content

```py
"""PMO compatibility package.

Small shim to satisfy folder-structure validator and provide common PMO utilities.
"""

# Expose top-level helpers if any in scripts/pmo
__all__ = []
```

## Acceptance criteria

- `scripts/validate_folder_structure.py --strict` no longer reports missing `pmo/` package error

Please confirm if the package should be added at repository root (`/pmo`) or under `ollama/` (`ollama/pmo`).

---
Status: Closed
Resolution: `pmo/__init__.py` and `ollama/pmo/__init__.py` compatibility shims added; PMO metadata validation passes. Remaining PMO consolidation is planned as part of the staged migration PRs. (2026-01-30)
