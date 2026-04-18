---
title: "Reduce top-level directories to <=10 (Landing Zone compliance)"
labels: [infra, cleanup, pmo-compliance]
assignees: [kushin77]
---

## Summary

The repository currently has too many top-level directories (21). The GCP Landing Zone onboarding mandates a maximum of 10 top-level directories at the repository root. This task consolidates and archives files to meet that requirement.

## Goals

- Reduce number of top-level entries to 10 or fewer
- Move archival or large documentation into `docs/archive/` or `docs/` subfolders
- Preserve links and references; add README redirects where needed

## Proposed Changes (example plan)

1. Create `docs/archive/` and move historical and completion reports there:
   - `FINAL_*`, `SESSION_*`, `PHASE_*`, `WEEK_*`, `FINAL_VERIFICATION_REPORT.md`, etc.
2. Move deployment-related assets into `docker/`, `k8s/`, or `config/` already present.
3. Keep top-level mandatory files only: `pyproject.toml`, `README.md`, `pmo.yaml`, `docs/`, `ollama/`, `scripts/`, `docker/`, `k8s/`, `.github/` (adjust as needed to reach <=10)
4. Add `docs/archive/README.md` with list of moved files and reasons for archive.

## Acceptance criteria

- `scripts/validate_folder_structure.py --strict` reports no root-level directory count error
- All moved files maintain working links from `docs/` or other references (update relative links)
- CI runs and tests pass (no broken imports or paths introduced by the move)

## Risks & Mitigations

- Risk: CI or scripts referencing absolute paths could break. Mitigation: update CI and add compatibility README with search-and-replace guidance.
- Risk: External references (docs or issue templates). Mitigation: create redirect README files at previous paths for one release cycle.

## Implementation steps

1. Create `docs/archive/` and move listed files.
2. Update relative links in moved files to new locations.
3. Add `docs/archive/README.md` documenting moves.
4. Run validators and tests; fix any failing references.

Please review plan and approve before I prepare a PR with the changes.

---
Status: Closed
Resolution: Root-directory cleanup executed; low-impact directories moved into `archive/`, `SECURITY` consolidated into `docs/`, and `docs/` populated. Remaining work: normalize `ollama/` package (staged). (2026-01-30)
