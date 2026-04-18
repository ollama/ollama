# Repository Rules

This document is the canonical rule set for working in this repository.

## Core Rules

- Keep deployment scripts repo-relative and target-server-local.
- Keep host-specific values in `config/hosts/*.env` or a checked-in inventory file.
- Keep runtime evidence ephemeral unless it must be persisted as declared state.
- Treat immutable, checked-in configuration as the source of truth.
- Prefer shared loaders and canonical indexes over repeating the same workflow in multiple docs.
- Use the host-profile matrix and idempotency report when validating reruns on development or production targets.

## Documentation Rules

- [docs/meta/README.md](../meta/README.md) defines ownership and canonical documents.
- [docs/shared/README.md](../shared/README.md) is the shared documentation navigation layer.
- [docs/operations/ON_PREM_DEPLOYMENT_MODEL.md](../operations/ON_PREM_DEPLOYMENT_MODEL.md) is the canonical target-server-local execution model and host-profile contract.
- [docs/deep/README.md](../deep/README.md) is the canonical long-form evidence and deep-scan index.
- [docs/roadmaps/README.md](../roadmaps/README.md) is the canonical planning and enhancement roadmap index.
- [docs/indexed/README.md](../indexed/README.md) is the legacy compatibility index hub.
- [docs/ssot/README.md](../ssot/README.md) is the SSOT compatibility alias.

## Operational Rules

- Run repo-local scripts from the target server checkout when validating development or production hosts.
- Keep deployment and troubleshooting steps in the canonical guides, then link to them from quick references.
- Keep cloud-only or historical material labeled as legacy if it remains in the repository.
- Keep deep-scan and long-form evidence in `docs/deep/` instead of scattering it across unrelated report lists.
- Keep roadmap proposals in `docs/roadmaps/` instead of scattering them across reports or deep-dive artifacts.
- Avoid adding new canonical content to compatibility snapshots.

## Standard Naming Convention

- Use lower-case kebab-case for new documentation folders.
- Use `README.md` as the landing page for a canonical doc folder.
- Use purpose-specific filenames only when the file is the sole canonical source for that topic.
- Keep compatibility snapshots at their existing paths until callers are migrated.
- Keep host inventory filenames descriptive and environment-based.
- Do not create a pnpm docs bucket unless the active repository contains a real pnpm workspace file.
