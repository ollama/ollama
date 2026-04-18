# Documentation Single Source of Truth

This file defines the canonical documentation map for the repository.

## Canonical Documents

- [README.md](../README.md) - project overview and primary entry point
- [docs/operations/ON_PREM_EXECUTION_INDEX.md](../operations/ON_PREM_EXECUTION_INDEX.md) - shared target-server-local operational navigation
- [docs/operations/ON_PREM_DEPLOYMENT_MODEL.md](../operations/ON_PREM_DEPLOYMENT_MODEL.md) - target-server-local execution model
- [docs/operations/ON_PREM_REDEPLOY_RUNBOOK.md](../operations/ON_PREM_REDEPLOY_RUNBOOK.md) - canonical redeploy and rollback procedure
- [docs/shared/README.md](../shared/README.md) - shared navigation layer
- [docs/DEPLOYMENT.md](../DEPLOYMENT.md) - canonical deployment procedures
- [docs/indexed/README.md](../indexed/README.md) - legacy compatibility index hub
- [docs/repo-rules/README.md](../repo-rules/README.md) - canonical repository rules
- [docs/structure/README.md](../structure/README.md) - repository layout and structure reference
- [docs/instructions/README.md](../instructions/README.md) - instruction file registry
- [docs/snc/README.md](../snc/README.md) - standard naming convention
- [docs/reports/HOST_PROFILE_IDEMPOTENCY.md](../reports/HOST_PROFILE_IDEMPOTENCY.md) - rerun evidence and failure taxonomy
- [scripts/host-profile.sh](../../scripts/host-profile.sh) - host inventory loader
- [scripts/host-profile-matrix.sh](../../scripts/host-profile-matrix.sh) - deterministic rerun harness
- [config/hosts/development.env](../../config/hosts/development.env) - development host inventory
- [config/hosts/production.env](../../config/hosts/production.env) - production host inventory

## Rules

- Keep command examples in one canonical guide and link to them from the other docs.
- Keep target-host values in host inventories, not in deployment scripts.
- Keep target-server-local workflow details in the on-prem deployment model, not repeated in every quick reference.
- Keep redeploy and rollback steps in the on-prem redeploy runbook and link to it from indexes.
- Treat compatibility snapshots as references only; do not expand them with new procedures.

## Navigation Model

- **Shared**: [docs/shared/README.md](../shared/README.md)
- **Execution Index**: [docs/operations/ON_PREM_EXECUTION_INDEX.md](../operations/ON_PREM_EXECUTION_INDEX.md)
- **Indexed**: [docs/indexed/README.md](../indexed/README.md)
- **Rules**: [docs/repo-rules/README.md](../repo-rules/README.md)
- **Structure**: [docs/structure/README.md](../structure/README.md)
- **Instructions**: [docs/instructions/README.md](../instructions/README.md)
- **SSOT alias**: [docs/ssot/README.md](../ssot/README.md)
- **Meta**: this file
