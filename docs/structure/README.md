# Repository Structure

This is the canonical layout reference for the repository.

## Top-Level Layout

```text
.
├── README.md                      # Primary entry point
├── docs/                          # Canonical documentation, indexes, rules, structure
├── docs/shared/                  # Shared navigation layer
├── docs/indexed/                 # Legacy compatibility index hub
├── docs/meta/                    # Documentation SSOT and ownership
├── docs/deep/                    # Deep scans and long-form evidence landing page
├── docs/roadmaps/                # Planning artifacts and enhancement roadmaps
├── docs/structure/               # Repository layout and structure reference
├── docs/repo-rules/              # Repository rules and guardrails
├── docs/instructions/             # Registry of .github instruction files
├── docs/ssot/                    # SSOT compatibility alias
├── docs/snc/                     # Standard naming convention
├── docs/operations/              # Target-server-local operational navigation and legacy snapshots
├── docs/reports/                 # Long-lived evidence and report artifacts
├── scripts/                       # Repo-local automation and host helpers
├── config/hosts/                  # Checked-in host inventories
├── docker/                        # Compose manifests and container config
├── ollama/                        # Python application package
├── cmd/                           # Go CLI entry points
├── server/                        # Go HTTP server and request handling
├── tests/                         # Test suite
├── k8s/                           # Kubernetes assets
├── kubernetes/                    # Additional Kubernetes assets
├── terraform/                     # IaC
└── archive/                       # Historical compatibility material
```

## Placement Rules

- Put new canonical documentation in a dedicated docs subfolder instead of the root of `docs/` when the topic has a clear home.
- Put shared navigation in `docs/shared/`.
- Put compatibility indexes in `docs/indexed/`.
- Put ownership and canonical-doc rules in `docs/meta/`.
- Put deep scans and long-form evidence in `docs/deep/`.
- Put planning artifacts and enhancement roadmaps in `docs/roadmaps/`.
- Put repository layout guidance in `docs/structure/`.
- Put repository rules in `docs/repo-rules/`.
- Put instruction registries in `docs/instructions/`.
- Put SSOT aliases in `docs/ssot/`.
- Put naming conventions in `docs/snc/`.
- Put pnpm-specific documentation in `docs/pnpm/` only if a real pnpm workspace exists in the active repository; otherwise keep pnpm references in legacy or archived material.
- Put target-server-local navigation in `docs/operations/`.
- Put long-lived reports in `docs/reports/`.
- Put host inventories in `config/hosts/`.
- Put repo-local automation in `scripts/`.
- Put infrastructure definitions in `terraform/` or `docker/` as appropriate.
- Avoid adding new loose root-level docs when an existing docs subfolder fits the content.

## Python Package Layout

```text
ollama/
├── __init__.py
├── __version__.py
├── api/
├── inference/
├── models/
├── embeddings/
├── rag/
├── database/
├── cache/
├── monitoring/
├── security/
├── utils/
├── cli/
└── server.py
```

## Package Organization Principles

- Single responsibility per module.
- Clear separation of concerns.
- Easy to test and extend.
- Type-safe with full annotations.
