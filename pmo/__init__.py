"""PMO compatibility package.

This package provides a minimal shim so the repository satisfies the
Landing Zone folder-structure validator which expects `pmo/` to be a
Python package. It is intentionally minimal and contains no runtime
secrets or behavior.

Do not import secret values from here. Secrets remain stored in
`pmo.yaml` (metadata) and in secret stores (GCP Secret Manager,
GitHub Actions secrets) as appropriate.
"""

__all__ = []
