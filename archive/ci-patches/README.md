# CI Patches

This folder contains ready-to-copy GitHub Actions workflow files for the PMO agent repositories.

Usage (maintainers of `pmo-agent-*` repos):

1. Copy the relevant file from this directory into the agent repository under `.github/workflows/ci.yml`.
2. Commit to a branch and open a PR targeting `main`.
3. Ensure the repository has a `CODECOV_TOKEN` secret if you want coverage uploads.

Files:

- `pmo-agent-remediation-ci.yml`
- `pmo-agent-drift-predictor-ci.yml`
- `pmo-agent-scheduler-ci.yml`
- `pmo-agent-audit-ci.yml`

Notes:

- These workflows run linting (`ruff`), type checking (`mypy`), tests with coverage (`pytest --cov`), and `pip-audit` security scans.
- Adjust `python-version` matrix or dependencies as needed for each repo.
