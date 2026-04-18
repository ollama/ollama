# CI Guide

This document explains how the GitHub Actions CI templates work and how to reproduce CI locally.

## Files

- `.github/workflows/pmo-ci.yml` — PMO CI workflow for main repo.
- `.github/ci-templates/` — Templates for agent repositories. Copy the appropriate template into the agent repo at `.github/workflows/ci.yml`.

## Secrets

- `CODECOV_TOKEN` — required to upload coverage to Codecov. Add this as a repository secret in each repo where Codecov uploads are enabled.

## Running CI locally

Install dependencies and dev tooling:

```bash
python -m pip install -U pip
pip install -e .[dev]
pip install pytest mypy ruff pip-audit
```

Run linting, type checking, and tests with coverage:

```bash
ruff check .
mypy . --strict
pytest --cov=./ --cov-report=xml:coverage.xml tests/
```

## Uploading coverage

Codecov upload requires a token in repository secrets. The CI workflow uses `codecov/codecov-action@v4` and expects `CODECOV_TOKEN` to be defined.

## Security scanning

The workflow runs `pip-audit --fail-on high`. Adjust the policy if you want to tolerate `moderate` issues for non-production branches.

## Troubleshooting

- If `mypy` fails, fix the type errors or narrow the `mypy` target to the package(s) that are relevant.
- If `pytest` fails, run the failing test locally with `-k` to reproduce.
- If Codecov upload fails, ensure the `CODECOV_TOKEN` secret is present and has correct permissions.
