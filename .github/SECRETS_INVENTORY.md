# Secrets Inventory (Sanitized)

This file lists secret names, owners, and recommended storage locations.
It contains no secret values — store actual secrets in GCP Secret Manager
for production and GitHub Actions secrets for CI/CD pipelines.

Usage: maintain this file as the canonical inventory of secret identifiers
used by the repository and reference it from the PMO documentation.

## Recommended format

| Secret Name | Purpose | Owner | Recommended Storage | Notes |
|-------------|---------|-------|---------------------|-------|
| `GCP_SA_KEY` | GCP service account JSON | infra@elevatediq.ai | GCP Secret Manager (project-level) | CI: GitHub Actions secret `GCP_SA_KEY` (base64 JSON) |
| `GCP_PROJECT_ID` | GCP project id | infra@elevatediq.ai | GitHub Actions secret | non-sensitive but treated as secret in workflows |
| `NEXT_PUBLIC_FIREBASE_API_KEY` | Frontend Firebase key | frontend@elevatediq.ai | GitHub Actions secret | public key for client use; treat as env var for prod builds |
| `OPENAI_API_KEY` | External model API key (example) | ml@elevatediq.ai | GCP Secret Manager | rotate regularly |
| `POSTGRES_PASSWORD` | DB password | db-admin@elevatediq.ai | GCP Secret Manager | Mount into runtime via Workload Identity |
| `DATABASE_URL` | DB connection string (no creds) | db-admin@elevatediq.ai | Build from secrets at runtime | Store credentials in Secret Manager and assemble at startup |
| `REDIS_PASSWORD` | Redis auth | infra@elevatediq.ai | GCP Secret Manager | rotate regularly |
| `OLLAMA_API_KEY` | Ollama model server API key | ml@elevatediq.ai | GCP Secret Manager | per-environment secret
| `API_KEY_SEED` | Application API key seed | security@elevatediq.ai | GCP Secret Manager | used to derive API keys, rotate frequently

## Notes and Guidance

- Never store actual secret values in the repository. Use `pmo.yaml` only
  for metadata, not secret values.
- Use Workload Identity and GCP Secret Manager for production runtime secrets.
- For GitHub Actions, add repository or organization secrets (Actions → Secrets).
- Add a rotation schedule and owner for each secret in the `Notes` column.
