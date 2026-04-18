# GCS Backup Setup Guide

This guide configures Google Cloud Storage (GCS) for automated backups of models, databases, and snapshots from the Ollama stack.

## Prerequisites
- GCP project with billing enabled
- `gcloud` CLI installed and authenticated (`gcloud auth login`)
- Project ID (e.g., `elevatediq-ollama`)
- Organization permissions to create service accounts and roles

## 1) Enable Required APIs
```bash
gcloud services enable storage.googleapis.com
```

## 2) Create GCS Bucket
- Bucket name: `elevatediq-ollama-backups`
- Location: `us-central1` (or your preferred region)
- Storage class: `STANDARD`
- Uniform bucket-level access: **Enabled**
- Versioning: **Enabled** (keeps previous object versions)

```bash
gsutil mb -p <PROJECT_ID> -c STANDARD -l us-central1 gs://elevatediq-ollama-backups
# Enable versioning
gsutil versioning set on gs://elevatediq-ollama-backups
```

## 3) Create Service Account for Backups
```bash
gcloud iam service-accounts create ollama-backup \
  --description="Service account for Ollama backups" \
  --display-name="ollama-backup"
```

## 4) Assign Roles (Least Privilege)
Grant only storage access to the bucket.
```bash
gcloud projects add-iam-policy-binding <PROJECT_ID> \
  --member="serviceAccount:ollama-backup@<PROJECT_ID>.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"
```

Optional: if you use lifecycle rules via script, also grant `roles/storage.admin` (or attach a lifecycle JSON to the bucket instead).

## 5) Create Service Account Key
```bash
gcloud iam service-accounts keys create gcp-service-account.json \
  --iam-account=ollama-backup@<PROJECT_ID>.iam.gserviceaccount.com
```
Move this file into `./secrets/gcp-service-account.json` and ensure permissions:
```bash
mkdir -p secrets
mv gcp-service-account.json secrets/
chmod 600 secrets/gcp-service-account.json
```

## 6) Configure Bucket Lifecycle (Retention)
Keep 30 daily backups, 14 Redis snapshots, 7 Qdrant snapshots by date prefix.
Example rule file `lifecycle.json`:
```json
{
  "rule": [
    {
      "action": {"type": "Delete"},
      "condition": {"age": 40, "matchesPrefix": ["postgres/"]}
    },
    {
      "action": {"type": "Delete"},
      "condition": {"age": 20, "matchesPrefix": ["redis/"]}
    },
    {
      "action": {"type": "Delete"},
      "condition": {"age": 14, "matchesPrefix": ["qdrant/"]}
    }
  ]
}
```
Apply:
```bash
gsutil lifecycle set lifecycle.json gs://elevatediq-ollama-backups
```

## 7) Update Environment
Set these in `.env.production`:
```
GCS_BUCKET=gs://elevatediq-ollama-backups
GCS_SYNC_INTERVAL=3600
GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/gcp_sa_key
```

## 8) Wire into docker-compose.elite.yml
- `gcs-sync` service mounts `./secrets/gcp-service-account.json` to `/run/secrets/gcp_sa_key`.
- Ensure volume paths exist:
  - `/mnt/data/ollama/models`
  - `/mnt/backups/postgres`
  - `/mnt/backups/qdrant`
  - Redis data volume (`redis-data`)

## 9) Run Initial Sync
```bash
docker compose -f docker-compose.elite.yml up -d gcs-sync
```
Check logs:
```bash
docker logs -f ollama-gcs-sync
```

## 10) Validate Backups
```bash
gsutil ls gs://elevatediq-ollama-backups/models
gsutil ls gs://elevatediq-ollama-backups/postgres
gsutil ls gs://elevatediq-ollama-backups/redis
gsutil ls gs://elevatediq-ollama-backups/qdrant
```

## 11) (Optional) VPC Egress Controls
If using a private GCE host, restrict egress to only Storage endpoints via VPC-SC or firewall.

## 12) (Optional) Customer-Managed Encryption Keys (CMEK)
If you require CMEK, create a KMS key and set bucket default KMS key:
```bash
gcloud kms keyrings create ollama-backups --location=us-central1
gcloud kms keys create gcs-cmek --location=us-central1 --keyring=ollama-backups --purpose=encryption

gsutil kms enable -k projects/<PROJECT_ID>/locations/us-central1/keyRings/ollama-backups/cryptoKeys/gcs-cmek \
  gs://elevatediq-ollama-backups
```

## 13) Restore Procedures (Pointers)
- Postgres: use `scripts/restore-postgres.sh /mnt/backups/postgres/<file>.sql.gz`
- Qdrant: download snapshot from GCS, place into `/mnt/backups/qdrant`, and restore via Qdrant API
- Redis: download `dump.rdb` and/or `appendonly.aof` into `redis-data` and restart Redis

---
**Done.** Your GCS backup pipeline is now ready for production use.
