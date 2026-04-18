# Runbook: GCP Quota Exceeded

**Version**: 1.0 | **Severity**: SEV2 | **Time to Resolution**: 30 min

---

## Detection

- **Alert**: GCP quota exceeded on [resource_type]
- **Symptom**: Deployments fail, new inference requests rejected with 429 errors
- **Dashboard**: [GCP Quotas Console](https://console.cloud.google.com/iam-admin/quotas)

---

## Immediate Actions (0-5 min)

```bash
# Check which quota was exceeded
gcloud compute project-info describe --project=$PROJECT_ID | grep quota

# Identify exceeded quota
# Common ones: Compute instances, GPUs, API requests per minute, Cloud Run concurrent requests
```

---

## Diagnosis (5-15 min)

```bash
# Option A: API rate limit
# Check current usage
gcloud monitoring time-series list \
  --filter='metric.type="serviceruntime.googleapis.com/allocation/consumer/quota_used_count"' \
  --limit=10

# Option B: Compute quota
# Check active instances/GPUs
gcloud compute instances list --filter="zone:us-central1*"
gcloud compute accelerator-types list

# Option C: Cloud Run quota (concurrent requests)
# This is handled automatically but can spike with traffic surge
```

---

## Remediation

### Option A: Request Quota Increase

```bash
# Go to GCP Console → Quotas → Select quota → Edit → Request increase
# Or via gcloud:
gcloud compute project-info \
  --project=$PROJECT_ID \
  update --limit=[new-limit]

# Typical increase: 2x current usage
# Time: Instant to 24 hours depending on quota type
```

### Option B: Reduce Load (Temporary)

```bash
# Reduce concurrent requests to stay under current quota
# 1. Enable Cloud Armor rate limiting
# 2. Reduce agent batch size
# 3. Implement client-side rate limiting

# This buys time for quota increase to be approved
```

---

## Escalation

- If quota increase not approved within 4 hours: Escalate to @cto
- May require architectural changes (reduce batch size, distribute across regions)

**Created**: 2026-01-26
