# Runbook: Database Connection Pool Exhausted

**Version**: 1.0 | **Severity**: SEV1 | **Time to Resolution**: 10 min

---

## Detection

- **Alert**: `postgresql_connections > 95`
- **Symptom**: All database queries timeout, API returns 503 Service Unavailable
- **Dashboard**: [Cloud SQL Monitoring](https://console.cloud.google.com/sql)

---

## Immediate Actions (0-3 min)

```bash
# Create incident channel
#incident-db-pool-[timestamp]

# Check current connections
psql $PROD_DB -c "SELECT count(*) FROM pg_stat_activity;"

# If > 95: Confirm pool exhaustion
```

---

## Diagnosis & Fix (3-10 min)

```bash
# Step 1: Find long-running queries
psql $PROD_DB -c "SELECT pid, usename, state, query_start FROM pg_stat_activity WHERE state='active' AND query_start < now() - interval '30 seconds' ORDER BY query_start;"

# Step 2: Kill long-running transactions
psql $PROD_DB -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state='active' AND query_start < now() - interval '30 seconds';"

# Step 3: Verify connection recovery
psql $PROD_DB -c "SELECT count(*) FROM pg_stat_activity;"
# Should drop to < 50

# Step 4: If still high, restart FastAPI service
gcloud run services update-traffic ollama-api --to-revisions LATEST=100 --region=us-central1
```

---

## Prevention

- Implement connection pool limits in SQLAlchemy
- Add alerting at 80% capacity
- Regular query optimization reviews

**Created**: 2026-01-26
