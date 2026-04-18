# Runbook: Performance Degradation (>20% Latency Increase)

**Version**: 1.0 | **Severity**: SEV2 | **Time to Resolution**: 20 min

---

## Detection

- **Alert**: `ollama_api_latency_p95_ms > [baseline * 1.2]`
- **Symptom**: API responses taking 2-3x longer than normal
- **Dashboard**: [Performance Dashboard](https://grafana.example.com/d/performance)

---

## Immediate Actions (0-3 min)

```bash
# Create: #incident-perf-[timestamp]

# Confirm degradation
gcloud monitoring time-series list \
  --filter='metric.type="custom.googleapis.com/ollama_api_latency_p95_ms"' \
  --format="table(points[0].value.double_value)"

# Compare to baseline from yesterday at this time
# If 20%+ higher: Confirmed performance degradation
```

---

## Diagnosis (3-12 min)

```bash
# Check 1: Is it a spike or gradual?
# From Grafana: Look at 24h view
# Spike = recent deployment issue
# Gradual = traffic growth or memory leak

# Check 2: Which endpoint is slow?
gcloud logging read 'severity=WARNING' --limit=50 | grep "slow query"

# Check 3: Database query performance
psql $PROD_DB -c "SELECT query, calls, mean_time FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"

# Check 4: Check Cloud Run metrics
gcloud run services describe ollama-api --region=us-central1 | grep cpu

# Check 5: Check GCP resources
# CPU usage, memory usage, disk I/O
```

---

## Remediation

### Option A: Optimize Slow Query (5-10 min)

```bash
# If slow query identified (mean_time > 100ms):
# 1. Add index to frequently searched column
# 2. Query optimization

# After optimization:
psql $PROD_DB -c "REINDEX INDEX [index_name];"
# Wait 2 minutes, verify latency drops
```

### Option B: Scale Resources (5 min)

```bash
# Increase Cloud Run memory/CPU
gcloud run services update ollama-api \
  --memory=2Gi \
  --cpu=2 \
  --region=us-central1

# Takes 2 minutes to roll out
# Verify latency drops
```

### Option C: Enable Caching (10 min)

```bash
# Enable Redis caching for frequent queries
gcloud firestore update documents/config/cache \
  --update-mask="enabled=true,ttl=300"

# Warm cache with common requests
# Verify latency drops
```

---

## Escalation

- If latency still >1.5x baseline after 15 min: Page @engineering-lead
- If >3x baseline: Consider rollback to previous revision

**Created**: 2026-01-26
