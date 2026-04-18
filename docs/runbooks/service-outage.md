# Runbook: Complete Service Outage

**Version**: 1.0 | **Severity**: SEV1 | **Time to Resolution**: 30 min

---

## Detection

- **Alert**: All monitoring fails, API completely down
- **Symptom**: 100% requests returning errors or timing out
- **Dashboard**: [Status Page](https://status.elevatediq.ai)

---

## Immediate Actions (0-3 min)

**CRITICAL: This is an all-hands incident**

```bash
# Create: #incident-outage-[timestamp]

# Page EVERYONE:
# @engineering-lead @cto @on-call-backup @founders
# (Do not wait for normal escalation, go straight to top)

# Verify outage is real:
curl https://elevatediq.ai/ollama/api/v1/health
# Should return 200 OK
# If timeout or 503: Outage confirmed

# Post to status page immediately:
# "INVESTIGATING: Service may be experiencing issues"
```

---

## Diagnosis (0-5 min)

```bash
# Check 1: Is GCP region down?
gcloud compute regions describe us-central1
# Status should be UP

# Check 2: Is Cloud Run service responding?
gcloud run services describe ollama-api --region=us-central1
# State should be ACTIVE

# Check 3: Check GCP status dashboard
# https://status.cloud.google.com
# Look for: Any reported outages in us-central1

# Check 4: Database alive?
psql $PROD_DB -c "SELECT 1"
# If timeout: Database is down

# Check 5: Load Balancer alive?
gcloud compute backend-services describe ollama-lb-backend
# Status should be HEALTHY
```

---

## Remediation

### If GCP Region Down

```bash
# Cannot be fixed locally, wait for GCP recovery
# Estimated time: 5-30 minutes
# Post: "GCP us-central1 region is experiencing an outage. ETA for recovery: TBD"
# Monitor GCP status page for updates
```

### If Cloud Run Service Down

```bash
# Step 1: Check service logs
gcloud run services describe ollama-api --region=us-central1
# Status: ACTIVE should be visible

# Step 2: Check recent revisions
gcloud run revisions list ollama-api --region=us-central1 --limit=3

# Step 3: If recent revision is broken, rollback
gcloud run services update-traffic ollama-api \
  --to-revisions [PREVIOUS_STABLE_REVISION]=100 \
  --region=us-central1

# Step 4: Wait 30 seconds for traffic shift
sleep 30

# Step 5: Verify health
curl https://elevatediq.ai/ollama/api/v1/health
```

### If Database Down

```bash
# Step 1: Check Cloud SQL status
gcloud sql instances describe ollama-prod-db

# Step 2: If status is RUNNABLE but not responding:
#    Restart instance
gcloud sql instances restart ollama-prod-db

# Takes 2-3 minutes

# Step 3: Meanwhile, enable read-only mode
# Disable writes to prevent data corruption
gcloud firestore update documents/config/database \
  --update-mask="read_only=true"

# Step 4: After database restarts, verify
psql $PROD_DB -c "SELECT 1"

# Step 5: Re-enable writes
gcloud firestore update documents/config/database \
  --update-mask="read_only=false"
```

### If Load Balancer Down

```bash
# This is GCP infrastructure, requires GCP support
# Temporary workaround: Point DNS directly to Cloud Run service

# Step 1: Get Cloud Run service IP
gcloud run services describe ollama-api --region=us-central1 | grep routes

# Step 2: Update DNS (requires Cloud DNS access)
gcloud dns record-sets update ollama.example.com \
  --rrdatas=[CLOUD_RUN_IP] \
  --ttl=60 \
  --type=A \
  --zone=[ZONE]

# Step 3: Wait for DNS propagation (< 2 min)

# Step 4: Test connectivity
curl https://ollama.example.com/api/v1/health
```

---

## Communication

Post updates every 5 minutes to:

1. **Status Page**: https://status.elevatediq.ai (for customers)
2. **Slack**: #incident-outage channel (for team)
3. **Email**: Incident distribution list (if >30 min outage)

Example update:

```
[TIME] UPDATE
Status: Still investigating root cause
Scope: Service is unavailable
ETA: Should be resolved within 15 minutes
Latest action: Checking Cloud Run service health
Next update: [TIME + 5 min]
```

---

## Escalation

- **After 10 min**: Notify @founders
- **After 30 min**: Contact GCP support if infrastructure issue
- **After 60 min**: Prepare public communication / incident announcement

**Created**: 2026-01-26
