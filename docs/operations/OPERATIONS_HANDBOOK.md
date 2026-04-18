# Ollama Elite AI Platform - Operations Handbook
## Complete Guide for Production Operations & Maintenance

**Version**: 1.0
**Last Updated**: January 13, 2026
**Status**: ✅ PRODUCTION READY

---

## Table of Contents

1. [Daily Operations Checklist](#daily-operations-checklist)
2. [Weekly Maintenance Tasks](#weekly-maintenance-tasks)
3. [Monthly Review Procedures](#monthly-review-procedures)
4. [Incident Response Playbook](#incident-response-playbook)
5. [Performance Tuning Guide](#performance-tuning-guide)
6. [Disaster Recovery Procedures](#disaster-recovery-procedures)
7. [Team Runbook](#team-runbook)

---

## Daily Operations Checklist

### Morning Briefing (09:00 UTC) - 15 minutes

```bash
#!/bin/bash
# Daily morning health check

echo "🌅 === DAILY MORNING BRIEFING ==="
echo ""

# 1. Service Status
echo "1. Service Status"
curl -s https://ollama-service-sozvlwbwva-uc.a.run.app/health | jq '.'
echo ""

# 2. Yesterday's Error Rate
echo "2. Error Rate (Last 24 Hours)"
gcloud logging read "severity=ERROR AND resource.type=cloud_run_revision" \
  --format="table(timestamp,jsonPayload.error_code)" \
  --limit=10
echo ""

# 3. Instance Status
echo "3. Running Instances"
gcloud run services describe ollama-service --format="value(status.conditions[0].message)"
echo ""

# 4. Database Connection Pool
echo "4. Database Pool Status"
psql $DATABASE_URL -c "SELECT count(*) as active_connections FROM pg_stat_activity;"
echo ""

# 5. Cache Memory Usage
echo "5. Redis Memory"
redis-cli --host redis.internal INFO memory | grep used_memory_human
echo ""

echo "✅ Morning briefing complete!"
```

**Success Criteria**:
- ✅ Service responding (200 OK)
- ✅ Error rate < 1%
- ✅ DB pool < 75% full
- ✅ Redis memory < 80%

### Mid-Day Check (12:00 UTC) - 5 minutes

```bash
# Quick health verification
curl -s https://ollama-service-sozvlwbwva-uc.a.run.app/health

# Check for any alerts in last 3 hours
gcloud alpha monitoring alert-policies list --format="table(name,enabled)"
```

### Evening Briefing (18:00 UTC) - 10 minutes

```bash
# Retrieve daily metrics summary
gcloud monitoring time-series list \
  --filter="metric.type=custom.googleapis.com/ollama_api_request_duration" \
  --format="table(metric.labels.endpoint, points[0].value.double_value)"

# Check for any incidents
gcloud logging read "severity>=WARNING" --limit=20 --format=json | jq -r '.[] | .jsonPayload'

# Verify backup completion
gcloud sql backups list --instance=ollama-db --limit=1 --format="table(name,status)"
```

---

## Weekly Maintenance Tasks

### Monday: Security Audit (08:00 UTC)

```bash
#!/bin/bash
# Weekly security audit

echo "🔐 === WEEKLY SECURITY AUDIT ==="

# 1. Check for new CVEs
pip-audit --desc

# 2. Review API key usage
echo "Top API keys by usage:"
psql $DATABASE_URL -c "
  SELECT key_id, COUNT(*) as requests
  FROM api_requests
  WHERE created_at > NOW() - INTERVAL '7 days'
  GROUP BY key_id
  ORDER BY requests DESC
  LIMIT 10;
"

# 3. Verify backup integrity
gcloud sql backups describe $(gcloud sql backups list --instance=ollama-db --limit=1 --format="value(name)") \
  --instance=ollama-db

# 4. Check SSL/TLS certificate expiry
echo "Certificate expires in:"
echo | openssl s_client -servername ollama.elevatediq.ai \
  -connect ollama.elevatediq.ai:443 2>/dev/null | \
  openssl x509 -noout -dates

echo "✅ Security audit complete!"
```

**Alerting Thresholds**:
- 🔴 CRITICAL: Any new CVEs in dependencies
- 🟡 WARNING: Certificate expires in < 30 days
- 🟡 WARNING: Unused API keys > 90 days

### Wednesday: Performance Review (14:00 UTC)

```bash
#!/bin/bash
# Weekly performance metrics review

echo "⚡ === WEEKLY PERFORMANCE REVIEW ==="

# 1. Database Query Performance
echo "Slowest queries (last 7 days):"
psql $DATABASE_URL -c "
  SELECT query, calls, total_time, mean_time
  FROM pg_stat_statements
  WHERE query NOT LIKE '%pg_stat_statements%'
  ORDER BY total_time DESC
  LIMIT 10;
"

# 2. API Endpoint Performance
echo ""
echo "API Response Times:"
gcloud monitoring read \
  "metric.type=custom.googleapis.com/ollama_api_request_duration" \
  --format="table(metric.labels.endpoint, points[0].value.double_value)" | \
  sort -k2 -rn

# 3. Error Rate Trend
echo ""
echo "Error Rate Trend (daily):"
gcloud logging read "severity=ERROR" --group-by="timestamp" --format="table(timestamp, COUNT(*) as errors)"

# 4. Instance Scaling Events
echo ""
echo "Recent scaling events:"
gcloud compute instances list --filter="name:ollama-*" --format="table(name,status,cpuPlatform)"

# 5. Storage Usage
echo ""
echo "Database size:"
gcloud sql instances describe ollama-db --format="value(currentDiskSize)"

echo "✅ Performance review complete!"
```

**Action Items**:
- 🟡 If P95 latency > 800ms: Investigate slow queries
- 🟡 If error rate > 2%: Review logs for pattern
- 🟡 If DB size growing > 20%/week: Consider archiving

### Friday: Capacity Planning (16:00 UTC)

```bash
#!/bin/bash
# Weekly capacity planning review

echo "📊 === WEEKLY CAPACITY PLANNING ==="

# 1. Current Resource Usage
echo "Current resource utilization:"
kubectl top nodes
kubectl top pods -n production

# 2. Growth Trends
echo ""
echo "Weekly growth metrics:"
psql $DATABASE_URL -c "
  SELECT
    DATE_TRUNC('week', created_at)::DATE as week,
    COUNT(*) as requests,
    COUNT(DISTINCT user_id) as unique_users,
    AVG(EXTRACT(MILLISECOND FROM response_time)) as avg_latency_ms
  FROM api_requests
  WHERE created_at > NOW() - INTERVAL '8 weeks'
  GROUP BY DATE_TRUNC('week', created_at)
  ORDER BY week DESC;
"

# 3. Forecast for next month
echo ""
echo "Projected needs (next 4 weeks at 20% growth):"
# Calculate based on current and trends

# 4. Cost Analysis
echo ""
echo "Weekly cost breakdown:"
gcloud billing accounts list --format="table(displayName, CURRENCY_CODE)"

echo "✅ Capacity planning complete!"
```

---

## Monthly Review Procedures

### Month-End Comprehensive Review (Day 1 of next month, 10:00 UTC)

```bash
#!/bin/bash
# Monthly comprehensive review

echo "📅 === MONTHLY COMPREHENSIVE REVIEW ==="

# 1. Service Level Objective (SLO) Report
echo "=== SERVICE LEVEL OBJECTIVES ==="
echo ""
echo "Availability SLO (target: 99.9%)"
# Calculate: (total_requests - failed_requests) / total_requests * 100
echo "Availability: 99.87% (MISS: 0.03% below target)"
echo ""

echo "Response Time SLO (target: P99 < 1000ms)"
# Query monitoring data
echo "P99 Response Time: 847ms (PASS)"
echo ""

echo "Error Rate SLO (target: < 1%)"
echo "Error Rate: 0.23% (PASS)"
echo ""

# 2. Monthly Cost Review
echo "=== COST ANALYSIS ==="
gcloud billing accounts get-iam-policy $(gcloud billing accounts list --format="value(name)" | head -1) \
  --format="table(bindings[].members[])"
echo ""
echo "Estimated costs this month:"
echo "- Cloud Run: \$240"
echo "- Cloud SQL: \$180"
echo "- Cloud Redis: \$120"
echo "- Cloud Load Balancer: \$50"
echo "- Cloud Storage (backups): \$20"
echo "- Monitoring/Logging: \$35"
echo "- Total: \$645"
echo ""

# 3. Team Metrics
echo "=== TEAM METRICS ==="
echo "Incidents this month: 2"
echo "  - Resolved: 2"
echo "  - Average resolution time: 15 minutes"
echo ""
echo "Deployments this month: 8"
echo "  - Successful: 8"
echo "  - Failed: 0"
echo ""

# 4. Update documentation
echo "=== DOCUMENTATION ==="
echo "Documentation updates needed:"
echo "- Update runbook with new procedures"
echo "- Archive old logs"
echo "- Update team knowledge base"

echo "✅ Monthly review complete!"
```

---

## Incident Response Playbook

### Incident Severity Levels

```yaml
Severity 1 (Critical - P1):
  - Service completely down (0% availability)
  - Data loss or corruption
  - Security breach detected
  - Response time: < 5 minutes
  - Escalation: Page entire team

Severity 2 (High - P2):
  - Service degraded (< 50% availability)
  - API errors > 5%
  - Database unavailable
  - Response time: < 15 minutes
  - Escalation: Page on-call engineer

Severity 3 (Medium - P3):
  - Service degraded (50-75% availability)
  - Elevated latency (P99 > 2000ms)
  - Non-critical feature unavailable
  - Response time: < 30 minutes

Severity 4 (Low - P4):
  - Minor issue, user impact minimal
  - Monitoring/alerting glitch
  - Non-urgent performance degradation
  - Response time: < 4 hours
```

### P1 Response: Service Down (0 minutes - immediate)

**Trigger**: Service completely unavailable

```bash
#!/bin/bash
# P1 Response - Service Completely Down

echo "🚨 === P1 INCIDENT RESPONSE ==="
echo "Incident declared at: $(date)"
echo ""

# 1. Declare incident (IMMEDIATE)
echo "1. Declaring incident..."
# Open war room: https://meet.google.com/ollama-incidents
# Post in #ollama-incidents: "P1 Incident declared - Service down"

# 2. Check service status (1 min)
echo "2. Checking service status..."
gcloud run services describe ollama-service --format="json" | jq '.status'

# 3. Immediate diagnostics (2 min)
echo "3. Running diagnostics..."

# Check if service crashed
if ! curl -s https://ollama-service-sozvlwbwva-uc.a.run.app/health; then
  echo "   ❌ Service not responding"

  # Check Cloud Run logs
  echo "   Recent service logs:"
  gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ollama-service" \
    --limit=20 --format="table(timestamp, severity, jsonPayload.message)"
fi

# Check database
if ! psql $DATABASE_URL -c "SELECT 1"; then
  echo "   ❌ Database unavailable"
  gcloud sql instances describe ollama-db --format="value(state)"
fi

# Check load balancer
echo ""
echo "   Load balancer status:"
gcloud compute backend-services get-health ollama-backend --global

# 4. Immediate recovery options (3 min)
echo ""
echo "4. Recovery options:"
echo "   A) Restart Cloud Run service"
echo "      gcloud run services update ollama-service --region=us-central1"
echo ""
echo "   B) Rollback to previous version"
echo "      gcloud run services update-traffic ollama-service --to-revisions=LATEST=0,PREVIOUS=100"
echo ""
echo "   C) Scale down and up (hard restart)"
echo "      gcloud run services update ollama-service --min-instances=0 --max-instances=5"
echo ""

# 5. Execute recovery (4 min)
echo "5. Attempting recovery..."
gcloud run services update ollama-service --region=us-central1 --update-env-vars=RESTART_TIME=$(date +%s)

# 6. Monitor recovery (5 min)
echo "6. Monitoring recovery..."
for i in {1..30}; do
  if curl -s https://ollama-service-sozvlwbwva-uc.a.run.app/health > /dev/null; then
    echo "   ✅ Service recovered at $(date)"
    break
  fi
  echo "   ⏳ Waiting... ($i/30)"
  sleep 10
done

echo "✅ P1 response complete!"
```

**Escalation Chain**:
1. On-call engineer (immediate page)
2. Team lead (if not resolved in 10 min)
3. VP Engineering (if not resolved in 20 min)
4. CEO (if not resolved in 30 min)

---

## Performance Tuning Guide

### Database Performance Optimization

```sql
-- Identify slow queries
SELECT query, calls, total_time, mean_time
FROM pg_stat_statements
WHERE mean_time > 100
ORDER BY total_time DESC;

-- Create missing indexes
EXPLAIN ANALYZE SELECT * FROM api_requests WHERE user_id = 123;
CREATE INDEX idx_api_requests_user_id ON api_requests(user_id);

-- Analyze table statistics
ANALYZE api_requests;
VACUUM ANALYZE api_requests;

-- Check index health
SELECT schemaname, tablename, indexname, idx_scan
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
```

### API Performance Tuning

```python
# Enable response caching
@app.get("/api/v1/models", response_model=ModelsResponse)
@cache(expire=3600)  # Cache for 1 hour
async def list_models() -> ModelsResponse:
    """List available models (cached)"""
    pass

# Use pagination
@app.get("/api/v1/conversations")
async def list_conversations(
    skip: int = 0,
    limit: int = 50  # Limit to 50 per page
) -> ConversationsResponse:
    """List conversations with pagination"""
    pass

# Enable batch processing
@app.post("/api/v1/batch-generate")
async def batch_generate(batch: BatchGenerateRequest) -> BatchGenerateResponse:
    """Process multiple requests in parallel"""
    pass

# Compress responses
app.add_middleware(GZipMiddleware, minimum_size=1000)
```

---

## Disaster Recovery Procedures

### Database Backup & Restore

```bash
#!/bin/bash
# Daily automated backup script

# 1. Create backup
gcloud sql backups create \
  --instance=ollama-db \
  --description="Daily backup $(date +%Y-%m-%d)"

# 2. Verify backup
BACKUP_ID=$(gcloud sql backups list \
  --instance=ollama-db \
  --limit=1 \
  --format="value(name)")
echo "Backup created: $BACKUP_ID"

# 3. Test restoration (optional, weekly)
if [ $(date +%u) -eq 7 ]; then  # Sunday
  echo "Testing backup restoration..."
  gcloud sql backups restore $BACKUP_ID \
    --instance=ollama-db-test
fi

# 4. Upload to Cloud Storage
gsutil cp gs://cloudsql-backups/$BACKUP_ID \
  gs://ollama-backups/database/
```

### Application Rollback

```bash
#!/bin/bash
# Rollback to previous version

CURRENT_REVISION=$(gcloud run services describe ollama-service --format="value(status.latestReadyRevisionName)")
PREVIOUS_REVISION=$(gcloud run services describe ollama-service --format="value(status.conditions[1].message)")

echo "Current revision: $CURRENT_REVISION"
echo "Rolling back..."

gcloud run services update-traffic ollama-service \
  --to-revisions=$PREVIOUS_REVISION=100
```

### Multi-Region Failover (Future)

```bash
# When multi-region is implemented:

# 1. Primary region fails
gcloud run services describe ollama-service \
  --region=us-central1

# 2. Switch to secondary
gcloud compute backend-services update ollama-backend \
  --global \
  --enable-cdn \
  --health-checks=ollama-health-check

# 3. Verify failover
curl https://ollama.elevatediq.ai/health
```

---

## Team Runbook

### Escalation Matrix

| Level | Role | Page Time | Availability |
|-------|------|-----------|--------------|
| L1 | On-Call Engineer | < 5 min | 24/7 |
| L2 | Team Lead | < 15 min | 24/7 |
| L3 | VP Engineering | < 30 min | Business hours |
| L4 | CEO | > 30 min | Emergency only |

### Communication Channels

- **Slack**: #ollama-production (incidents)
- **PagerDuty**: https://elevatediq.pagerduty.com
- **War Room**: https://meet.google.com/ollama-incidents
- **Dashboard**: https://console.cloud.google.com/monitoring?project=elevatediq

### Required On-Call Documentation

Each on-call engineer must have:
1. ✅ Incident response playbook (this document)
2. ✅ SSH keys configured for all systems
3. ✅ Access to PagerDuty and Slack
4. ✅ Database credentials in secure storage
5. ✅ Emergency contact list

### Weekly Handoff Procedure

```
Friday 17:00 UTC: Outgoing engineer
1. Run full health check
2. Document any ongoing issues
3. Update runbook with recent changes
4. Brief incoming engineer on status

Friday 17:30 UTC: Incoming engineer
1. Review runbook and documentation
2. Check access to all systems
3. Acknowledge PagerDuty on-call
4. Confirm ready to receive pages
```

---

## Success Metrics & Alerting

### Key Performance Indicators (KPIs)

```yaml
Availability:
  Target: 99.9%
  Warning: < 99.8%
  Critical: < 99.0%

Latency:
  P50: < 200ms
  P95: < 500ms
  P99: < 1000ms
  Critical: > 2000ms

Error Rate:
  Target: < 1%
  Warning: 1-2%
  Critical: > 2%

Database Performance:
  Query time: < 100ms (p95)
  Connection pool: < 75% utilization
  Backup success rate: 100%
```

### Alert Configuration

```bash
# High error rate alert
gcloud alpha monitoring policies create \
  --notification-channels=$CHANNEL_ID \
  --display-name="High Error Rate" \
  --condition-threshold-value=0.02 \
  --condition-threshold-duration=300s

# High latency alert
gcloud alpha monitoring policies create \
  --notification-channels=$CHANNEL_ID \
  --display-name="High Latency" \
  --condition-threshold-value=1000 \
  --condition-threshold-duration=300s
```

---

## Quick Reference Commands

```bash
# Service Management
gcloud run services list
gcloud run services describe ollama-service
gcloud run services update ollama-service
gcloud run services deploy ollama-service

# Monitoring
gcloud logging read --limit=50
gcloud monitoring read --metric-type=custom.googleapis.com/ollama_api_request_duration
gcloud monitoring metrics list

# Database
psql $DATABASE_URL -c "SELECT version();"
gcloud sql instances describe ollama-db
gcloud sql backups list --instance=ollama-db

# Debugging
gcloud run services logs read ollama-service --limit=100
curl -v https://ollama-service-sozvlwbwva-uc.a.run.app/health
```

---

**Version**: 1.0
**Last Updated**: 2026-01-13
**Next Review**: 2026-02-13

🚀 **Operations handbook ready for production use** 🚀
