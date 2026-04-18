# 📋 Production Operational Runbooks

**Date**: January 13, 2026
**Status**: Phase 4 Complete
**Environment**: Production

---

## Table of Contents

1. [Incident Response](#incident-response)
2. [Performance Troubleshooting](#performance-troubleshooting)
3. [Database Operations](#database-operations)
4. [Scaling Operations](#scaling-operations)
5. [Security Incidents](#security-incidents)
6. [Disaster Recovery](#disaster-recovery)

---

## Incident Response

### P1: Service Down / 5xx Errors

**Severity**: Critical
**Response Time**: Immediate
**Resolution Target**: < 15 minutes

#### Detection
- Automated alert triggers when uptime check fails
- Error rate > 1%
- Status dashboard shows red

#### Immediate Actions (0-2 min)
```bash
# 1. Verify service status
gcloud run services describe ollama --platform managed --region us-central1

# 2. Check recent logs for errors
gcloud logging read "resource.labels.service_name='ollama' AND severity>=ERROR" \
  --limit 50 --format json

# 3. Check metrics
gcloud monitoring metrics-descriptors list --filter='metric.type:custom.googleapis.com'
```

#### Diagnosis (2-5 min)
- [ ] Is Cloud Run service running? (`gcloud run services list`)
- [ ] Are backend services responding? (PostgreSQL, Redis, Qdrant)
- [ ] Check database connection pool (`SELECT count(*) FROM pg_stat_activity`)
- [ ] Check memory usage in Cloud Run
- [ ] Check CPU usage in Cloud Run
- [ ] Review application logs for stack traces

#### Resolution (5-15 min)

**Option A: Restart Service**
```bash
# This triggers a new Cloud Run revision
gcloud run deploy ollama \
  --image gcr.io/ollama-prod/ollama:latest \
  --platform managed --region us-central1 \
  --update-env-vars "RESTART_TOKEN=$(date +%s)"
```

**Option B: Rollback to Previous Version**
```bash
# Use emergency rollback script
./scripts/rollback-production.sh
```

**Option C: Scale Up Resources**
```bash
# Increase instances and resources
gcloud run deploy ollama \
  --memory 12Gi --cpu 8 --max-instances 30 \
  --platform managed --region us-central1
```

#### Post-Incident (15+ min)
- [ ] Verify service stable for 5 minutes
- [ ] Check error rate returned to normal
- [ ] Document root cause
- [ ] Create incident ticket
- [ ] Schedule post-mortem within 24 hours

---

### P2: Degraded Performance (Latency > 10s)

**Severity**: High
**Response Time**: < 5 minutes
**Resolution Target**: < 30 minutes

#### Detection
- Latency alerts trigger (p99 > 10s)
- Error rate slightly elevated
- Some requests timing out

#### Immediate Actions
```bash
# 1. Check query performance
gcloud logging read "resource.labels.service_name='ollama'" \
  --filter="jsonPayload.duration_ms > 5000" \
  --limit 20 --format json

# 2. Identify slow endpoints
gcloud logging read "resource.labels.service_name='ollama'" \
  --format 'json' | \
  jq '.[] | {endpoint: .jsonPayload.endpoint, duration_ms: .jsonPayload.duration_ms}' | \
  sort -k 3 | tail -20

# 3. Check resource utilization
gcloud run describe ollama --platform managed --region us-central1
```

#### Diagnosis
- [ ] Database query performance (`EXPLAIN ANALYZE SELECT...`)
- [ ] Cache hit rate (Redis memory/evictions)
- [ ] Network latency to Ollama service
- [ ] CPU throttling in Cloud Run
- [ ] Memory pressure (GC pauses)

#### Resolution

**Option A: Scale Horizontally**
```bash
gcloud run deploy ollama \
  --min-instances 5 --max-instances 50 \
  --platform managed --region us-central1
```

**Option B: Optimize Database Queries**
```bash
# Create missing indexes
psql -h $DB_HOST -U $DB_USER -d ollama <<EOF
CREATE INDEX CONCURRENTLY idx_conversations_user_id ON conversations(user_id);
CREATE INDEX CONCURRENTLY idx_messages_conversation_id ON messages(conversation_id);
VACUUM ANALYZE;
EOF
```

**Option C: Clear Cache**
```bash
# Flush Redis cache (careful: will drop all cached data)
redis-cli -h $REDIS_HOST FLUSHDB
```

#### Monitoring After
- [ ] Latency normalized
- [ ] Error rate stable
- [ ] Resource utilization reasonable
- [ ] Continue monitoring for 30 minutes

---

## Performance Troubleshooting

### High Memory Usage

**Symptoms**: Memory approaching 85%+, slowness, potential OOM kills

```bash
# 1. Check memory usage
gcloud run metrics describe ollama \
  --metric-type=compute.googleapis.com/instance/memory/used_bytes

# 2. Identify memory leaks
# Check application logs for repeated allocations without cleanup

# 3. Increase memory allocation
gcloud run deploy ollama --memory 16Gi \
  --platform managed --region us-central1

# 4. Restart with garbage collection
gcloud run deploy ollama \
  --set-env-vars="GC_THRESHOLD=700000000" \
  --platform managed --region us-central1
```

### High CPU Usage

**Symptoms**: CPU > 90%, slow responses, potential CPU throttling

```bash
# 1. Check CPU usage
gcloud run metrics describe ollama \
  --metric-type=compute.googleapis.com/instance/cpu/utilization

# 2. Profile CPU usage (requires profiling enabled)
# Use py-spy or cProfile data from logs

# 3. Identify hot functions
# Review slow query logs, check inference bottlenecks

# 4. Scale horizontally
gcloud run deploy ollama \
  --min-instances 3 --max-instances 100 \
  --platform managed --region us-central1

# 5. Increase vCPUs
gcloud run deploy ollama --cpu 8 \
  --platform managed --region us-central1
```

### High Latency on Specific Endpoints

**Symptoms**: `/api/v1/embeddings` slow, `/api/v1/generate` timeout

```bash
# 1. Enable detailed logging
gcloud run deploy ollama \
  --set-env-vars="LOG_LEVEL=debug,LOG_SQL=true" \
  --platform managed --region us-central1

# 2. Review query patterns
psql -h $DB_HOST -U $DB_USER -d ollama <<EOF
SELECT query, calls, total_time, mean_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;
EOF

# 3. Check Ollama service latency
curl -X POST http://ollama:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"llama2","prompt":"test","stream":false}' \
  -w "Total time: %{time_total}s"

# 4. Optimize database schema or queries
# Create appropriate indexes, denormalize if needed
```

---

## Database Operations

### Backup & Recovery

#### Automated Backups
```bash
# Backups run automatically via Cloud SQL
# Verify backup schedule
gcloud sql instances describe ollama-prod \
  --format="value(backupConfiguration)"

# Manual backup
gcloud sql backups create \
  --instance=ollama-prod \
  --description="Manual backup before deployment"
```

#### Point-in-Time Recovery
```bash
# Restore to specific timestamp
gcloud sql backups restore \
  --backup-id=<BACKUP_ID> \
  --instance=ollama-prod \
  --backup-configuration=<CONFIG>

# Or clone database to new instance for testing
gcloud sql instances clone ollama-prod ollama-test-recovery
```

### Database Maintenance

#### Vacuum & Analyze
```bash
psql -h $DB_HOST -U $DB_USER -d ollama <<EOF
-- Reclaim space from deleted rows
VACUUM FULL;

-- Update table statistics for query planner
ANALYZE;
EOF
```

#### Index Maintenance
```bash
psql -h $DB_HOST -U $DB_USER -d ollama <<EOF
-- Reindex all tables (may lock tables temporarily)
REINDEX DATABASE ollama;

-- Check index bloat
SELECT indexname, (1 - (pg_relation_size(indexrelname)::float /
  pg_relation_size(tablename)::float)) * 100 AS waste
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY waste DESC;
EOF
```

### Connection Pool Troubleshooting

```bash
# Check connection pool status
psql -h $DB_HOST -U $DB_USER -d ollama <<EOF
SELECT count(*) FROM pg_stat_activity;
SELECT datname, usename, state, count(*)
FROM pg_stat_activity
GROUP BY datname, usename, state;
EOF

# Kill idle connections if pool exhausted
psql -h $DB_HOST -U $DB_USER -d ollama <<EOF
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE state = 'idle' AND state_change < NOW() - INTERVAL '10 minutes';
EOF
```

---

## Scaling Operations

### Horizontal Scaling (Add Instances)

```bash
# Increase number of instances
gcloud run deploy ollama \
  --min-instances 3 \
  --max-instances 100 \
  --platform managed --region us-central1

# Monitor scaling
watch -n 5 'gcloud run describe ollama --platform managed --region us-central1'
```

### Vertical Scaling (Add Resources)

```bash
# Increase vCPU and memory
gcloud run deploy ollama \
  --cpu 8 \
  --memory 16Gi \
  --platform managed --region us-central1

# Note: Creates new revision, automatically handles traffic migration
```

### Load Testing

```bash
# Install load testing tool
pip install locust

# Run load test
locust -f load_test.py \
  --host https://elevatediq.ai/ollama \
  --users 100 \
  --spawn-rate 10 \
  --run-time 5m

# Analyze results and adjust scaling
```

---

## Security Incidents

### P1: Suspected Breach / Compromised Credentials

**Immediate Actions**:
```bash
# 1. Rotate API keys
# Go to Cloud Console > API Keys > Rotate

# 2. Check access logs for suspicious activity
gcloud logging read "resource.labels.service_name='ollama' AND severity>=WARNING" \
  --format="table(timestamp, jsonPayload.user, jsonPayload.ip, jsonPayload.method)"

# 3. Block suspicious IPs (if applicable)
# Update Cloud Armor security policies

# 4. Force re-authentication
# Invalidate all sessions by rotating JWT signing keys
```

### P2: Vulnerability Detected

```bash
# 1. Identify affected version
gcloud run services describe ollama --format='value(spec.template.spec.containers[0].image)'

# 2. Patch dependency
pip install --upgrade vulnerable_package

# 3. Rebuild image
docker build -t gcr.io/ollama-prod/ollama:patched .
docker push gcr.io/ollama-prod/ollama:patched

# 4. Deploy patched version
gcloud run deploy ollama \
  --image gcr.io/ollama-prod/ollama:patched \
  --platform managed --region us-central1
```

### Access Control Review

```bash
# List all users with production access
gcloud projects get-iam-policy ollama-prod

# Remove unnecessary permissions
gcloud projects remove-iam-policy-binding ollama-prod \
  --member=user:email@example.com \
  --role=roles/run.admin

# Audit service account permissions
gcloud iam service-accounts list
gcloud iam service-accounts get-iam-policy <SERVICE_ACCOUNT>
```

---

## Disaster Recovery

### Full Service Recovery (Region Down)

```bash
# 1. Failover to secondary region
# Update DNS to point to us-east1 instance

# 2. Restore from backup to new region
gcloud sql instances clone \
  --async ollama-prod ollama-recovery \
  --region=us-east1

# 3. Deploy service to new region
gcloud run deploy ollama-recovery \
  --image gcr.io/ollama-prod/ollama:latest \
  --region us-east1 \
  --platform managed

# 4. Run integration tests
pytest tests/integration/test_recovery.py \
  --endpoint=https://ollama-recovery.a.run.app

# 5. Update DNS
# Point ollama.example.com to new region

# 6. Verify traffic flowing correctly
gcloud logging read "resource.labels.service_name='ollama'" \
  --limit 100 --format json | jq '.[] | .httpRequest.latency'
```

### Database Recovery (Data Corruption)

```bash
# 1. Create backup clone
gcloud sql instances clone ollama-prod ollama-recovery-temp

# 2. Connect to clone and verify data
psql -h <RECOVERY_HOST> -U postgres -d ollama

# 3. If valid, promote clone to production
# Failover or manually migrate connections

# 4. If corrupted, restore from earlier backup
gcloud sql backups restore <EARLIER_BACKUP_ID> \
  --backup-configuration=<CONFIG> \
  --instance=ollama-prod
```

### Complete Environment Recovery

```bash
# Run comprehensive disaster recovery test
./scripts/test-disaster-recovery.sh

# Validate all components are operational
# - Application servers
# - Database
# - Cache
# - Message queues
# - Monitoring
# - DNS
```

---

## Contact & Escalation

### On-Call Rotation
- Primary: [@slack-handle](https://slack.com)
- Secondary: [@slack-handle](https://slack.com)

### Emergency Contacts
- VP Engineering: [Contact Info]
- Security Team: security@company.com
- Infrastructure Team: infra@company.com

### Escalation Path
1. On-call engineer (this runbook)
2. Secondary on-call (if no response in 5 min)
3. Team lead (if P1 after 15 min)
4. VP Engineering (if no resolution in 30 min)

---

## Common Commands Reference

```bash
# Service management
gcloud run services list
gcloud run services describe ollama
gcloud run services update-traffic ollama --to-revisions PREVIOUS=100

# Logs and monitoring
gcloud logging read "resource.labels.service_name='ollama'" --limit 100
gcloud monitoring metrics list
gcloud monitoring time-series list

# Database operations
gcloud sql instances describe ollama-prod
gcloud sql backups list --instance ollama-prod
psql -h $DB_HOST -U postgres -d ollama

# Deployment
gcloud run deploy ollama --image gcr.io/ollama-prod/ollama:latest
./scripts/deploy-staging.sh
./scripts/deploy-production.sh
./scripts/rollback-production.sh
```

---

## Document Updates

- **Last Updated**: January 13, 2026
- **Next Review**: January 20, 2026
- **Owner**: Engineering Team
- **Version**: 1.0

**For Questions**: Contact the on-call engineer or open an issue in the ops repository.
