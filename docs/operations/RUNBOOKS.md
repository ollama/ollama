# Ollama Operational Runbooks

**Version**: 0.1.0
**Last Updated**: January 18, 2026
**Owner**: AI Infrastructure Team

---

## Table of Contents

- [Emergency Contacts](#emergency-contacts)
- [Escalation Procedures](#escalation-procedures)
- [Incident Response](#incident-response)
- [Common Incident Scenarios](#common-incident-scenarios)
  - [Service Unavailable (503)](#service-unavailable-503)
  - [High Error Rate (500s)](#high-error-rate-500s)
  - [Slow Response Times](#slow-response-times)
  - [Database Connection Issues](#database-connection-issues)
  - [Redis Cache Failures](#redis-cache-failures)
  - [Ollama Model Failures](#ollama-model-failures)
  - [API Authentication Failures](#api-authentication-failures)
  - [Rate Limit Issues](#rate-limit-issues)
  - [Out of Memory (OOM)](#out-of-memory-oom)
  - [Disk Space Exhaustion](#disk-space-exhaustion)
- [Monitoring & Alerting](#monitoring--alerting)
- [Routine Maintenance](#routine-maintenance)
- [Post-Incident Review](#post-incident-review)

---

## Emergency Contacts

### On-Call Rotation

**Primary On-Call**: Check PagerDuty for current rotation

**Escalation Path**:
1. **L1 Support**: AI Infrastructure Team (response: 15 minutes)
2. **L2 Support**: Platform Engineering Lead (response: 30 minutes)
3. **L3 Support**: CTO (response: 1 hour)

### Communication Channels

| Channel | Purpose | Response Time |
|---------|---------|---------------|
| **PagerDuty** | Critical incidents | Immediate |
| **Slack: #ai-infrastructure** | Team coordination | 5 minutes |
| **Slack: #incidents** | Cross-team incidents | 15 minutes |
| **Email: ai-infrastructure@elevatediq.ai** | Non-urgent issues | 24 hours |
| **GCP Console Alerts** | Infrastructure alerts | Automatic |

### External Contacts

| Service | Contact | Use For |
|---------|---------|---------|
| **GCP Support** | support.google.com/cloud | Infrastructure issues |
| **Ollama Community** | discord.gg/ollama | Model/engine issues |
| **Docker Hub Status** | status.docker.com | Image pull failures |

---

## Escalation Procedures

### When to Escalate

**Escalate Immediately** if:
- Service completely unavailable > 5 minutes
- Data corruption detected
- Security breach suspected
- Customer-facing impact > 100 users
- Unable to diagnose root cause within 15 minutes

### Escalation Steps

**1. Initial Assessment** (0-5 minutes):
```bash
# Quick health check
curl https://elevatediq.ai/ollama/health

# Check all services
docker-compose ps

# Check recent logs
docker-compose logs --tail=100 api
```

**2. Notify Team** (5-10 minutes):
```
# Slack message template
@here INCIDENT: Ollama API unavailable
Severity: P1
Impact: All API requests failing (503)
Started: 2026-01-18 10:30 UTC
War room: #incident-2026-01-18
```

**3. Engage On-Call** (10-15 minutes):
- Page on-call engineer via PagerDuty
- Include incident summary and initial findings
- Share war room link

**4. Escalate to Leadership** (if needed):
- After 30 minutes with no resolution
- If impact > 500 users
- If data loss possible

---

## Incident Response

### Incident Severity Levels

| Level | Description | Response Time | Example |
|-------|-------------|---------------|---------|
| **P0** | Critical - Complete outage | 5 minutes | API down, database unavailable |
| **P1** | High - Major degradation | 15 minutes | High error rate, slow responses |
| **P2** | Medium - Partial impact | 1 hour | Single endpoint failing, cache down |
| **P3** | Low - Minor issues | 4 hours | Logging issues, monitoring gaps |

### Incident Response Template

```markdown
## Incident Report: [TITLE]

**Severity**: P0 / P1 / P2 / P3
**Start Time**: YYYY-MM-DD HH:MM UTC
**End Time**: YYYY-MM-DD HH:MM UTC
**Duration**: X hours Y minutes
**Affected Services**: API, Database, etc.
**Impact**: X users affected, Y% error rate

### Timeline
- HH:MM - Incident detected (alert/user report)
- HH:MM - On-call engineer paged
- HH:MM - Root cause identified
- HH:MM - Fix applied
- HH:MM - Service restored
- HH:MM - Monitoring confirmed stable

### Root Cause
[Detailed explanation of what went wrong]

### Resolution
[What was done to fix the issue]

### Action Items
- [ ] Fix X (Owner: @name, Due: YYYY-MM-DD)
- [ ] Improve monitoring for Y (Owner: @name, Due: YYYY-MM-DD)
- [ ] Update runbook with Z (Owner: @name, Due: YYYY-MM-DD)

### Lessons Learned
[What we learned and how to prevent in future]
```

---

## Common Incident Scenarios

### Service Unavailable (503)

**Symptoms**:
- Health check returns 503
- All API requests failing
- Users cannot connect

**Diagnosis**:

```bash
# 1. Check if API container is running
docker-compose ps api
# Expected: State = Up

# 2. Check API logs
docker-compose logs --tail=50 api

# 3. Check resource usage
docker stats
# Look for: CPU 100%, Memory > 90%

# 4. Check external dependencies
curl http://ollama:11434/api/version  # Ollama
docker-compose exec postgres pg_isready  # PostgreSQL
docker-compose exec redis redis-cli ping  # Redis
```

**Resolution**:

**If API container crashed**:
```bash
# Restart API
docker-compose restart api

# Watch startup logs
docker-compose logs -f api

# Verify health
curl https://elevatediq.ai/ollama/health
```

**If resource exhaustion**:
```bash
# Scale down (reduce workers)
docker-compose scale api=1

# OR restart with more resources
docker-compose down
docker-compose up -d --scale api=2
```

**If dependency unavailable**:
```bash
# Restart failed dependency
docker-compose restart postgres  # or redis, ollama

# Restart API after dependency recovers
docker-compose restart api
```

**Prevention**:
- Implement auto-restart policies (already configured)
- Add resource limits to prevent OOM
- Implement circuit breakers for dependencies
- Add retry logic with exponential backoff

---

### High Error Rate (500s)

**Symptoms**:
- Error rate > 5% for 5 minutes
- 500 Internal Server Error responses
- Alert: "High error rate detected"

**Diagnosis**:

```bash
# 1. Check error logs
docker-compose logs api | grep -i error | tail -50

# 2. Find common error pattern
docker-compose logs api | grep "500" | awk '{print $NF}' | sort | uniq -c | sort -nr

# 3. Check specific error details
docker-compose logs api | grep -i "traceback" -A 20
```

**Resolution**:

**If database connection errors**:
```bash
# Check database health
docker-compose exec postgres pg_isready

# Check connection pool
docker-compose exec postgres psql -U ollama_user -d ollama -c \
  "SELECT count(*) FROM pg_stat_activity;"

# Restart database connections
docker-compose restart api
```

**If model loading failures**:
```bash
# Check available disk space
df -h

# Check Ollama health
docker-compose exec ollama ollama list

# Re-pull model if corrupted
docker-compose exec ollama ollama rm llama3.2
docker-compose exec ollama ollama pull llama3.2
```

**If uncaught exceptions**:
```bash
# Review exception traceback
docker-compose logs api | grep -i "traceback" -A 20

# Fix code bug (deploy hotfix)
# OR rollback to previous version
docker-compose down
git checkout previous-stable-tag
docker-compose up -d
```

**Prevention**:
- Add comprehensive error handling
- Increase test coverage for edge cases
- Implement proper exception logging
- Add retry logic for transient failures

---

### Slow Response Times

**Symptoms**:
- P99 latency > 10 seconds
- Alert: "High latency detected"
- Users reporting slow API

**Diagnosis**:

```bash
# 1. Check overall latency
curl -w "@curl-format.txt" -o /dev/null -s \
  -H "Authorization: Bearer <api-key>" \
  https://elevatediq.ai/ollama/api/v1/models

# curl-format.txt:
# time_namelookup:  %{time_namelookup}\n
# time_connect:  %{time_connect}\n
# time_appconnect:  %{time_appconnect}\n
# time_pretransfer:  %{time_pretransfer}\n
# time_redirect:  %{time_redirect}\n
# time_starttransfer:  %{time_starttransfer}\n
# time_total:  %{time_total}\n

# 2. Check inference latency
docker-compose logs api | grep "inference_completed" | tail -20

# 3. Check database query time
docker-compose exec postgres psql -U ollama_user -d ollama -c \
  "SELECT query, mean_exec_time, calls FROM pg_stat_statements \
   ORDER BY mean_exec_time DESC LIMIT 10;"

# 4. Check cache hit rate
docker-compose exec redis redis-cli info stats | grep keyspace
```

**Resolution**:

**If database slow**:
```bash
# Add missing indexes
docker-compose exec postgres psql -U ollama_user -d ollama -c \
  "CREATE INDEX idx_conversations_user_id ON conversations(user_id);"

# Analyze and vacuum
docker-compose exec postgres psql -U ollama_user -d ollama -c \
  "ANALYZE; VACUUM;"
```

**If cache misses high**:
```bash
# Warm cache with popular models
docker-compose exec ollama ollama pull llama3.2
docker-compose exec ollama ollama pull mistral

# Increase cache TTL
# Edit .env: CACHE_TTL=3600
docker-compose restart api
```

**If model loading slow**:
```bash
# Pre-load models on startup
docker-compose exec api python -c "
from ollama.services.models import OllamaModelManager
import asyncio
async def warmup():
    manager = OllamaModelManager()
    await manager.load_model('llama3.2')
asyncio.run(warmup())
"
```

**Prevention**:
- Implement caching at multiple layers
- Add database query optimization
- Pre-load frequently used models
- Implement connection pooling
- Add CDN for static content

---

### Database Connection Issues

**Symptoms**:
- "Could not connect to database" errors
- API returns 503
- Database queries timing out

**Diagnosis**:

```bash
# 1. Check PostgreSQL is running
docker-compose ps postgres

# 2. Test database connection
docker-compose exec postgres pg_isready

# 3. Check connection limits
docker-compose exec postgres psql -U ollama_user -d ollama -c \
  "SHOW max_connections;"

# 4. Check active connections
docker-compose exec postgres psql -U ollama_user -d ollama -c \
  "SELECT count(*) FROM pg_stat_activity;"

# 5. Check for locks
docker-compose exec postgres psql -U ollama_user -d ollama -c \
  "SELECT * FROM pg_locks WHERE NOT granted;"
```

**Resolution**:

**If PostgreSQL down**:
```bash
# Restart PostgreSQL
docker-compose restart postgres

# Check logs for errors
docker-compose logs postgres | tail -100

# If corrupted, restore from backup
docker-compose exec postgres psql -U ollama_user -d ollama < \
  backups/ollama-latest.sql
```

**If connection pool exhausted**:
```bash
# Restart API to reset connections
docker-compose restart api

# Increase connection pool size (if needed)
# Edit config/production.yaml:
# database:
#   pool_size: 30  # Increased from 20
docker-compose restart api
```

**If deadlocks**:
```bash
# Kill blocking queries
docker-compose exec postgres psql -U ollama_user -d ollama -c \
  "SELECT pg_terminate_backend(pid) FROM pg_stat_activity \
   WHERE state = 'idle in transaction' AND state_change < NOW() - INTERVAL '5 minutes';"
```

**Prevention**:
- Implement connection pooling properly
- Set connection timeouts
- Monitor connection usage
- Add automatic connection retry
- Use read replicas for scaling

---

### Redis Cache Failures

**Symptoms**:
- "Connection to Redis refused" errors
- Cache misses increasing
- Performance degradation

**Diagnosis**:

```bash
# 1. Check Redis is running
docker-compose ps redis

# 2. Test Redis connection
docker-compose exec redis redis-cli ping
# Expected: PONG

# 3. Check Redis memory usage
docker-compose exec redis redis-cli info memory

# 4. Check eviction stats
docker-compose exec redis redis-cli info stats | grep evicted
```

**Resolution**:

**If Redis down**:
```bash
# Restart Redis
docker-compose restart redis

# Verify persistence
docker-compose exec redis redis-cli BGSAVE
```

**If memory full**:
```bash
# Check memory limit
docker-compose exec redis redis-cli config get maxmemory

# Increase memory limit
docker-compose exec redis redis-cli config set maxmemory 2gb

# OR clear cache
docker-compose exec redis redis-cli FLUSHALL
```

**If persistence issues**:
```bash
# Check AOF status
docker-compose exec redis redis-cli info persistence

# Repair AOF if corrupted
docker-compose exec redis redis-check-aof --fix /data/appendonly.aof
docker-compose restart redis
```

**Prevention**:
- Configure maxmemory-policy: allkeys-lru
- Enable AOF persistence
- Monitor memory usage
- Implement cache warming
- Add Redis Sentinel for HA (future)

---

### Ollama Model Failures

**Symptoms**:
- "Model not found" errors
- Inference timeouts
- Model loading failures

**Diagnosis**:

```bash
# 1. Check Ollama service
docker-compose ps ollama

# 2. List available models
docker-compose exec ollama ollama list

# 3. Check Ollama logs
docker-compose logs ollama | tail -50

# 4. Test model directly
docker-compose exec ollama ollama run llama3.2 "test prompt"
```

**Resolution**:

**If model missing**:
```bash
# Pull model
docker-compose exec ollama ollama pull llama3.2

# Verify model loaded
docker-compose exec ollama ollama list
```

**If model corrupted**:
```bash
# Remove corrupted model
docker-compose exec ollama ollama rm llama3.2

# Re-download
docker-compose exec ollama ollama pull llama3.2
```

**If disk space full**:
```bash
# Check disk usage
df -h

# Remove old/unused models
docker-compose exec ollama ollama rm old-model-name

# Clean Docker cache
docker system prune -a
```

**Prevention**:
- Pre-download models during deployment
- Monitor disk usage
- Implement model versioning
- Add model health checks
- Document required models

---

### API Authentication Failures

**Symptoms**:
- 401 Unauthorized errors increasing
- Valid API keys rejected
- Users cannot authenticate

**Diagnosis**:

```bash
# 1. Check recent auth failures
docker-compose logs api | grep "401" | tail -20

# 2. Verify API key format
echo "<api-key>" | grep -E "^sk-[a-f0-9]{64}$"

# 3. Check database connectivity
docker-compose exec postgres psql -U ollama_user -d ollama -c \
  "SELECT count(*) FROM users;"

# 4. Test authentication manually
curl -H "Authorization: Bearer <api-key>" \
  https://elevatediq.ai/ollama/api/v1/models
```

**Resolution**:

**If database connection issues**:
```bash
# Restart database connection
docker-compose restart api

# Verify database access
docker-compose exec postgres psql -U ollama_user -d ollama -c \
  "SELECT email FROM users LIMIT 5;"
```

**If API key expired/revoked**:
```bash
# Generate new API key for user
docker-compose exec api python -c "
from ollama.services.auth import generate_api_key
print(generate_api_key(prefix='sk'))
"

# Update user record
docker-compose exec postgres psql -U ollama_user -d ollama -c \
  "UPDATE users SET api_key_hash = '<new-hash>' WHERE email = 'user@example.com';"
```

**Prevention**:
- Implement API key rotation
- Add key expiration monitoring
- Log authentication failures
- Implement rate limiting on auth attempts
- Add account lockout after failures

---

### Rate Limit Issues

**Symptoms**:
- 429 Too Many Requests errors
- Legitimate users being blocked
- Alert: "High rate limit rejections"

**Diagnosis**:

```bash
# 1. Check rate limit stats
docker-compose logs api | grep "429" | wc -l

# 2. Identify top requesters
docker-compose logs api | grep "429" | \
  awk '{print $10}' | sort | uniq -c | sort -nr | head -10

# 3. Check Redis rate limit keys
docker-compose exec redis redis-cli keys "ratelimit:*" | wc -l
```

**Resolution**:

**If legitimate traffic spike**:
```bash
# Temporarily increase rate limit
# Edit .env:
# RATE_LIMIT_REQUESTS=200  # Increased from 100
docker-compose restart api

# OR whitelist specific API key
docker-compose exec api python -c "
from ollama.services.rate_limiter import whitelist_key
whitelist_key('sk-trusted-partner-key')
"
```

**If attack/abuse detected**:
```bash
# Block malicious API key
docker-compose exec postgres psql -U ollama_user -d ollama -c \
  "UPDATE users SET is_active = false WHERE api_key_hash = '<hash>';"

# Clear rate limit counters
docker-compose exec redis redis-cli del "ratelimit:<api-key>:*"
```

**Prevention**:
- Implement tiered rate limits
- Add IP-based rate limiting
- Use Cloud Armor for DDoS protection
- Monitor rate limit usage
- Alert on unusual patterns

---

### Out of Memory (OOM)

**Symptoms**:
- Container killed (exit code 137)
- "Out of memory" in logs
- Service restarts frequently

**Diagnosis**:

```bash
# 1. Check memory usage
docker stats --no-stream

# 2. Check container exit codes
docker-compose ps -a | grep "Exited (137)"

# 3. Check system memory
free -h

# 4. Check Docker logs
docker-compose logs api | grep -i "memory"
```

**Resolution**:

**Immediate**:
```bash
# Restart service with more memory
docker-compose down
# Edit docker-compose.yml:
# services:
#   api:
#     mem_limit: 4g  # Increased from 2g
docker-compose up -d
```

**Long-term**:
```bash
# Identify memory leak
docker-compose exec api python -c "
import objgraph
objgraph.show_most_common_types(limit=20)
"

# Profile memory usage
docker-compose exec api memory_profiler app.py

# Reduce memory usage
# - Decrease worker count
# - Reduce cache size
# - Implement pagination
# - Fix memory leaks
```

**Prevention**:
- Set memory limits in docker-compose.yml
- Monitor memory trends
- Implement memory profiling
- Add memory alerts
- Regular leak testing

---

### Disk Space Exhaustion

**Symptoms**:
- "No space left on device" errors
- Container failures
- Log rotation failures

**Diagnosis**:

```bash
# 1. Check disk usage
df -h

# 2. Find large files
du -sh /* | sort -hr | head -10

# 3. Check Docker disk usage
docker system df

# 4. Check log sizes
du -sh /var/lib/docker/containers/*/*-json.log | sort -hr | head -10
```

**Resolution**:

**Immediate**:
```bash
# Clean Docker system
docker system prune -a -f

# Clean logs
truncate -s 0 /var/lib/docker/containers/*/*-json.log

# Remove old models
docker-compose exec ollama ollama rm <old-model>
```

**Long-term**:
```bash
# Configure log rotation
cat > /etc/docker/daemon.json <<EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
EOF
systemctl restart docker

# Add disk monitoring
# Alert if disk > 80%
```

**Prevention**:
- Configure log rotation
- Monitor disk usage
- Automate cleanup scripts
- Set retention policies
- Add disk alerts

---

## Monitoring & Alerting

### Key Metrics to Monitor

**API Metrics**:
- Request rate (requests/second)
- Error rate (% of requests)
- Latency (p50, p95, p99)
- Active connections

**Infrastructure Metrics**:
- CPU usage (%)
- Memory usage (%)
- Disk usage (%)
- Network I/O

**Application Metrics**:
- Model cache hit rate
- Database query time
- Redis response time
- Queue depth

### Alert Thresholds

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| Error rate | > 1% | > 5% | Page on-call |
| Latency (p99) | > 5s | > 10s | Investigate |
| CPU | > 70% | > 90% | Scale up |
| Memory | > 80% | > 95% | Restart/scale |
| Disk | > 80% | > 95% | Clean up |
| Health check | 2 failures | 3 failures | Page on-call |

---

## Routine Maintenance

### Daily Tasks

- [ ] Check alert dashboards
- [ ] Review error logs
- [ ] Monitor resource usage
- [ ] Verify backups completed

### Weekly Tasks

- [ ] Review performance metrics
- [ ] Update dependencies (security patches)
- [ ] Clean up old logs
- [ ] Test rollback procedures

### Monthly Tasks

- [ ] Review and update runbooks
- [ ] Load testing
- [ ] Disaster recovery drill
- [ ] Security audit

---

## Post-Incident Review

### Review Process

**1. Schedule Review** (within 48 hours of incident)

**2. Collect Data**:
- Timeline of events
- Metrics/logs during incident
- Actions taken
- Impact assessment

**3. Conduct Review** (45-60 minutes):
- What happened? (facts)
- Why did it happen? (root cause)
- How did we respond? (actions)
- What went well?
- What could be improved?

**4. Create Action Items**:
- Technical fixes
- Process improvements
- Documentation updates
- Training needs

**5. Follow Up**:
- Track action item completion
- Update runbooks
- Share learnings with team

### Blameless Culture

✅ **Do**:
- Focus on systems and processes
- Ask "why" multiple times
- Document learnings
- Celebrate good incident response

❌ **Don't**:
- Blame individuals
- Skip post-incident review
- Repeat same mistakes
- Hide incidents

---

## Support

- **On-Call**: PagerDuty rotation
- **Team Slack**: #ai-infrastructure
- **Incident Channel**: #incidents
- **Email**: ai-infrastructure@elevatediq.ai
- **Documentation**: https://github.com/kushin77/ollama/docs

---

**Last Updated**: January 18, 2026
**Version**: 0.1.0
**Next Review**: February 18, 2026
