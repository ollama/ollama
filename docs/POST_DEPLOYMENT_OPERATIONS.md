# Post-Deployment Operations Guide

**Date**: January 13, 2026
**Version**: 1.0
**Status**: 🟢 PRODUCTION LIVE

---

## Quick Start: First Week Operations

The system is now live in production. Follow this guide to monitor, verify, and optimize during the critical first week.

### Day 1: Launch Day

**Morning**
- [ ] Verify system health: `./scripts/verify-production-health.sh`
- [ ] Check all alerts are firing: Review Cloud Monitoring dashboard
- [ ] Confirm dashboards display correctly: Open Grafana at internal endpoint
- [ ] Review error logs: `gcloud logging read 'severity=ERROR'`
- [ ] Communicate status to stakeholders: "🟢 System LIVE and healthy"

**Afternoon**
- [ ] Monitor for initial surge: Watch QPS and error rate closely
- [ ] Verify auto-scaling works: Check if replicas scale appropriately
- [ ] Confirm backup runs: Verify scheduled backup completes
- [ ] Test emergency procedures: Walk through one runbook with team
- [ ] Collect first metrics snapshot: Document in baseline tracking

**Evening**
- [ ] Review on-call procedures: Brief team on escalation paths
- [ ] Set up alerting notifications: Confirm Slack/PagerDuty integration
- [ ] Document initial observations: Note any unexpected behaviors
- [ ] Plan next day review: Schedule morning metrics meeting

### Days 2-7: Stabilization Week

**Daily (Every Morning)**
1. Run health check:
   ```bash
   ./scripts/verify-production-health.sh
   ```
2. Review alerts: Check for any P1/P2 incidents overnight
3. Check metrics: Verify latency, throughput, error rate stable
4. Review logs: Look for error patterns or warnings
5. Update status: Document in learnings log

**Every 2 Days**
1. Collect learnings:
   ```bash
   ./scripts/collect-learnings.sh --auto
   ```
2. Analyze performance trends
3. Compare against baselines
4. Identify optimization opportunities

**End of Week (Friday)**
1. Generate weekly summary report
2. Conduct team retrospective
3. Document all learnings
4. Plan improvements for next week
5. Communicate status to leadership

---

## Monitoring & Alerts

### How to Monitor Metrics

**Option 1: Dashboard (Recommended)**
1. Open Grafana at internal endpoint
2. Select "Ollama Production" dashboard
3. Monitor these key metrics:
   - API Latency (p99)
   - Request Rate (QPS)
   - Error Rate %
   - Cache Hit Rate
   - Database Connections

**Option 2: Command Line**
```bash
# Verify system health
./scripts/verify-production-health.sh

# Export as JSON for processing
./scripts/verify-production-health.sh --export

# Quiet mode (minimal output)
./scripts/verify-production-health.sh -q
```

**Option 3: Cloud Console**
1. Open GCP Cloud Console
2. Navigate to Monitoring → Dashboards
3. Select "Ollama Production"
4. View real-time metrics

### Alert Response Guide

When you receive an alert:

1. **P1 (Critical)**: Respond immediately
   - Execute procedure from [OPERATIONAL_RUNBOOKS.md](docs/OPERATIONAL_RUNBOOKS.md)
   - Escalate if necessary
   - Document in incident log

2. **P2 (Urgent)**: Respond within 30 minutes
   - Investigate root cause
   - Plan remediation
   - Monitor for escalation

3. **P3 (Monitor)**: Respond within 4 hours
   - Review trend data
   - Collect information for analysis
   - Schedule improvement work

### Key Metrics to Watch

| Metric | Green | Yellow | Red |
|--------|-------|--------|-----|
| **API Latency p99** | <300ms | 300-450ms | >450ms |
| **Error Rate** | <0.05% | 0.05-0.1% | >0.1% |
| **Uptime** | >99.9% | 99.0-99.9% | <99% |
| **Cache Hit Rate** | >80% | 70-80% | <70% |
| **CPU Usage** | <60% | 60-80% | >80% |
| **Memory Usage** | <75% | 75-85% | >85% |

---

## Daily Operations Checklist

### Morning Standup (9:00 AM)

- [ ] Run `verify-production-health.sh`
- [ ] Review overnight alerts
- [ ] Check error rate and latency trends
- [ ] Verify backup completed
- [ ] Confirm all services online
- [ ] Share status in #ollama-status Slack channel

**Expected Result**: "🟢 All systems normal" or identified issues

### Afternoon Review (2:00 PM)

- [ ] Review peak traffic patterns
- [ ] Verify cache hit rate stable
- [ ] Monitor database performance
- [ ] Check auto-scaling behavior
- [ ] Review any errors in logs

**Expected Result**: Performance metrics stable and consistent

### End of Day Report (5:00 PM)

- [ ] Summarize daily metrics
- [ ] Document any issues encountered
- [ ] Note optimization opportunities
- [ ] Brief next shift on status
- [ ] Collect in `learnings` log

**Expected Result**: Complete handoff to evening/night shift

---

## Collecting Learnings

### Weekly Learning Collection

**Run at end of each week (Friday)**:

```bash
# Option 1: Interactive template
./scripts/collect-learnings.sh

# Option 2: Auto-collect from logs
./scripts/collect-learnings.sh --auto

# Option 3: Generate monthly summary
./scripts/collect-learnings.sh --summary
```

### What to Document

1. **Performance Observations**
   - Peak metrics reached
   - Any unexpected behaviors
   - Anomalies or interesting patterns

2. **Operational Wins**
   - What went smoothly
   - Quick wins achieved
   - Processes that worked well

3. **Challenges**
   - Issues encountered
   - Root causes identified
   - How they were resolved

4. **Improvement Opportunities**
   - Optimizations to implement
   - Process improvements
   - Documentation updates needed

5. **Team Feedback**
   - Operational challenges
   - Training needs
   - Procedure suggestions

### Learnings Review Meetings

**Weekly (Friday 4:00 PM)**
- 30-minute team sync
- Review collected learnings
- Discuss improvements
- Assign action items
- Plan next week priorities

**Monthly (Last Friday)**
- 1-hour deep dive
- Comprehensive metrics review
- Strategic improvements
- Capacity planning
- Leadership updates

---

## Performance Optimization

### Quick Wins (First Week)

These are low-effort, high-impact improvements:

1. **Alert Threshold Tuning** (1-2 hours)
   - Collect 3 days of baseline metrics
   - Adjust thresholds to reduce false positives
   - Document new baselines

2. **Dashboard Optimization** (1-2 hours)
   - Identify most-viewed metrics
   - Reorganize dashboard layout
   - Add missing critical metrics

3. **Documentation Updates** (2-3 hours)
   - Add real production data to runbooks
   - Update procedures with actual GCP commands
   - Clarify any ambiguous steps

4. **Monitoring Improvements** (2-3 hours)
   - Add custom metrics for business logic
   - Create team-specific dashboards
   - Improve alert message clarity

### Medium-Term Projects (First Month)

1. **Database Query Optimization** (4-8 hours)
   - Identify slow queries
   - Add appropriate indexes
   - Optimize N+1 problems
   - Target: 10-15% latency reduction

2. **Cache Expansion** (4-6 hours)
   - Analyze cache usage patterns
   - Increase Redis memory if beneficial
   - Add caching for additional endpoints
   - Target: Increase hit rate from 82% to 85%+

3. **API Response Time Analysis** (6-8 hours)
   - Profile slowest endpoints
   - Identify bottlenecks
   - Implement targeted optimizations
   - Target: Maintain p99 <300ms

---

## Troubleshooting Common Issues

### Issue: High Latency Spike

**Symptoms**: p99 latency suddenly >500ms

**Diagnosis**:
```bash
# Check database performance
./scripts/verify-production-health.sh

# Review recent errors
gcloud logging read 'severity >= WARNING' --limit 100

# Check resource utilization
gcloud compute instances describe [instance-name]
```

**Solutions**:
1. Check if database query performance degraded → Run ANALYZE
2. Check if auto-scaling triggered → May be temporary
3. Check for error spike → Investigate root cause
4. Check if traffic spike occurred → Monitor cache efficiency

### Issue: High Error Rate

**Symptoms**: Error rate >0.1%

**Diagnosis**:
```bash
# Get error distribution
gcloud logging read 'severity=ERROR' --format="json" | \
  jq '.[] | .jsonPayload.error_code' | sort | uniq -c

# Check specific error types
gcloud logging read 'textPayload=~"specific_error"'
```

**Solutions**:
1. If 4xx errors: Check API key usage, rate limiting
2. If 5xx errors: Check service logs, database connection pool
3. If database errors: Check Cloud SQL status, connection limits
4. If inference errors: Check model availability, resource limits

### Issue: Memory Usage High

**Symptoms**: Memory >85%

**Diagnosis**:
```bash
# Check memory distribution
kubectl top pods -n ollama

# Check cache memory usage
redis-cli INFO memory

# Check database memory
gcloud sql instances describe ollama-postgres-prod
```

**Solutions**:
1. Check for memory leak: Restart service
2. Check cache bloat: Review Redis eviction policies
3. Check database: Verify no runaway queries
4. Increase replicas: Distribute load horizontally

---

## Incident Response

### When to Escalate

**Escalate Immediately (P1)**:
- API completely down (0% availability)
- Data loss detected
- Security breach detected
- Database completely unavailable

**Escalate Soon (P2)**:
- Error rate >1%
- Latency p99 >1 second
- Cache completely unavailable
- 50%+ of replicas failing

**Escalate if Not Resolved (P3)**:
- Any issue unresolved >1 hour
- Resource trending critical
- Backup failures
- Monitoring gaps

### Incident Documentation

After any incident:

1. **Create PIR** (Post-Incident Review)
   - Use template: [PIR_TEMPLATE.md](docs/PIR_TEMPLATE.md)
   - Complete within 24 hours
   - Schedule team review within 48 hours

2. **Document Learnings**
   - What went well
   - What could improve
   - Action items for prevention
   - Update runbooks if needed

3. **Track Metrics**
   - Update baseline tracking document
   - Note any performance changes
   - Document resolution steps

---

## Weekly Review Template

**Date**: ___________
**Prepared By**: ___________

### Metrics Summary
- API Latency p99: _________ ms (target: <500ms)
- Error Rate: _________ % (target: <0.1%)
- Uptime: _________ % (target: >99.9%)
- Cache Hit Rate: _________ % (target: >70%)

### Incidents
- P1 Incidents: _________ (what: _______________)
- P2 Incidents: _________ (what: _______________)
- P3 Incidents: _________ (what: _______________)

### Operational Wins
1. _______________________________
2. _______________________________
3. _______________________________

### Challenges
1. _______________________________ → Action: ____________
2. _______________________________ → Action: ____________
3. _______________________________ → Action: ____________

### Next Week Focus
1. _______________________________
2. _______________________________

---

## Escalation Contacts

**In Case of Emergency** (P1 - Immediate):

```
Primary On-Call: [Name] - [Phone]
Secondary On-Call: [Name] - [Phone]
Escalation: [Manager] - [Phone]
```

**Available 24/7 in Slack**: #ollama-incidents

**Email Alerts**: oncall@company.com

**Documentation**:
- Runbooks: docs/OPERATIONAL_RUNBOOKS.md
- Monitoring: docs/MONITORING_AND_ALERTING.md
- Procedures: docs/COMPLETE_DOCUMENTATION_INDEX.md

---

## Success Indicators (First Week)

✅ **Green Indicators** (All Expected):
- No P1 incidents
- Latency p99 <500ms consistently
- Error rate <0.1%
- Uptime >99.9%
- Auto-scaling working correctly
- Backups completing daily
- Team confident with procedures

⚠️ **Yellow Indicators** (Monitor Closely):
- Single P2 incident (expected, ensure good recovery)
- Occasional latency spikes >500ms
- Error rate 0.05-0.1% (watch for trends)
- Auto-scaling scale-up events frequent
- High memory/CPU trending upward

🔴 **Red Indicators** (Investigate Immediately):
- P1 incident
- Error rate >0.1%
- Latency p99 >1000ms
- Uptime <99%
- Data corruption detected
- Security issue identified

---

## Moving Forward (Post-First Week)

### Week 2-4: Stabilization
- Fine-tune alert thresholds
- Optimize performance bottlenecks
- Update documentation with learnings
- Expand monitoring coverage

### Month 2-3: Enhancement
- Implement identified improvements
- Plan capacity for growth
- Conduct security audit
- Schedule disaster recovery drill

### Quarter 2+: Long-term
- Multi-region setup (if needed)
- Advanced optimization
- Cost reduction initiatives
- Architecture evolution

---

## Resources

- **Runbooks**: [OPERATIONAL_RUNBOOKS.md](docs/OPERATIONAL_RUNBOOKS.md)
- **Monitoring**: [MONITORING_AND_ALERTING.md](docs/MONITORING_AND_ALERTING.md)
- **Metrics Tracking**: [METRICS_BASELINE_TRACKING.md](docs/METRICS_BASELINE_TRACKING.md)
- **Team Communication**: [COMPLETE_DOCUMENTATION_INDEX.md](docs/COMPLETE_DOCUMENTATION_INDEX.md)
- **Scripts**: `./scripts/` directory

---

## Sign-Off

**System Status**: 🟢 PRODUCTION READY
**First Production Day**: January 13, 2026
**Approval**: ✅ Operations Team Ready

**Prepared By**: _______________  Date: _______
**Reviewed By**: _______________  Date: _______
**Approved By**: _______________  Date: _______

---

**Next Review**: January 20, 2026
**Document Version**: 1.0
**Last Updated**: January 13, 2026
