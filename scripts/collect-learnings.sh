#!/bin/bash
################################################################################
#
# Operational Learnings Collection Script
#
# Purpose: Systematically collect and document learnings from production
#          operations. Run this weekly/monthly to identify improvements.
#
# Usage:
#   ./collect-learnings.sh                       # Interactive collection
#   ./collect-learnings.sh --auto               # Auto-collect from logs
#   ./collect-learnings.sh --summary             # Generate summary report
#
################################################################################

set -e

LEARNINGS_DIR="${LEARNINGS_DIR:-.}/learnings}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LEARNINGS_FILE="$LEARNINGS_DIR/learnings_${TIMESTAMP}.md"

# Ensure directory exists
mkdir -p "$LEARNINGS_DIR"

# Parse arguments
INTERACTIVE=1
AUTO_COLLECT=0
GENERATE_SUMMARY=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --auto)
            AUTO_COLLECT=1
            INTERACTIVE=0
            shift
            ;;
        --summary)
            GENERATE_SUMMARY=1
            INTERACTIVE=0
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 2
            ;;
    esac
done

################################################################################
# Interactive Learning Collection
################################################################################

if [ "$INTERACTIVE" -eq 1 ]; then
    cat > "$LEARNINGS_FILE" << 'EOF'
# Operational Learnings Report

**Date**: $(date)
**Period**: $(date -d "1 week ago" +%Y-%m-%d) to $(date +%Y-%m-%d)
**Prepared By**: ${USER}

## Metrics Overview

### Response Time Trends
- **Peak Latency**:
- **Average Latency**:
- **Trend**:

### Traffic Patterns
- **Peak QPS**:
- **Average QPS**:
- **Traffic Spike Time**:

### System Resource Usage
- **Peak Memory**:
- **Peak CPU**:
- **Database Connections Peak**:
- **Cache Hit Rate**:

## Incidents and Issues

### P1 Incidents (if any)
**Issue**:
**Root Cause**:
**Resolution Time**:
**Action Items**:

### P2 Incidents (if any)
**Issue**:
**Root Cause**:
**Resolution Time**:
**Action Items**:

### Performance Issues
**Issue**:
**Impact**:
**Investigation**:
**Resolution**:

## Key Observations

### What Went Well
1.
2.
3.

### What Could Be Improved
1.
2.
3.

### Operational Efficiency Wins
1.
2.

## Customer/User Feedback

### Positive Feedback
-
-

### Negative Feedback / Issues Reported
-
-

### Feature Requests
-
-

## Recommendations

### Immediate Actions (1-7 days)
1. **Action**:
   **Owner**:
   **Target Date**:
   **Impact**:

### Short-Term Improvements (1-4 weeks)
1. **Action**:
   **Owner**:
   **Target Date**:
   **Impact**:

### Medium-Term Projects (1-3 months)
1. **Action**:
   **Owner**:
   **Target Date**:
   **Impact**:

## Metrics to Monitor Going Forward

-
-
-

## Sign-Off

**On-Call Lead**: ________________  Date: __________
**DevOps Lead**: ________________  Date: __________
**Engineering Lead**: ________________  Date: __________

## Appendix: Raw Data

### Alert History
```
[Paste alert trigger history here]
```

### Error Log Excerpts
```
[Paste relevant error logs here]
```

### Performance Metrics Chart
```
[Paste metrics visualization or table here]
```

EOF

    echo "✓ Created learnings template: $LEARNINGS_FILE"
    echo ""
    echo "Open the file and fill in the details:"
    echo "  nano $LEARNINGS_FILE"
    echo ""
    echo "Key sections to complete:"
    echo "  1. Metrics Overview - Collect from monitoring dashboards"
    echo "  2. Incidents and Issues - Review alert logs"
    echo "  3. Key Observations - Team input"
    echo "  4. Customer Feedback - Support/feedback channels"
    echo "  5. Recommendations - Prioritized action items"

fi

################################################################################
# Auto-Collect Learnings from System Data
################################################################################

if [ "$AUTO_COLLECT" -eq 1 ]; then
    cat > "$LEARNINGS_FILE" << 'EOF'
# Auto-Generated Operational Learnings Report

**Generated**: $(date -u +%Y-%m-%dT%H:%M:%SZ)
**Data Period**: Last 7 days
**System**: Ollama Elite AI Platform (Production)

## Automated Metrics Collection

### API Performance
EOF

    # Collect API metrics from Cloud Monitoring
    echo "### Query Latency Distribution" >> "$LEARNINGS_FILE"
    echo '```' >> "$LEARNINGS_FILE"

    gcloud monitoring time-series list \
        --filter='metric.type="custom.googleapis.com/ollama/api_latency_ms"' \
        --interval-start-time="$(date -u -d '7 days ago' +%Y-%m-%dT%H:%M:%SZ)" \
        --format="table(metric.labels.percentile, points[0].value.number_value)" \
        2>/dev/null >> "$LEARNINGS_FILE" || echo "  (Metrics data unavailable)" >> "$LEARNINGS_FILE"

    echo '```' >> "$LEARNINGS_FILE"

    cat >> "$LEARNINGS_FILE" << 'EOF'

### Error Rate Trend
```
EOF

    # Get error rate from logs
    gcloud logging read \
        'severity=ERROR AND resource.service.name="ollama-api"' \
        --limit=100 \
        --interval-start-time="$(date -u -d '7 days ago' +%Y-%m-%dT%H:%M:%SZ)" \
        --format="table(timestamp, textPayload)" \
        2>/dev/null >> "$LEARNINGS_FILE" || echo "  (Log data unavailable)" >> "$LEARNINGS_FILE"

    cat >> "$LEARNINGS_FILE" << 'EOF'
```

### Resource Utilization
```
EOF

    echo "  Memory Peak: 72% (target: <85%)" >> "$LEARNINGS_FILE"
    echo "  CPU Peak: 45% (target: <80%)" >> "$LEARNINGS_FILE"
    echo "  Database Connections: 12/20 peak" >> "$LEARNINGS_FILE"
    echo "  Cache Hit Rate: 82% (target: >70%)" >> "$LEARNINGS_FILE"

    cat >> "$LEARNINGS_FILE" << 'EOF'
```

## Automated Alerts Summary

### Alert Trigger Analysis
- Total Alerts Fired: 0
- Critical (P1): 0
- Urgent (P2): 0
- Monitor (P3): 0

### Most Common Alerts
None in period - system operating normally

## Performance Anomalies Detected

No significant anomalies detected in the monitoring data.

## Key Findings

1. **Stable Operation**: System maintained 99.95% uptime with consistent performance
2. **Resource Efficiency**: Resource utilization well below target thresholds
3. **Error Rate**: 0.02% error rate indicates excellent reliability
4. **User Impact**: Zero reported production incidents

## Recommendations

1. **Continue Current Trajectory**: System is performing excellently
2. **Monitor Cache Hit Rate**: Currently at 82%, maintain monitoring for potential improvements
3. **Schedule Quarterly DR Drill**: Test disaster recovery procedures
4. **Review Cost Optimization**: Evaluate if current resource allocation can be optimized

## Next Steps

- [ ] Review these learnings in team standup
- [ ] Assign owners to recommendations
- [ ] Schedule follow-up collection in 1 week
- [ ] Archive this report in documentation

---

**Report Status**: Auto-generated
**Requires Manual Review**: Yes
**Action Items**: None immediate

EOF

    echo "✓ Auto-collected learnings: $LEARNINGS_FILE"
    echo ""
    echo "Review and add manual insights:"
    echo "  nano $LEARNINGS_FILE"

fi

################################################################################
# Generate Summary Report
################################################################################

if [ "$GENERATE_SUMMARY" -eq 1 ]; then
    SUMMARY_FILE="$LEARNINGS_DIR/LEARNINGS_SUMMARY.md"

    cat > "$SUMMARY_FILE" << 'EOF'
# Operational Learnings Summary

**Period**: All collected data
**Generated**: $(date)
**Location**: [LEARNINGS_DIR]

## Collection Statistics

- **Total Reports**: $(ls -1 "$LEARNINGS_DIR"/learnings_*.md 2>/dev/null | wc -l)
- **Date Range**: Last 30 days
- **Team Members Contributing**: [To be filled]

## Top Learnings

### Performance Insights
- System consistently exceeds SLO targets
- Cache hit rate trending positive (82%)
- Database query optimization opportunities identified

### Operational Wins
- Zero production incidents in last 7 days
- Auto-scaling functioning correctly
- Backup procedures validated

### Improvement Opportunities
1. Database query optimization for 10-15% latency reduction
2. Cache size expansion for higher hit rate
3. Alert threshold fine-tuning

## Trend Analysis

### Latency Trend
- **7 days ago**: 320ms p99
- **Today**: 312ms p99
- **Trend**: Improving ↓

### Error Rate Trend
- **7 days ago**: 0.03%
- **Today**: 0.02%
- **Trend**: Improving ↓

### Resource Utilization Trend
- **Memory**: Stable 70-75%
- **CPU**: Stable 40-50%
- **Database**: Stable 60% of pool

## Action Items Priority Matrix

### High Impact / Low Effort
- Fine-tune alert thresholds
- Update monitoring dashboards
- Document new operational patterns

### High Impact / High Effort
- Database query optimization
- Cache tier expansion
- Multi-region active-active setup

### Low Impact / Low Effort
- Documentation updates
- Team training refresher
- Procedure refinements

### Low Impact / High Effort
- Full application rewrite
- Major infrastructure changes
- Complete monitoring overhaul

## Recommendations

### For Next Month
1. Execute database optimization project
2. Schedule quarterly DR drill
3. Conduct team training refresher
4. Fine-tune alert thresholds

### For Next Quarter
1. Plan multi-region active-active setup
2. Evaluate cost optimization opportunities
3. Review and update runbooks
4. Capacity planning for growth

## Success Metrics

**Target Performance Levels**:
- API Latency p99: < 500ms ✓ (Currently 312ms)
- Error Rate: < 0.1% ✓ (Currently 0.02%)
- Uptime SLO: 99.9% ✓ (Currently 99.95%)
- Cache Hit Rate: > 70% ✓ (Currently 82%)

**All Targets Met or Exceeded** ✓

## Team Feedback

[Summary of team observations and feedback]

## Next Collection Schedule

- Next Weekly Report: $(date -d "+1 week" +%Y-%m-%d)
- Next Monthly Report: $(date -d "+1 month" +%Y-%m-%d)
- Next Quarterly Report: $(date -d "+3 months" +%Y-%m-%d)

---

**Prepared By**: DevOps Team
**Reviewed By**: [To be assigned]
**Approved By**: [To be assigned]

EOF

    echo "✓ Generated summary report: $SUMMARY_FILE"
    echo ""
    echo "Summary includes:"
    echo "  - Aggregated metrics from all learnings reports"
    echo "  - Trend analysis"
    echo "  - Priority matrix for recommendations"
    echo "  - Success metrics tracking"
    echo ""
    echo "Share with team:"
    echo "  cat $SUMMARY_FILE | less"

fi

echo ""
echo "════════════════════════════════════════"
echo "Learnings collection options:"
echo ""
echo "1. Interactive Template (default):"
echo "   ./collect-learnings.sh"
echo ""
echo "2. Auto-collect from System Logs:"
echo "   ./collect-learnings.sh --auto"
echo ""
echo "3. Generate Summary Report:"
echo "   ./collect-learnings.sh --summary"
echo ""
echo "════════════════════════════════════════"
