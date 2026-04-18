# Weekly Metrics Dashboard & Compliance Tracking

**Version**: 1.0
**Status**: MANDATORY for production operations
**Last Updated**: 2026-01-26

---

## Overview

The Weekly Metrics Dashboard provides comprehensive tracking of agent performance, system health, and compliance metrics. Metrics are collected continuously and reviewed weekly with automated kill signal detection.

## Metrics Categories

### Agent Performance Metrics

**Hallucination Rate**

- Metric: Percentage of outputs that are factually incorrect or contradictory
- Threshold: <2% on critical actions
- Collection: From `tests/agents/hallucination_detection.py` validation
- Review Frequency: Weekly
- Kill Signal: ≥2% for 2 consecutive weeks

**Action Accuracy**

- Metric: Percentage of remediation actions that are correct and safe
- Threshold: >95%
- Collection: From red-team simulation suite results
- Review Frequency: Weekly
- Kill Signal: <95% for 2 consecutive weeks

**Response Time (P95 Latency)**

- Metric: 95th percentile response latency in milliseconds
- Threshold: <30s for triage, <5min for complex investigations
- Collection: From production request logging
- Review Frequency: Weekly
- Kill Signal: >5min for complex tasks (immediate investigation)

**Human Override Rate**

- Metric: Percentage of agent actions requiring human review/override
- Threshold: <10% for medium severity, <30% for critical severity
- Collection: From post-deployment telemetry
- Review Frequency: Weekly
- Kill Signal: >30% for critical actions for 2 consecutive weeks

### System Health Metrics

**Availability**

- Metric: Percentage of time agents are available and responsive
- Threshold: >99.5%
- Collection: From health check monitoring
- Review Frequency: Daily (trending weekly)

**Error Rate**

- Metric: Percentage of requests resulting in errors
- Threshold: <1%
- Collection: From application logs
- Review Frequency: Daily (trending weekly)

**Cost Per Inference**

- Metric: Average cost per agent action execution
- Threshold: Track trending, alert on 20%+ increase
- Collection: From GCP billing integration
- Review Frequency: Weekly

### Compliance Metrics

**Audit Log Entries**

- Metric: Total audit log entries recorded
- Threshold: Track trending
- Collection: From Chronicle logs
- Review Frequency: Weekly

**Security Violations**

- Metric: Number of detected security policy violations
- Threshold: Zero tolerance for critical violations
- Collection: From security scanning
- Review Frequency: Daily (immediate escalation on critical)

**Data Privacy Incidents**

- Metric: Number of potential PII exposures or data leaks
- Threshold: Zero tolerance
- Collection: From data loss prevention tools
- Review Frequency: Continuous (immediate escalation)

## Metrics Collection

### Architecture

```
Agent Execution
    ↓
Metrics Instrumentation (ollama/monitoring/metrics.py)
    ↓
Time-Series Database (BigQuery)
    ↓
Weekly Aggregation
    ↓
Weekly Report (metrics/weekly_review.ipynb)
    ↓
Slack Notification + Dashboard Update
```

### Collection Methods

**1. Agent Quality Metrics**

- Source: `tests/agents/` test suite execution
- Frequency: On each agent update/deployment
- Storage: BigQuery table `ollama_metrics.agent_quality`

**2. Performance Metrics**

- Source: Production request logging
- Collection: Prometheus exporters
- Frequency: Real-time collection, hourly aggregation
- Storage: BigQuery table `ollama_metrics.performance`

**3. Business Metrics**

- Source: Customer success platform + internal dashboards
- Frequency: Daily
- Storage: BigQuery table `ollama_metrics.business`

**4. Security & Compliance Metrics**

- Source: GCP Cloud SCC, Chronicle logs, Wiz scans
- Frequency: Continuous (daily summary)
- Storage: BigQuery table `ollama_metrics.security`

## Weekly Review Process

### Schedule

**Every Friday 3 PM UTC**:

1. **Run Aggregation** (automatically triggered)

   ```bash
   python -m ollama.monitoring.weekly_review --week=$(date +%V)
   ```

2. **Generate Report** (automatic)
   - Reads metrics from BigQuery
   - Calculates aggregates
   - Checks kill signals
   - Generates recommendations

3. **Review Meeting** (4 PM UTC, 30 minutes)
   - Engineering lead presents findings
   - Discuss any kill signals
   - Agree on action items
   - Update sprint/roadmap if needed

4. **Distribution**
   - Slack notification to #metrics channel
   - Email to engineering team
   - Link posted to wiki
   - Add to team knowledge base

### Report Format

The weekly report includes:

```python
{
    "report_date": "2026-01-31T15:00:00Z",
    "week_of": "2026-01-27T00:00:00Z",

    "summary": {
        "total_agents": 5,
        "agents_meeting_quality_bar": 4,
        "agents_needing_attention": [
            {
                "agent_id": "security-scanner-v2",
                "reason": "accuracy 92% <= 95%"
            }
        ]
    },

    "agents": [
        {
            "agent_id": "threat-detector-v1",
            "agent_name": "Threat Detection Agent",
            "agent_type": "security",
            "quality_metrics": {
                "hallucination_rate": "0.8%",
                "action_accuracy": "97.2%",
                "average_latency_ms": "2450",
                "p95_latency_ms": "5800",
                "p99_latency_ms": "8200",
                "human_override_rate": "8.5%"
            },
            "execution_metrics": {
                "total_actions": 15234,
                "successful_actions": 14890,
                "failed_actions": 344,
                "success_rate": "97.74%"
            },
            "compliance_metrics": {
                "violations": 0,
                "audit_log_entries": 47582
            },
            "meets_quality_bar": true
        }
    ],

    "analysis": {
        "quality_trend": {
            "threat-detector-v1": {
                "quality_score": 94.5,
                "meets_bar": true
            }
        },
        "alerts": [
            {
                "severity": "warning",
                "agent_id": "security-scanner-v2",
                "message": "Accuracy 92% approaching 95% threshold"
            }
        ]
    },

    "kill_signals": {
        "has_signals": false,
        "signals": []
    },

    "recommendations": [
        {
            "agent_id": "security-scanner-v2",
            "agent_name": "Security Scanner",
            "type": "training",
            "priority": "high",
            "action": "Add domain-specific training and adversarial scenarios",
            "expected_impact": "Improve accuracy to >97%"
        }
    ]
}
```

## Kill Signals & Escalation

### Automatic Detection

Kill signals are automatically detected and escalated:

| Signal             | Threshold        | Detection     | Action                      |
| ------------------ | ---------------- | ------------- | --------------------------- |
| High Hallucination | ≥2% for 2 weeks  | Continuous    | Escalate to #agents-quality |
| Low Accuracy       | <95% for 2 weeks | Weekly review | Archive agent after 2 weeks |
| High Latency       | >5min P95        | Immediate     | Investigate bottleneck NOW  |
| High Override Rate | >30% for 2 weeks | Weekly review | Retrain or archive          |

### Escalation Flow

**Level 1 - Team Alert** (automatic):

```
Kill signal detected → Slack notification to #agents-quality → Agent owner assigned
```

**Level 2 - Engineering Lead Review** (24 hours):

```
If not resolved → Assigned to engineering lead → Root cause analysis
```

**Level 3 - CTO Escalation** (48 hours):

```
If still not resolved → Escalated to @cto → Emergency decision (retrain/archive/investigate)
```

**Level 4 - Deployment Block** (72 hours):

```
If kill signal persists → Agent blocked from production → Forced remediation or archival
```

## Implementation

### Code Location

```
ollama/monitoring/
├── __init__.py
├── metrics.py                 # MetricsCollector class
└── weekly_review.py          # Report generation and analysis
```

### Usage Examples

**Collect Metrics**:

```python
from ollama.monitoring.metrics import MetricsCollector

collector = MetricsCollector()

# Record individual metrics
collector.record_metric(
    agent_id="threat-detector-v1",
    metric_name="hallucination_rate",
    value=0.008,
    tags={"severity": "critical"}
)

# Aggregate metrics
metrics = collector.aggregate_metrics(
    agent_id="threat-detector-v1",
    agent_name="Threat Detection Agent",
    agent_type="security"
)
```

**Generate Weekly Report**:

```python
from ollama.monitoring.weekly_review import generate_weekly_report

report = generate_weekly_report(collector)

# Check for kill signals
if report["kill_signals"]["has_signals"]:
    print("⚠️ Kill signals detected!")
    for signal in report["kill_signals"]["signals"]:
        print(f"  - {signal['type']}: {signal['agent_name']}")

# Export to JSON
collector.export_to_json("metrics/weekly_report_2026_04.json")
```

## Dashboard

### Grafana Dashboard

**Name**: Ollama Agent Quality Dashboard
**URL**: `https://grafana.example.com/d/agent-quality`

**Panels**:

- Agent Hallucination Rate (by agent, by severity)
- Action Accuracy Trend (week-over-week)
- P95 Latency Distribution
- Human Override Rate (by severity)
- Quality Score Heatmap (all agents)
- Kill Signals (real-time alerts)

### BigQuery Dashboards

**Tables**:

- `ollama_metrics.agent_quality` - Aggregated agent metrics
- `ollama_metrics.performance` - Latency and throughput metrics
- `ollama_metrics.business` - Customer-facing metrics
- `ollama_metrics.security` - Security and compliance metrics

## Monitoring & Alerting

### Prometheus Metrics

Exported metrics (for Prometheus scraping):

```
# HELP ollama_agent_hallucination_rate Percentage of hallucinated outputs
# TYPE ollama_agent_hallucination_rate gauge
ollama_agent_hallucination_rate{agent_id="threat-detector-v1"} 0.008

# HELP ollama_agent_accuracy Percentage of correct remediation actions
# TYPE ollama_agent_accuracy gauge
ollama_agent_accuracy{agent_id="threat-detector-v1"} 0.972

# HELP ollama_agent_latency_p95_ms P95 response latency in milliseconds
# TYPE ollama_agent_latency_p95_ms gauge
ollama_agent_latency_p95_ms{agent_id="threat-detector-v1"} 5800

# HELP ollama_agent_override_rate Percentage of human overrides
# TYPE ollama_agent_override_rate gauge
ollama_agent_override_rate{agent_id="threat-detector-v1"} 0.085
```

### Alerting Rules

```yaml
# alert_rules.yaml
groups:
  - name: agent_quality
    rules:
      - alert: AgentHallucinationHigh
        expr: ollama_agent_hallucination_rate > 0.02
        for: 1h
        annotations:
          summary: "Agent {{ $labels.agent_id }} hallucination rate >2%"

      - alert: AgentLatencyHigh
        expr: ollama_agent_latency_p95_ms > 300000
        for: 5m
        annotations:
          summary: "Agent {{ $labels.agent_id }} P95 latency >5min"

      - alert: AgentAccuracyLow
        expr: ollama_agent_accuracy < 0.95
        for: 1h
        annotations:
          summary: "Agent {{ $labels.agent_id }} accuracy <95%"

      - alert: AgentOverrideRateHigh
        expr: ollama_agent_override_rate > 0.30
        for: 1h
        annotations:
          summary: "Agent {{ $labels.agent_id }} override rate >30%"
```

## Related Documentation

- [Agent Quality Standards](./agent-quality-standards.md)
- [Elite Execution Protocol](../copilot-instructions.md#velocity--quality-metrics-tracked-weekly)
- [Knowledge Management](./knowledge-management.md)

---

**Maintained By**: Metrics & Observability Team
**Last Updated**: 2026-01-26
**Next Review**: 2026-02-26 (Monthly)
