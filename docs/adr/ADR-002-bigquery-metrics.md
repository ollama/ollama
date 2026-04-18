# ADR-002: BigQuery for Metrics Aggregation

**Status**: Accepted
**Date**: 2026-01-26
**Author**: @data-team

---

## Context

### Problem

We need to aggregate and analyze metrics from agent inference service (100+ req/s). Requirements:

- Store 7 years of metrics for compliance
- Query 1B+ rows efficiently
- Weekly aggregation reports
- Cost-effective at scale (1 year = 10B+ rows)

### Constraints

- Must be on GCP (landing zone)
- Weekly aggregation must complete in < 10 minutes
- Cost must scale sublinearly with data volume

---

## Decision

**Chosen**: Google BigQuery (data warehouse)

BigQuery is the optimal choice because:

1. **Query Performance**: SQL queries on 1B+ rows complete in < 10 seconds
2. **Cost Efficiency**: $6.25/TB of data scanned (only pay for data scanned)
3. **7-Year Retention**: Can store 10B+ rows for fraction of cost of traditional DB
4. **Built-in Analytics**: WINDOW functions, ML integration for future forecasting
5. **GCP Native**: Seamless integration with Cloud Run, Cloud Logging

---

## Consequences

### Positive

1. **Exceptional Query Performance**: 1B rows queried in < 10 seconds
2. **Cost Efficiency**: $0.006/GB for storage vs. $50-100/GB for traditional databases
3. **Built-in ML**: Can add predictions/forecasting later without new infrastructure
4. **Compliance-Ready**: Native support for 7-year retention, audit logging

### Negative

1. **Not Real-Time**: Slight 5-10 minute ingestion delay (acceptable for weekly reports)
2. **Immutable Data**: Can't update/delete historical data (not a problem for metrics)
3. **Query Costs**: If team runs many exploratory queries, costs can spike
4. **Requires SQL Knowledge**: Team must learn BigQuery SQL syntax

---

## Alternatives Considered

### Alternative A: PostgreSQL with Time-Series Extension (TimescaleDB)

**Pros**:

- Team familiar with PostgreSQL
- Can update/delete data if needed
- Lower operational complexity

**Cons**:

- 10B rows = $50k+/month in storage
- Query performance degrades as data grows
- Requires manual partitioning management

**Why Not Chosen**: 10x cost of BigQuery.

---

### Alternative B: Google Cloud SQL for Time-Series

**Pros**:

- Fully managed like BigQuery
- Native PostgreSQL support

**Cons**:

- Not optimized for analytics (optimized for transactional workloads)
- Cost scales linearly with data volume
- Query performance poor on large datasets

**Why Not Chosen**: Poor economics at 10B+ row scale.

---

## Implementation

### Steps

1. **Create BigQuery Dataset**: `ollama_metrics`
2. **Define Schema**: Tables for `agent_metrics`, `performance`, `business`, `security`
3. **Setup Data Pipeline**: Cloud Logging → Cloud Dataflow → BigQuery
4. **Create Weekly Job**: Cloud Scheduler → Cloud Function → BigQuery aggregation
5. **Setup Grafana**: Connect BigQuery as Grafana data source

### Success Criteria

- ✅ 30-day baseline metrics collected
- ✅ Weekly aggregation completes in < 10 minutes
- ✅ Cost tracking dashboard showing $$ vs. baseline
- ✅ Team able to run ad-hoc queries for analysis

---

## Cost Analysis

### Cost Model

- Ingestion: $6/TB = $0.006/GB
- Storage: $0.02/GB/month (long-term storage)
- Queries: $6.25/TB scanned

### Projected Monthly Cost

- Current traffic: 100 req/s
- Metrics: ~1KB per request
- Daily ingestion: 8.6 GB
- Monthly ingestion: 260 GB
- Monthly query cost: $1.60 (estimated)
- Monthly storage: $5.20
- **Total**: ~$6.80/month (negligible)

### At Scale (1000 req/s)

- Monthly ingestion: 2.6 TB
- Monthly cost: ~$20
- **Still negligible**

---

## Monitoring

### Key Metrics

- Query completion time (should be < 10 min for weekly job)
- Data ingestion latency (should be < 10 min)
- Cost per GB (should be < $1/GB total cost)

### Review Schedule

- Monthly cost review
- Quarterly query optimization review

---

## Related Decisions

- ADR-001: Cloud Run for orchestration
- Issue #13: Weekly Metrics Dashboard

---

**Created**: 2026-01-26
**Status**: Production (Active since 2025-12-20)
