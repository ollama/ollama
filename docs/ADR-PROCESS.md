# Architecture Decision Records (ADR) Process

## Overview

**Purpose**: Document major architectural decisions in a structured, searchable format.

**When to Create**:

- New service/module with significant architectural implications
- Major refactoring affecting multiple services
- Technology choices (database, framework, library)
- API design decisions
- Performance optimizations with tradeoffs
- Security decisions

**Not Needed For**:

- Bug fixes
- Feature additions within existing architecture
- Minor optimizations
- Documentation updates

---

## ADR Format

Create file: `docs/adr/ADR-XXX-short-title.md`

### Template

```markdown
# ADR-001: [Decision Title]

## Status

[Proposed | Accepted | Deprecated | Superseded by ADR-XXX]

## Context

<!-- What problem are we solving? What's the background? -->

## Decision

<!-- What approach did we choose? What are we doing? -->

## Rationale

<!-- Why this decision? What influenced us? -->

## Consequences

<!-- What are the benefits? What are the costs? -->

## Alternatives Considered

<!-- What else did we evaluate? Why not those? -->

## Examples

<!-- Code examples showing the decision in practice -->

## References

<!-- Links to related docs, issues, RFCs -->

## Authors

<!-- Who made this decision -->

## Date

<!-- When was this decided -->
```

### Example: ADR-001

````markdown
# ADR-001: Use PostgreSQL as Primary Database

## Status

Accepted (2026-01-15)

## Context

Our system needs to store structured user data, model information, and inference logs.
We evaluated PostgreSQL, MongoDB, and DynamoDB for this use case.

## Decision

We chose PostgreSQL as our primary database for:

- Structured relational data with ACID guarantees
- Complex queries requiring joins across tables
- Strong consistency requirements for user data
- Ecosystem of tools and libraries
- Cost efficiency at our current scale

## Rationale

**Structured Data**: User profiles, models, and relationships are highly structured.
Relational model with schemas is appropriate.

**Transactions**: We need ACID transactions for operations like user registration
(create user → create API key → send email).

**Query Complexity**: We need complex queries (rankings by usage, analytics aggregations).
PostgreSQL's SQL capabilities exceed document stores.

**Consistency**: User API keys, authentication tokens require strong consistency.

## Consequences

### Benefits

✅ Reliable, battle-tested database system
✅ SQL ecosystem and tooling
✅ Strong consistency guarantees
✅ Mature ORM support (SQLAlchemy)
✅ Easy scaling through replication and read replicas

### Costs

❌ Schema migrations required for changes
❌ Vertical scaling limits (must use read replicas for scale)
❌ NoSQL simplicity for unstructured data

## Alternatives Considered

### MongoDB

**Pros**: Flexible schema, horizontal scaling
**Cons**: Eventual consistency, slower queries with joins, larger data size
**Decision**: Rejected - need ACID transactions and complex queries

### DynamoDB

**Pros**: Serverless, auto-scaling, AWS native
**Cons**: Limited query capabilities, expensive for complex queries, proprietary
**Decision**: Rejected - need SQL and self-hosted control

## Examples

Structured relationships using PostgreSQL:

```python
# User has many API keys
class User(Base):
    id: int = Column(Integer, primary_key=True)
    api_keys: list[APIKey] = relationship("APIKey")

class APIKey(Base):
    id: int = Column(Integer, primary_key=True)
    user_id: int = Column(Integer, ForeignKey("user.id"))

# Query with join:
user_with_keys = session.query(User).filter(
    User.email == "user@example.com"
).options(
    joinedload(User.api_keys)
).first()
```
````

## References

- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [SQLAlchemy ORM](https://docs.sqlalchemy.org/)
- Issue #45: Database Selection
- [Database Comparison Matrix](docs/database-comparison.md)

## Authors

- @kushin77 (Architecture Lead)
- @team-backends (Backend Team)

## Date

2026-01-15

## Related ADRs

- ADR-002: Redis for Caching Layer
- ADR-003: Event-Driven Architecture for Logs

````

---

## ADR Registry

**View all decisions**:
```bash
ls docs/adr/ADR-*.md | sort
````

**ADRs by Status**:

```bash
grep "^## Status" docs/adr/*.md | grep "Accepted"
```

**Search ADRs**:

```bash
grep -r "PostgreSQL" docs/adr/
```

---

## ADR Lifecycle

### 1. Proposal (Proposed)

```markdown
## Status

Proposed
```

- Create ADR with current thinking
- Link in PR/issue for discussion
- Gather feedback from team

### 2. Acceptance (Accepted)

```markdown
## Status

Accepted (2026-01-15)
```

- Decision made after team review
- Implementation proceeds
- Reference in code commits

### 3. Deprecation (Deprecated)

```markdown
## Status

Deprecated (2026-06-01, reason: [explanation])
Replaced by: ADR-XXX
```

- Technology or approach no longer recommended
- Gradual migration path provided
- New ADR explains why

### 4. Supersession (Superseded by ADR-XXX)

```markdown
## Status

Superseded by ADR-XXX (2026-09-01)
```

- Previous decision overridden by new decision
- Link to replacement ADR
- Historical record maintained

---

## Guidelines

### ✅ DO

✅ **Write clearly**: Assume reader unfamiliar with context
✅ **Use rationale**: Explain WHY not just WHAT
✅ **List tradeoffs**: Show you considered alternatives
✅ **Include examples**: Show decision in practice
✅ **Link related**: Connect to issues, RFCs, other ADRs
✅ **Date everything**: Record when decision was made
✅ **Archive old ADRs**: Mark superseded decisions, keep for history

### ❌ DON'T

❌ **Skip rationale**: "We chose X" isn't enough
❌ **Hide costs**: Be honest about consequences
❌ **Ignore alternatives**: Show why other options were worse
❌ **Leave ambiguous**: Readers shouldn't need to ask questions
❌ **Commit implementation code**: ADR is decision, not code
❌ **Make ADRs for everything**: Reserve for architectural decisions

---

## Integration with Development

### In Code Reviews

```
Reviewers should check:
- Is this decision aligned with existing ADRs?
- Should this create a new ADR?
- Are there ADRs that should be updated?
```

### In Architecture Discussions

```
Reference ADRs to avoid re-litigating old decisions:
"See ADR-001 for why we chose PostgreSQL"
```

### In Onboarding

```
New team members read key ADRs to understand:
- Technology choices and rationale
- Architecture constraints and why
- Historical context for current system
```

### In RFC Process

```
For major changes:
1. Create RFC document with ADR template
2. Team discussion and feedback
3. Accept as ADR when decision made
4. Implementation follows accepted ADR
```

---

## Examples by Category

### Data & Storage

- ADR-001: PostgreSQL as Primary Database
- ADR-002: Redis for Caching Layer
- ADR-004: Event Streaming for Audit Logs

### API & Protocols

- ADR-005: REST API Design with Pagination
- ADR-006: Server-Sent Events (SSE) for Streaming
- ADR-007: Rate Limiting Strategy

### Architecture

- ADR-008: Microservices vs Monolith
- ADR-009: Event-Driven Architecture
- ADR-010: API Gateway Pattern

### Security

- ADR-011: API Key Authentication Strategy
- ADR-012: JWT Token Expiration & Refresh
- ADR-013: Secrets Management with Vault

### Operations

- ADR-014: Kubernetes for Orchestration
- ADR-015: Prometheus for Metrics
- ADR-016: Structured Logging Format

---

## Creating Your First ADR

1. **Identify decision**: What architectural choice are you making?
2. **Write template**: Use the template above
3. **Fill sections**: Context, Decision, Rationale, Consequences, Alternatives
4. **Include examples**: Show decision in practice
5. **Request feedback**: Share in PR for team input
6. **Iterate**: Refine based on feedback
7. **Accept**: Mark as "Accepted" when decision finalized
8. **Reference**: Link in code and documentation

---

**Last Updated**: January 14, 2026
**Status**: 🟢 Active
**Maintainer**: Architecture Team
