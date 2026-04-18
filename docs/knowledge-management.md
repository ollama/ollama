# Knowledge Management Infrastructure

**Version**: 1.0
**Last Updated**: 2026-01-26
**Status**: ACTIVE

---

## Overview

Knowledge management system enables our team to:

1. **Prevent incident recurrence** through blameless postmortems
2. **Accelerate onboarding** with runbooks and ADRs
3. **Share learnings** through weekly demos and knowledge logs
4. **Make better decisions** through documented architecture choices

---

## System Components

### 1. Incident Postmortems

**Location**: `/incidents/` directory

Every production incident → postmortem within 48 hours

**Structure**:

- **Template**: `/incidents/POSTMORTEM_TEMPLATE.md`
- **Naming**: `YYYY-MM-DD-[incident-type].md`
- **Example**: `2026-01-15-hallucination-spike.md`

**Process**:

1. Incident occurs
2. War room opened, incident documented live
3. 24 hours later: Team writes postmortem
4. Postmortem reviewed by engineering lead
5. Action items tracked and completed
6. Learnings added to runbooks/ADRs

**Ownership**: On-call engineer (writes postmortem), Engineering lead (reviews)

---

### 2. Runbooks (Standard Operating Procedures)

**Location**: `/docs/runbooks/` directory

Runbooks provide step-by-step procedures for every known incident type.

**Standard Runbooks** (Created):

1. **Agent Hallucination Detected**: `/docs/runbooks/agent-hallucination-detected.md`
   - Detection symptoms, diagnosis steps, remediation options

2. **Database Connection Pool Exhausted**: `/docs/runbooks/database-connection-pool-exhausted.md`
   - Connection limits, how to identify long queries, how to kill them

3. **GCP Quota Exceeded**: `/docs/runbooks/gcp-quota-exceeded.md`
   - Which quotas are critical, how to request increases

4. **Security Vulnerability Found**: `/docs/runbooks/security-vulnerability-found.md`
   - Assessment, patching, escalation procedures

5. **Performance Degradation**: `/docs/runbooks/performance-degradation.md`
   - Diagnosis of slow queries, resource scaling, caching

6. **Data Corruption Detected**: `/docs/runbooks/data-corruption-detected.md`
   - Forensics, backup restoration, escalation

7. **Service Outage**: `/docs/runbooks/service-outage.md`
   - All-hands incident response, communication, rollback

**Runbook Template**: `/docs/runbooks/template.md`

**Key Sections** (All Runbooks Include):

- Detection (how to confirm incident)
- Immediate Actions (0-5 minutes)
- Diagnosis (5-15 minutes)
- Remediation (15+ minutes)
- Escalation (criteria for paging team)
- Post-Incident (create postmortem, action items)

**Usage**:

- On-call engineer executes runbook when incident fires
- Typical execution time: 15-30 minutes from detection to resolution
- Team becomes faster at handling incidents over time

---

### 3. Architecture Decision Records (ADRs)

**Location**: `/docs/adr/` directory

ADRs document major architectural choices and their trade-offs.

**Created ADRs**:

1. **ADR-001**: Cloud Run for Agent Orchestration
   - Why we chose Cloud Run over GKE/Lambda
   - Trade-offs: cost vs. ops burden
   - Consequences: cold starts, memory limits

2. **ADR-002**: BigQuery for Metrics Aggregation
   - Why BigQuery over PostgreSQL/Cloud SQL
   - Cost analysis: scale to 10B+ rows efficiently
   - Performance characteristics

3. **ADR-003**: Pydantic for All Schema Validation
   - Unified validation approach
   - Type safety with mypy
   - Performance considerations

**ADR Template**: `/docs/adr/template.md`

**Key Sections** (All ADRs Include):

- Context (problem statement, constraints)
- Decision (what we chose)
- Consequences (positive, negative, risks)
- Alternatives Considered
- Implementation plan
- References

**Usage**:

- When proposing new technology: Write ADR, discuss with team, accept/reject
- When changing infrastructure: Update related ADR
- When hiring new engineers: Have them read ADRs to understand decisions

**Review Cycle**: Annually or when decision changes

---

### 4. Internal Wiki Structure

**Platform**: Notion/Confluence (to be configured)

**Categories**:

1. **Onboarding** (for new engineers)
   - New engineer checklist (laptop setup, access provisioning, etc.)
   - Environment setup guide (local dev, staging, prod access)
   - Code review process and standards
   - On-call rotation and responsibilities

2. **Agent Development** (for ML engineers)
   - Agent creation template and checklist
   - Prompt engineering guide and best practices
   - Testing requirements and test suite
   - Deployment procedures and approval gates

3. **GCP Integrations** (for platform engineers)
   - Cloud Run setup and deployment
   - Cloud Tasks job queue setup
   - Firestore schema and usage patterns
   - Secret Manager integration

4. **Security** (for all engineers)
   - Threat models by component
   - Vulnerability disclosure process
   - Incident response playbook (links to runbooks)
   - Security training materials and best practices

5. **Infrastructure** (for ops/platform team)
   - Architecture overview with diagrams
   - Disaster recovery procedures
   - Backup/restore procedures
   - Scaling procedures (when to scale, how to scale)

6. **Team Learnings** (for whole team)
   - Weekly discovery log (new insights, bugs found, fixes deployed)
   - Competitive intelligence (what competitors are doing)
   - Customer feedback summary
   - Performance improvements and optimizations

---

### 5. Weekly Demo Process

**Schedule**: Every Friday 3 PM UTC (during standup)

**Process**:

1. Each engineer prepares 3-5 minute demo
2. Demo can be: Live, recorded video, or screenshot walkthrough
3. Topic can be: New feature, bug fix, investigation, performance improvement
4. Demo recorded and linked in wiki for async viewing

**Demo Archive**: Wiki page with searchable index

- Searchable by: Engineer name, date, topic/tag
- Benefit: New team members can watch past demos to learn

**Requirement**: Every engineer ships 1 working demo per week

- Encourages shipping small increments
- Builds visibility across team
- Celebrates progress

---

### 6. Weekly Learning Log

**Location**: Wiki page (to be created)

**Updates**: Every Friday 4 PM UTC (after demos)

**Content**:

- What did we learn this week?
- Security discoveries (vulnerabilities, attack vectors)
- Performance insights (what made things faster/slower)
- Agent improvements (better prompts, new capabilities)
- Competitive intelligence (what competitors shipped)

**Example Entry**:

```
## Week of Jan 26, 2026

### Security Discovery
- Found that agent responses vulnerable to prompt injection
- Mitigation: Added input validation to block obvious injections
- Lesson: Test for adversarial prompts in every feature

### Performance Insight
- Reduced latency by 30% by adding caching layer
- Initial request: 5s, subsequent requests: 500ms
- Lesson: Identify common query patterns and cache aggressively

### Agent Improvement
- Improved hallucination rate from 2% to 0.8%
- Change: Better training data + prompt engineering
- Lesson: Small prompt tweaks have outsized impact
```

---

## Knowledge Management Workflow

### When an Incident Occurs

1. **During Incident** (0-30 min):
   - War room opened on Slack
   - Real-time incident updates
   - Actions documented in thread

2. **After Resolution** (Same day):
   - Post summary to #incident-postmortems
   - Create GitHub issue: `[POSTMORTEM] YYYY-MM-DD - [Type]`

3. **Next Day** (24 hours):
   - Team completes postmortem document
   - Root cause analysis completed
   - Action items assigned

4. **Within Week**:
   - Action items completed
   - Runbooks updated if needed
   - ADRs updated if needed
   - Wiki pages updated

5. **Knowledge Sharing** (Friday):
   - Incident learnings shared in Friday learning log
   - If systemic issue: Present in demo slot

---

### When Making Architecture Decisions

1. **Identify Decision**: Need to choose between 2+ options
2. **Write ADR**: Document problem, decision, consequences
3. **Discuss**: Review with team, iterate
4. **Accept/Reject**: Team consensus on final decision
5. **Implement**: Follow ADR implementation plan
6. **Review**: Annually assess if decision still makes sense

---

### Onboarding New Engineers

**Day 1**:

- [ ] Read Onboarding wiki page
- [ ] Setup laptop, access, dev environment

**Week 1**:

- [ ] Read all 3 ADRs (understand architecture choices)
- [ ] Read all 7 runbooks (know what to do in emergency)
- [ ] Review 3 past postmortems (understand incident patterns)

**Week 2**:

- [ ] Read Security wiki page
- [ ] Review agent development guide
- [ ] Deploy first agent to staging

---

## Success Metrics

| Metric                      | Target                                                 | Current                  | Status |
| --------------------------- | ------------------------------------------------------ | ------------------------ | ------ |
| **Knowledge Base Coverage** | 100% of incident types                                 | 7/7 runbooks created     | ✅     |
| **ADR Coverage**            | All major decisions documented                         | 3/3 ADRs written         | ✅     |
| **Postmortem Rate**         | 100% of SEV1/SEV2 incidents                            | TBD (first incident TBD) | 🔵     |
| **Demo Participation**      | 100% of engineers, weekly                              | TBD                      | 🔵     |
| **Wiki Adoption**           | 100% of new hires complete onboarding                  | TBD                      | 🔵     |
| **Prevention Rate**         | 90% of incident types prevented after first occurrence | TBD                      | 🔵     |

---

## Maintenance & Governance

### Ownership

- **Postmortems**: Engineering lead (reviews/approves)
- **Runbooks**: On-call rotation (updates after incidents)
- **ADRs**: Architecture committee (annual review)
- **Wiki**: Product manager (maintains structure, refreshes quarterly)
- **Demos**: Each engineer (presents own work)
- **Learning Log**: Rotating writer (each engineer owns one week/quarter)

### Review Schedule

- **Runbooks**: After every incident, annual full review
- **ADRs**: Annual review, or when decision changes
- **Wiki**: Quarterly refresh (old content archived)
- **Postmortems**: Action items due within 1 week

### Archival Process

- **Old Postmortems**: Archive after action items completed (keep for reference)
- **Old Runbooks**: Deprecate if incident type never recurs (keep for reference)
- **Deprecated ADRs**: Mark as "Deprecated" with reason and link to replacement

---

## Tools & Platforms

### GitHub (Current Repositories)

- Location: `/incidents/`, `/docs/adr/`, `/docs/runbooks/`
- Benefits: Version control, linked to issues, searchable

### Wiki (Notion/Confluence) (To Be Configured)

- Location: Internal wiki
- Benefits: Rich editing, easy navigation, search, team discussions
- Setup: Create 6 main categories (see above)

### Slack (Communication)

- `#incident-postmortems`: Links to completed postmortems
- `#incident-[type]`: War room channels during incidents
- `#learnings`: Weekly learning log (pinned messages)

### Grafana (Monitoring)

- Dashboards linked to relevant runbooks
- On-call engineers can access runbook from alert

---

## Compliance & Auditing

### Regulatory Requirements

- All postmortems retained for 7 years (for audit)
- All incidents logged with timeline (for compliance)
- All security incidents reported within 24 hours
- All personnel decisions documented (for legal)

### Audit Trail

- GitHub version history: All changes tracked with timestamps
- Slack: Searchable incident channels (1-year retention minimum)
- Wiki: Change history on all pages

---

## Getting Started

### Phase 1 (Complete)

- ✅ Create `/incidents/` directory with postmortem template
- ✅ Create `/docs/runbooks/` with 7 runbooks
- ✅ Create `/docs/adr/` with 3 example ADRs
- ✅ Write this knowledge management guide

### Phase 2 (Next)

- [ ] Setup Notion/Confluence wiki structure (5 hours)
- [ ] Create initial wiki pages in 6 categories (20 hours)
- [ ] Train team on knowledge management process (2 hours)

### Phase 3 (Ongoing)

- [ ] First incident → postmortem created
- [ ] Weekly demos every Friday
- [ ] Monthly wiki refresh
- [ ] Quarterly ADR review

---

## References

- **Postmortem Template**: `/incidents/POSTMORTEM_TEMPLATE.md`
- **Runbook Template**: `/docs/runbooks/template.md`
- **ADR Template**: `/docs/adr/template.md`
- **Related Issue**: #14 (Postmortem & Knowledge Management)
- **Related Issue**: #1 (Elite Execution Protocol - defines knowledge sharing requirements)

---

**Created**: 2026-01-26
**Version**: 1.0
**Last Updated**: 2026-01-26
**Next Review**: 2026-02-26 (one month after implementation)
**Maintained By**: Engineering Team
