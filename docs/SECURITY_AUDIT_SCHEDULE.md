# Security Audit Schedule & Procedures

**Document Version**: 1.0
**Created**: January 13, 2026
**Maintained By**: Security Team / DevOps Lead
**Last Review**: January 13, 2026

---

## Executive Summary

This document establishes a comprehensive security audit schedule to maintain the Ollama codebase's security posture. Audits are performed at multiple frequencies to catch different categories of vulnerabilities.

**Security Philosophy**: *Continuous vigilance with layered defense*

- **Daily**: Automated scanning (GitHub Actions)
- **Weekly**: Manual dependency review
- **Monthly**: Comprehensive security audit
- **Quarterly**: Full stack assessment

---

## Daily Security Checks (Automated)

### Schedule
**When**: Every commit + scheduled at midnight UTC
**Duration**: 5-10 minutes
**Runs On**: GitHub Actions

### Tasks

#### 1. Dependency Vulnerability Scanning
```bash
# Command
pip-audit --desc

# Purpose
- Detect known CVEs in installed packages
- Alert on security patches needed
- Track transitive dependencies

# Failure Action
- GitHub Action: WARN (continues but flags)
- Local: FIX before commit or override with approval
```

#### 2. Secrets Detection
```bash
# Commands
detect-secrets scan --all-files
trufflehog scan file:// --json

# Purpose
- Scan for accidentally committed secrets
- Check for hardcoded API keys, credentials
- Monitor for common patterns

# Failure Action
- GitHub Action: FAIL (blocks merge)
- Local: Remove secrets and rotate credentials
```

#### 3. Static Code Analysis
```bash
# Commands
bandit -r ollama/ -f json
mypy ollama/ --strict

# Purpose
- Detect common security anti-patterns
- Type checking catches logic errors
- Find hardcoded values, SQL injection risks

# Failure Action
- GitHub Action: WARN (continues)
- Local: Fix security issues before commit
```

#### 4. CodeQL Analysis
```bash
# Command (runs in GitHub Actions)
# Analyzes: data flow, control flow, information flow

# Purpose
- Advanced static analysis
- Detect SQL injection, XSS, command injection
- Find type confusion and unsafe operations

# Coverage
- Python, JavaScript (if applicable)
- All branches

# Failure Action
- GitHub Action: WARN (flags in PR)
- Review and remediate critical findings
```

---

## Weekly Security Review

### Schedule
**When**: Every Monday 09:00 UTC
**Duration**: 30-60 minutes
**Owner**: Senior Developer / Security Officer

### Tasks

#### 1. Dependency Update Review
```bash
# Command
pip list --outdated

# Checklist
- [ ] Review all outdated packages
- [ ] Check for security updates (check CVE databases)
- [ ] Prioritize by severity:
      - CRITICAL: Patch immediately
      - HIGH: Patch within 1 week
      - MEDIUM: Patch within 1 month
      - LOW: Patch in next release
- [ ] Create issues for updates needed
- [ ] Run full test suite with updates
```

#### 2. GitHub Security Alerts Review
```bash
# Location
GitHub Repo → Security → Vulnerability Alerts

# Checklist
- [ ] Review all new alerts
- [ ] Categorize by severity and exploitability
- [ ] Create issues for each alert
- [ ] Assign to developer
- [ ] Set deadline (based on severity)
```

#### 3. License Compliance Check
```bash
# Command
pip-licenses --format=csv --with-urls

# Checklist
- [ ] Verify no GPL/AGPL licenses (if incompatible)
- [ ] Check for license changes in updates
- [ ] Document any new dependencies
- [ ] Flag suspicious licenses for review
```

#### 4. Recent Commits Review
```bash
# Focus Areas
- [ ] Verify all commits are GPG-signed
- [ ] Check for secrets in commit history
- [ ] Review large file additions (potential data)
- [ ] Verify code review happened
```

---

## Monthly Security Audit

### Schedule
**When**: First Friday of each month at 10:00 UTC
**Duration**: 2-3 hours
**Owner**: Security Officer / Tech Lead
**Participants**: Security team + relevant developers

### Comprehensive Checklist

#### 1. Vulnerability Assessment
```bash
# Commands
pip-audit --desc --by-status
bandit -r ollama/ -f csv -o bandit-report.csv
safety check

# Tasks
- [ ] Generate fresh vulnerability reports
- [ ] Compare with previous month
- [ ] Identify any NEW vulnerabilities
- [ ] Check for "in progress" fixes
- [ ] Prioritize remediation

# Remediation SLA
- CRITICAL: Fix within 24-48 hours
- HIGH: Fix within 1 week
- MEDIUM: Fix within 2 weeks
- LOW: Fix in next release
```

#### 2. Access Control Review
```bash
# Tasks
- [ ] Review GitHub repository access
      - [ ] Verify appropriate permissions
      - [ ] Remove inactive contributors
      - [ ] Check for stale API keys/tokens
      - [ ] Review OAuth scopes
- [ ] Check GCP IAM roles
      - [ ] Least privilege principle
      - [ ] Service account keys rotation
      - [ ] Org policy compliance
- [ ] Database access control
      - [ ] Verify credential rotation (3 months)
      - [ ] Check for default passwords
      - [ ] Review connection pooling limits
```

#### 3. Configuration Security Review
```bash
# Tasks
- [ ] Review all .env.example for new vars
- [ ] Check no secrets in git history
- [ ] Verify TLS/certificate configuration
- [ ] Review rate limiting settings
- [ ] Check CORS allowlist accuracy
- [ ] Review API key policies
- [ ] Verify JWT secret rotation schedule
```

#### 4. Infrastructure Security
```bash
# GCP Load Balancer
- [ ] Verify firewall rules (ports 8000, 5432, 6379, 11434 CLOSED)
- [ ] Check DDoS protection (Cloud Armor)
- [ ] Review SSL/TLS certificate status
- [ ] Verify access logs are being collected

# Docker Security
- [ ] Review Dockerfile base images
- [ ] Check for hardcoded secrets
- [ ] Verify minimal images (no extra tools)
- [ ] Review container network isolation

# Database
- [ ] Verify encryption at rest
- [ ] Check backup encryption
- [ ] Review backup retention policy
- [ ] Verify no public accessibility
```

#### 5. Monitoring & Alerting
```bash
# Tasks
- [ ] Verify all security alerts are configured
- [ ] Check alert destinations (email, Slack)
- [ ] Review recent alert logs
- [ ] Test alert system (send test alert)
- [ ] Verify incident response plan is current
```

#### 6. Documentation Review
```bash
# Tasks
- [ ] Update SECURITY.md with findings
- [ ] Review security policies
- [ ] Check incident response procedures
- [ ] Verify breach notification procedures
- [ ] Update threat model (if applicable)
```

---

## Quarterly Security Assessment (Q1, Q2, Q3, Q4)

### Schedule
**When**: First Monday of each quarter (Jan, Apr, Jul, Oct) at 08:00 UTC
**Duration**: Half-day (4 hours)
**Owner**: Security Officer + External Security Review (optional)
**Participants**: Full team + stakeholders

### Full Stack Assessment

#### 1. Code Review (Security Focus)
```bash
# Tasks
- [ ] Review high-risk code areas
      - [ ] Authentication/Authorization
      - [ ] Cryptographic operations
      - [ ] External API calls
      - [ ] Database queries
      - [ ] File operations
- [ ] Check for security anti-patterns
- [ ] Verify defense-in-depth implementation
```

#### 2. Penetration Testing (Simulated)
```bash
# Scenarios to test
- [ ] SQL injection attempts
- [ ] XSS attack vectors
- [ ] CSRF attacks
- [ ] Rate limiting bypass
- [ ] Authentication bypass
- [ ] Authorization bypass
- [ ] Privilege escalation

# Tools
- Manual testing
- OWASP testing guide
- Security linting tools
```

#### 3. Threat Modeling
```bash
# Review (or create)
- [ ] STRIDE analysis
      - [ ] Spoofing
      - [ ] Tampering
      - [ ] Repudiation
      - [ ] Information Disclosure
      - [ ] Denial of Service
      - [ ] Elevation of Privilege
- [ ] Attack surface analysis
- [ ] Data flow diagram review
```

#### 4. Compliance Check
```bash
# Review against standards
- [ ] OWASP Top 10
- [ ] CWE Top 25
- [ ] Python Security Best Practices
- [ ] Industry standards (HIPAA, PCI if applicable)
```

#### 5. Incident Response Testing
```bash
# Tasks
- [ ] Simulate security incident
- [ ] Execute incident response plan
- [ ] Test communication procedures
- [ ] Verify backup/recovery procedures
- [ ] Document lessons learned
```

---

## Issue Tracking

### Create Security Issues

**For each finding**, create GitHub Issue with:

```markdown
## Title
[SECURITY] {Severity} {Component}: {Brief Description}

## Severity
- CRITICAL: Exploitation possible, high impact
- HIGH: Exploitation possible, medium impact
- MEDIUM: Exploitation unlikely but impact if happens
- LOW: Minor security concern

## Description
[Detailed description of vulnerability]

## Steps to Reproduce
[If applicable]

## Impact
[What an attacker could do]

## Remediation
[How to fix it]

## Evidence
[CVE links, code snippets, test results]

## Labels
- security
- {severity}: critical / high / medium / low
- {category}: dependency / code / config / infra

## Milestone
[Target date for fix]
```

---

## Metrics & Reporting

### Key Security Metrics

1. **Vulnerability Tracking**
   - Total open security issues
   - Average time to remediation
   - Critical vulns fixed within 24h (target: 100%)

2. **Code Coverage**
   - Overall coverage: Target ≥90%
   - Security-critical coverage: Target 100%

3. **Scan Results**
   - SAST findings per scan
   - Dependency vulnerabilities discovered
   - Secrets detected and remediated

4. **Access Control**
   - Days since credential rotation
   - Inactive accounts removed
   - Unauthorized access attempts

### Monthly Report Template

```markdown
# Security Report - {Month} {Year}

## Summary
- Critical Issues: {count}
- High Issues: {count}
- Medium Issues: {count}
- Low Issues: {count}

## Resolved This Month
- {list}

## In Progress
- {list}

## New Findings
- {list}

## Metrics
- Avg remediation time: {days}
- Code coverage: {%}
- Dep vulnerabilities: {count}

## Actions for Next Month
- {list}

## Approval
- Security Officer: {signature}
- Tech Lead: {signature}
```

---

## Incident Response

### Security Incident Escalation

**If CRITICAL vulnerability found:**

1. **Immediate (0-1 hour)**
   - [ ] Create emergency GitHub Issue
   - [ ] Notify security officer
   - [ ] Assess exploitability
   - [ ] Create incident war room (Zoom/Slack)

2. **Urgent (1-4 hours)**
   - [ ] Temporary mitigation (if needed)
   - [ ] Root cause analysis
   - [ ] Coordinate fix with team

3. **Follow-up (1-2 days)**
   - [ ] Deploy fix to production
   - [ ] Verify remediation
   - [ ] Notify stakeholders
   - [ ] Post-incident review

---

## Resources

### Security Tools & Databases

- **Dependency Scanning**: pip-audit, safety, OWASP DependencyCheck
- **Secrets Detection**: detect-secrets, TruffleHog
- **SAST**: Bandit, CodeQL, Semgrep
- **Vulnerability Data**: CVE, NVD, OSV
- **Security Standards**: OWASP, CWE, CAPEC

### Security References

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [Python Security](https://python.org/dev/peps/pep-0620/)
- [NIST Cybersecurity Framework](https://nist.gov/cyberframework)

---

## Sign-Off

**Schedule Established**: January 13, 2026
**Next Review Date**: April 13, 2026 (Q1 Assessment)
**Approved By**: [Security Officer Name]
**Acknowledged By**: [Tech Lead Name]

---

## Appendix: Quick Reference

### Daily
```bash
# Runs automatically on every commit
# GitHub Actions handles: pip-audit, detect-secrets, bandit, CodeQL
```

### Weekly (Monday)
```bash
# Manual review
git log --since="1 week ago" --pretty=format:"%H %s"
pip list --outdated
# Check GitHub Security alerts
```

### Monthly (First Friday)
```bash
# Comprehensive audit
pip-audit --desc
bandit -r ollama/ -f csv
safety check
# Review all findings and create issues
```

### Quarterly (First Monday)
```bash
# Full security assessment
# Full code review
# Threat modeling update
# Incident response drill
# Compliance check
```
