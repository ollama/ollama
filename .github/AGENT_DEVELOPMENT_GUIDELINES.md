# GitHub Copilot Agent Development Guidelines

**Document Version:** 1.0  
**Last Updated:** April 18, 2026  
**Audience:** Autonomous Development Agents

---

## Overview

This document defines the guidelines for autonomous development agents (like GitHub Copilot) to implement features, create PRs, and manage issues independently in the Ollama repository.

**Key Principle:** All work must follow Infrastructure as Code (IaC) principles - everything is versioned, auditable, and recoverable.

---

## What Agents CAN Do Autonomously ✅

Agents have full authority to do the following without human approval:

### Code Implementation
- ✅ Create feature branches
- ✅ Implement code against acceptance criteria
- ✅ Write unit tests and integration tests
- ✅ Update documentation
- ✅ Commit changes with clear messages
- ✅ Push branches to origin

### Quality Assurance
- ✅ Run tests locally
- ✅ Check code coverage
- ✅ Run linting and type checking
- ✅ Format code to project standards
- ✅ Verify against acceptance criteria

### Issue Management
- ✅ Create pull requests with detailed descriptions
- ✅ Link PRs to issues (Closes #123)
- ✅ Add comments to issues with updates
- ✅ Apply labels to issues
- ✅ Update issue descriptions with progress

### Workflow Automation
- ✅ Close issues with evidence comments
- ✅ Reopen issues if work incomplete
- ✅ Request reviews from code owners
- ✅ Address review feedback and force-push
- ✅ Manage branch protection exceptions (with reason)

---

## What Requires Human Approval ❌

Agents MUST escalate and NOT proceed without explicit human approval:

### Merge Authority
- ❌ Cannot merge PRs to main (humans only)
- ❌ Cannot delete branches
- ❌ Cannot merge without all checks passing

### Breaking Changes
- ❌ Cannot make breaking API changes
- ❌ Cannot remove public methods/functions
- ❌ Cannot change CLI interface without deprecation
- ❌ Cannot modify core infrastructure files

### Sensitive Operations
- ❌ Cannot modify authentication/security modules
- ❌ Cannot add/modify dependencies without justification
- ❌ Cannot commit secrets or credentials
- ❌ Cannot modify CI/CD pipelines

### Deployments
- ❌ Cannot deploy to production
- ❌ Cannot deploy to staging without approval
- ❌ Cannot execute production data migrations

### Critical Infrastructure
- ❌ Cannot modify .github/ files (workflows, protection rules) without approval
- ❌ Cannot modify Makefile without justification
- ❌ Cannot modify security configurations

---

## Quality Gates (MUST PASS)

All code must pass these gates before creating a PR:

### Testing
- ✅ All tests pass locally: `go test ./...`
- ✅ Code coverage ≥ 95%: `go tool cover -html=coverage.out`
- ✅ Zero test failures

### Code Quality
- ✅ No linting warnings: `golangci-lint run`
- ✅ Proper formatting: `go fmt ./...`
- ✅ Type safety checks pass

### Security
- ✅ No hardcoded secrets
- ✅ No SQL injection vulnerabilities
- ✅ No SSRF issues
- ✅ Proper input validation
- ✅ Error messages don't leak info

### Documentation
- ✅ Code comments for complex logic
- ✅ Function/method documentation
- ✅ README updated if user-facing
- ✅ Acceptance criteria documented in PR

---

## Workflow: Implementation to PR

### Phase 1: Analysis (Read-Only)

1. Read the GitHub issue completely
2. Extract acceptance criteria
3. Identify any blockers
4. Check for related issues
5. Review codebase context
6. Create implementation plan

**Deliverable:** Implementation plan comment on issue

### Phase 2: Branch & Setup

1. Create feature branch: `feature/<issue>-<description>`
   - Example: `feature/42-kubernetes-hub`
2. Pull latest main: `git pull origin main`
3. Create branch locally: `git checkout -b feature/42-kubernetes-hub`
4. Push to origin: `git push -u origin feature/42-kubernetes-hub`

**Documentation:**
```bash
# Branch naming rules
feature/<issue>-<kebab-case-description>  # New feature
fix/<issue>-<kebab-case-description>      # Bug fix
docs/<issue>-<kebab-case-description>     # Documentation
test/<issue>-<kebab-case-description>     # Tests
refactor/<issue>-<kebab-case-description> # Refactoring
```

### Phase 3: Implementation

1. Write implementation code
2. Write tests alongside code
3. Commit frequently with clear messages
4. Keep commits atomic and logical

**Commit message format:**
```
<area>: <concise description>

- <detail 1>
- <detail 2>

Implements #<issue>
```

**Example commits:**
```
kubernetes: implement provider connection logic

- Add Provider.Connect() method
- Add Provider.IsAvailable() check
- Add proper error handling
- Add unit tests

Implements #42
```

### Phase 4: Quality Checks

Before creating PR, verify:
- ✅ All tests pass
- ✅ Coverage ≥ 95%
- ✅ No linting warnings
- ✅ Code properly formatted
- ✅ Documentation complete
- ✅ All acceptance criteria met
- ✅ No breaking changes

### Phase 5: Pull Request

Create PR with template:

```markdown
## Issue
Closes #<issue-number>

## Summary
One paragraph describing changes.

## Type of Change
- [ ] Feature
- [ ] Bug fix
- [ ] Documentation
- [ ] Performance improvement

## Acceptance Criteria
- [ ] Requirement 1
- [ ] Requirement 2
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Code reviewed

## Testing
- Unit tests: <count>
- Integration tests: <count>
- Coverage: <percentage>%
- Test results: All passing

## Implementation Notes
- Key changes made
- Files modified
- New dependencies (if any)

## Related Issues
- Closes #<issue>
- Relates to #<other>
```

### Phase 6: Address Review Feedback

1. Read all review comments
2. Make requested changes
3. Commit with clear message
4. Do NOT force push (rebase only if necessary)
5. Request re-review

**Feedback commit message:**
```
Address review feedback for #<issue>

- Fixed: <item>
- Improved: <item>
- Clarified: <item>
```

### Phase 7: Post-Merge

1. Verify all checks passed in CI/CD
2. Wait for human approval to merge
3. Once merged, close related issues
4. Document any follow-up work needed

---

## Issue Closure Protocol

Issues can be closed by agents ONLY when:

1. ✅ All acceptance criteria met
2. ✅ All tests passing
3. ✅ Full documentation complete
4. ✅ PR merged to main (or linked commits)
5. ✅ Evidence documented in issue comment

### Closure Comment Template

```markdown
✅ **Implementation Complete**

## Summary
Brief description of implementation.

## Acceptance Criteria
- [x] Requirement 1
- [x] Requirement 2
- [x] Requirement 3
- [x] All tests passing

## Implementation
- Code: <files changed>
- Tests: <test count and coverage>
- Documentation: <docs updated>
- PR/Commits: <links to evidence>

## Ready For
- Code review
- Merge
- Usage in production

**Status:** Ready for next phase
```

---

## Labeling Strategy

All issues MUST have these labels for proper tracking:

### Status Labels
- `status/completed` - Fully implemented and verified
- `status/in-progress` - Active development
- `status/blocked` - Waiting on dependency
- `status/needs-triage` - Requires clarification
- `status/backlog` - Not yet started

### Priority Labels
- `priority/critical` - Urgent, blocks other work
- `priority/high` - Important, schedule soon
- `priority/medium` - Normal priority
- `priority/low` - Nice to have

### Type Labels
- `type/feature` - New functionality
- `type/bug` - Defect or regression
- `type/docs` - Documentation only
- `type/test` - Test infrastructure
- `type/refactor` - Code quality

### Area Labels
- `area/kubernetes` - Kubernetes integration
- `area/api` - REST API
- `area/cli` - Command-line interface
- `area/test` - Testing framework
- `area/docs` - Documentation
- `area/infra` - Infrastructure

### Capability Labels
- `agent-ready` - Ready for autonomous agent
- `good-first-issue` - Suitable for new contributors
- `help-wanted` - Looking for contributor
- `needs-review` - Awaiting human review

---

## Issue Creation Guidelines for Agents

When creating issues, include:

1. **Title:** Clear, concise, actionable
   - ❌ "Fix stuff"
   - ✅ "Add Kubernetes cluster health check endpoint"

2. **Description:** Background context
   - Why this work is needed
   - Current behavior vs desired behavior
   - Impact or benefit

3. **Acceptance Criteria:** Testable, measurable
   - Must be written as verifiable statements
   - Can be automated in tests
   - Each criterion should be independent

4. **Implementation Approach:** Optional guidance
   - Suggested implementation (not required)
   - Files likely to be modified
   - Key decisions to make

5. **Resources:** Links to relevant docs
   - Design documents
   - RFC links
   - Related issues
   - External documentation

6. **Labels:** Applied upon creation
   - Type: feature/bug/docs/test
   - Priority: critical/high/medium/low
   - Area: kubernetes/api/cli/etc

---

## Error Handling & Escalation

### When to Escalate

Stop and request human review if:

1. **Blocker Encountered**
   - Cannot proceed due to missing information
   - Architectural decision needed
   - External dependency issue

2. **Ambiguity Found**
   - Requirements unclear
   - Multiple valid approaches
   - Unclear acceptance criteria

3. **Quality Issue**
   - Cannot achieve required test coverage
   - Performance unacceptable
   - Security concern

4. **Scope Change**
   - Work growing beyond original issue
   - New requirements emerge
   - Dependencies discovered

### Escalation Format

```markdown
@human-reviewer

I'm unable to proceed on #issue because:

**Blocker:**
- <specific issue>

**Questions:**
- <what clarification needed>

**Options Considered:**
- <option 1> - pros: ..., cons: ...
- <option 2> - pros: ..., cons: ...

**Recommendation:**
<suggested path forward>

Please advise or provide clarification before I continue.
```

---

## Infrastructure as Code (IaC) Requirements

All changes must follow IaC principles:

### Immutability
- ✅ All changes committed to git
- ✅ No local-only changes
- ✅ All decisions documented in commits
- ✅ Audit trail preserved

### Idempotence
- ✅ Operations can run multiple times safely
- ✅ No side effects from repeated execution
- ✅ Configuration versioned, not state
- ✅ Deployments repeatable

### Global Consistency
- ✅ All issues tracked in GitHub
- ✅ All code in git repositories
- ✅ All documentation in version control
- ✅ All secrets in secure storage (not in git)
- ✅ All approvals logged in PR review history

---

## Security Best Practices

Agents MUST follow security guidelines:

### Code Security
- ✅ Input validation on all endpoints
- ✅ SQL injection prevention (use parameterized queries)
- ✅ CSRF token protection
- ✅ Authentication checks on sensitive endpoints
- ✅ Proper error messages (no info leakage)

### Secrets Management
- ✅ No hardcoded secrets in code
- ✅ Use environment variables for sensitive data
- ✅ Never commit .env files
- ✅ Document secret requirements
- ✅ Use GitHub Secrets in CI/CD

### Data Protection
- ✅ Encrypt sensitive data in transit (HTTPS)
- ✅ Hash passwords (never store plain text)
- ✅ Validate all external input
- ✅ Sanitize log output (no secrets)
- ✅ Clean up temporary files

### Access Control
- ✅ Verify authorization before operations
- ✅ Use least privilege principle
- ✅ Implement proper RBAC
- ✅ Audit sensitive operations
- ✅ Rate limit API endpoints

---

## Testing Requirements

All code must have comprehensive tests:

### Unit Tests
- Test all public functions
- Test error conditions
- Test edge cases
- Aim for ≥ 95% coverage
- Use table-driven tests for multiple scenarios

### Integration Tests
- Test component interactions
- Test with real dependencies (where practical)
- Test error propagation
- Test timeout handling
- Use test fixtures for setup

### Performance Tests
- Benchmark critical paths
- Identify performance regressions
- Document acceptable performance
- Profile memory usage
- Test under load

---

## Documentation Requirements

All code must be well-documented:

### Code-Level Documentation
```go
// Package kubernetes provides Kubernetes cluster integration
// for model deployment and management.
package kubernetes

// Provider manages connections to Kubernetes clusters and
// handles operations like deployment, scaling, and status.
type Provider struct {
    client kubernetes.Interface
    config *rest.Config
}

// Connect establishes a connection to a Kubernetes cluster
// using the provided kubeconfig path. If path is empty, it
// attempts in-cluster authentication.
func (p *Provider) Connect(ctx context.Context, kubeconfigPath string) error {
    // ...
}
```

### User-Facing Documentation
- README.md sections for new features
- API endpoint documentation
- CLI usage examples
- Configuration guide
- Troubleshooting section

### Operational Documentation
- Deployment instructions
- Monitoring and alerting setup
- Upgrade procedure
- Rollback procedure
- Known limitations and workarounds

---

## Approval & Merge Process

### Before Merge
1. ✅ All CI/CD checks passing
2. ✅ Code review approved
3. ✅ Test coverage verified
4. ✅ Documentation complete
5. ✅ Breaking changes approved

### Merge Execution
- Human merges PR to main (agent cannot merge)
- All checks must be green
- No conflicts (rebase if needed)
- Branch deleted after merge

### Post-Merge
- Close related issues with evidence
- Deploy to staging (if applicable)
- Prepare deployment documentation
- Mark as complete

---

## Continuous Improvement

Agents should monitor and report:

- 📊 Code coverage trends
- 📊 Test execution time
- 📊 Build failure patterns
- 📊 Review cycle time
- 📊 Issue resolution time

Update this document if you find:
- Unclear guidelines
- Impossible requirements
- Better practices
- Process improvements

---

## Summary

**Autonomous agents can:**
- ✅ Implement features
- ✅ Write tests
- ✅ Create PRs
- ✅ Manage issues
- ✅ Commit code
- ✅ Push branches

**Humans must:**
- ❌ Approve and merge PRs
- ❌ Approve breaking changes
- ❌ Handle security decisions
- ❌ Deploy to production
- ❌ Modify critical infrastructure

**All work must:**
- ✅ Follow IaC principles
- ✅ Pass quality gates
- ✅ Be committed to git
- ✅ Have audit trail
- ✅ Be fully documented

---

*Guidelines Version: 1.0*  
*Last Updated: 2026-04-18*  
*Status: ACTIVE*
