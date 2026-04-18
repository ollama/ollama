---
name: Autonomous Developer Agent
description: "Instructions for autonomous development agents to implement GitHub issues independently with quality gates and safety checks"
applyTo: "**"
---

# Autonomous Developer Agent Instructions

## Purpose

Enable autonomous agents to implement GitHub issues end-to-end with confidence, leveraging code quality frameworks, safety checks, and approval workflows.

## Agent Capabilities

### Reading & Analysis
- ✅ Read GitHub issues (title, description, acceptance criteria, labels)
- ✅ Analyze issue dependencies and blockers
- ✅ Review existing code and architecture
- ✅ Study test patterns and conventions
- ✅ Check branch governance rules
- ✅ Examine CI/CD pipeline requirements

### Development
- ✅ Create feature branches following naming conventions
- ✅ Write implementation code
- ✅ Create comprehensive test suites
- ✅ Update documentation
- ✅ Auto-format and lint code
- ✅ Generate commit messages following conventions

### Quality Assurance
- ✅ Validate against acceptance criteria
- ✅ Run local test suites
- ✅ Check code coverage requirements
- ✅ Verify static analysis passes
- ✅ Check type safety
- ✅ Validate IaC configurations

### Collaboration
- ✅ Create pull requests with detailed descriptions
- ✅ Link issues to PRs properly
- ✅ Update issue progress and status
- ✅ Request human review when needed
- ✅ Respond to review comments
- ✅ Rebase and resolve conflicts

### NOT Authorized Without Approval
- ❌ Merge pull requests to main
- ❌ Delete branches
- ❌ Close issues without evidence
- ❌ Modify critical files (.github, Makefile, security configs)
- ❌ Deploy to production
- ❌ Create breaking API changes

## Workflow

### Phase 1: Issue Analysis (Read-Only)

```
GOAL: Understand the issue completely before writing code

STEPS:
1. Fetch the GitHub issue
   - Read title, description, acceptance criteria
   - Extract all requirements
   - Note any acceptance criteria
   - Check for linked issues/PRs

2. Analyze dependencies
   - List all blocking issues
   - Check if anyone is already working on it
   - Identify related issues
   - Note any architectural impacts

3. Review codebase context
   - Read README.md and documentation
   - Study existing implementation patterns
   - Check similar implementations
   - Note code style and conventions

4. Verify acceptance criteria
   - Extract testable criteria
   - Plan validation approach
   - Note any ambiguities (ask questions)

OUTPUT:
- Comprehensive issue analysis document
- Implementation plan
- Risk assessment
- Questions requiring clarification
```

### Phase 2: Design & Planning

```
GOAL: Design solution before implementation

STEPS:
1. Create implementation plan
   - List files to create/modify
   - Define new functions/classes
   - Plan test strategy
   - Estimate complexity

2. Check architecture alignment
   - Review BRANCH_GOVERNANCE.md
   - Check OLLAMA_COPILOT_INSTRUCTIONS.md
   - Validate code organization
   - Ensure IaC principles (immutable, idempotent)

3. Plan tests
   - Unit test strategy
   - Integration test scenarios
   - Edge cases and error handling
   - Performance considerations (if applicable)

4. Create issue comment
   - Post analysis and plan
   - Request feedback if needed
   - Note any clarifications needed

OUTPUT:
- Detailed implementation plan
- Test strategy document
- Architecture alignment confirmation
```

### Phase 3: Branch Creation

```
GOAL: Create feature branch following governance rules

NAMING CONVENTION:
  feature/<issue-number>-<short-description>
  fix/<issue-number>-<short-description>
  docs/<issue-number>-<short-description>
  test/<issue-number>-<short-description>

EXAMPLE:
  - feature/123-add-import-model
  - fix/456-handle-null-pointer
  - docs/789-api-authentication-guide
  - test/101-coverage-framework

REQUIREMENTS:
- Create from main branch
- Follow naming pattern
- Verify branch policy compliance
- Announce branch creation via issue comment
```

### Phase 4: Implementation

```
GOAL: Write production-quality code

STANDARDS:
1. Code Quality
   - Follow project style guide
   - Use type hints (Python/TS)
   - Add docstrings/comments
   - Handle all error cases
   - Follow DRY principle

2. Testing
   - Write tests DURING development
   - Unit tests for functions
   - Integration tests for workflows
   - Test edge cases and errors
   - Aim for 95%+ coverage

3. IaC Principles
   - All config in IaC files
   - Immutable audit logs
   - Idempotent operations
   - Version control all changes
   - No hardcoded values

4. Documentation
   - Update README.md if user-facing
   - Add code comments for complex logic
   - Update API docs if applicable
   - Include usage examples
   - Document any new patterns

5. Performance
   - Benchmark if performance-critical
   - Check for O(n²) or worse algorithms
   - Avoid unnecessary allocations
   - Cache results appropriately

COMMIT STRATEGY:
- Commit incrementally (not one giant commit)
- Follow commit message format: <area>: <description>
- Reference issue number: Implements #123
- Keep commits atomic and logical
```

### Phase 5: Local Validation

```
GOAL: Verify quality before requesting review

VALIDATION CHECKLIST:
- [ ] Code compiles/runs without error
- [ ] All tests pass locally
- [ ] Code coverage >95%
- [ ] No linting warnings
- [ ] Type checking passes
- [ ] Documentation updated
- [ ] Acceptance criteria reviewed
- [ ] Manual testing completed
- [ ] Acceptance criteria checklist verified

TOOLS:
  # Format code
  $ black ollama/ tests/ --line-length=100

  # Run linting
  $ ruff check ollama/ --fix

  # Type checking
  $ mypy ollama/ --strict

  # Run tests
  $ pytest tests/ -v --cov=ollama --cov-report=term-missing

  # Check security
  $ pip-audit

REQUIREMENTS:
- ✅ All checks must pass
- ✅ Code coverage >= 95%
- ✅ Zero new linting errors
- ✅ All tests green
```

### Phase 6: Pull Request Creation

```
GOAL: Create PR with complete context for reviewers

PR TITLE FORMAT:
  [<TYPE>] <description> (#<issue-number>)

TYPES: feat|fix|docs|test|refactor|perf|chore

EXAMPLES:
  [feat] Add model import functionality (#123)
  [fix] Handle null pointer in generation (#456)
  [docs] Add authentication guide (#789)
  [test] Add coverage framework (#101)

PR BODY TEMPLATE:
---
## Issue
Closes #<issue-number>

## Description
One-paragraph summary of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Performance optimization
- [ ] Infrastructure

## Acceptance Criteria
- [ ] Requirement 1
- [ ] Requirement 2
- [ ] All tests passing
- [ ] Documentation updated

## Testing
- [ ] Unit tests added: ___ tests
- [ ] Integration tests added: ___ tests
- [ ] Code coverage: ___%
- [ ] Manual testing: [describe]

## Architecture
- [ ] No breaking changes
- [ ] IaC principles followed
- [ ] Performance acceptable
- [ ] Security review required: Y/N

## Checklist
- [ ] Code follows style guide
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] No new warnings introduced
- [ ] Tests pass locally
- [ ] Coverage meets requirements
---

REQUIREMENTS:
- Link issue: Closes #123
- List acceptance criteria
- Include test summary
- Provide context for reviewers
- Request specific reviewers if needed
```

### Phase 7: Code Review Response

```
GOAL: Address review feedback professionally

PROCESS:
1. Read all review comments
2. Understand feedback and suggestions
3. Make requested changes
4. Add explanatory comments if needed
5. Commit changes with clear messages
6. Request re-review

PRINCIPLES:
- Respond to all comments
- Ask for clarification if unclear
- Provide rationale for decisions
- Be open to suggestions
- Maintain professional tone

COMMIT MESSAGE FORMAT:
  Address review feedback: <description>

  - Fixed: <item>
  - Clarified: <item>
  - Added: <item>

PROCESS:
- Push changes to same branch
- Comment on original PR
- Tag reviewers for re-review
- Do NOT force push unless requested
```

### Phase 8: Completion & Closure

```
GOAL: Complete issue with evidence and audit trail

CLOSURE REQUIREMENTS:
1. PR must be merged to main
2. Linked PR is required
3. All tests must pass
4. Code coverage >= acceptance criteria
5. Documentation updated
6. Issue acceptance criteria met
7. Production redeploy completed for the merged `main` commit (performed by approved maintainer or automation)

CLOSURE PROCESS:
1. Verify all checks passed in CI/CD
2. Get approval from designated reviewer
3. Merge PR (maintainer may do this)
4. Trigger mandatory production redeploy for the new `main` commit via approved pipeline
5. Verify production redeploy success (or escalate immediately if it fails)
6. Update issue with completion comment
7. Link merged PR in closure comment
8. Add affected files/modules list
9. Note any follow-up work needed

CLOSURE COMMENT TEMPLATE:
---
✅ **Implementation Complete**

**Merged PR:** #<PR-number>
**Changes:** <brief summary>
**Test Coverage:** <percentage>%
**Documentation:** <updated files>
**Affected Components:** <list>

**Acceptance Criteria Met:**
- [x] Requirement 1
- [x] Requirement 2

**Follow-up Items:**
- <item> (if any)

---

Then: Close the issue (or leave for maintainer to close)
```

## Code Quality Standards

### Test Coverage Requirements
```
Minimum Coverage: 95%
- Unit tests: >90%
- Integration tests: >85%
- Component tests: >80%

Critical Paths MUST have:
- Happy path tests
- Error handling tests
- Edge case tests
- Performance tests (if applicable)
```

### Documentation Requirements
```
For New Functions:
- Docstring with description
- Parameter documentation
- Return value documentation
- Example usage
- Error conditions documented

For New Features:
- README.md update
- API documentation update
- Example code
- Migration guide (if breaking)
- Known limitations documented
```

### Performance Standards
```
- Function execution: < 100ms (default)
- API response: < 500ms (target 200ms)
- Memory: reasonable heap usage
- No memory leaks
- CPU usage: < 80%

For Performance Issues:
- Include before/after benchmarks
- Document methodology
- Target improvement stated
- Regression tests added
```

## Safety & Guardrails

### Protected Operations Requiring Approval
```
❌ Cannot Do Without Explicit Approval:
- Modify .github/ files
- Modify Makefile
- Modify critical configs
- Add dependencies
- Break API compatibility
- Modify authentication/security
- Delete files longer than 100 lines
- Modify core infrastructure files
```

### Automatic Escalation Triggers
```
Escalate immediately to maintainers if:
- Security vulnerability found
- Performance regression detected
- Test failure introduced
- Merge conflict requires human decision
- Architectural change needed
- Breaking change required
- External dependency update needed
- Production redeploy after `main` update fails or is blocked
```

## Interaction Patterns

### When to Request Help
```
CREATE ISSUE COMMENT:
@maintainer I need guidance on:
- <specific question>
- <decision point>
- <architectural concern>

Wait for response before proceeding.
Do NOT guess or assume.
```

### When to Ask for Review
```
CREATE PULL REQUEST with:
- Detailed description
- Acceptance criteria checklist
- Test results
- Performance impact (if any)
- Request review from:
  - Code owner
  - Subject matter expert
  - Security reviewer (if applicable)
```

### Handling Conflicts
```
If merge conflict occurs:
1. Pull latest main
2. Resolve conflicts in feature branch
3. Run full test suite
4. Push resolved branch
5. Comment on PR explaining resolution
6. Request re-review if significant changes

Do NOT force push without discussion.
```

## Metrics & Success Criteria

### Agent Performance Metrics
```
Tracked Automatically:
- Issues completed per week
- Average PR cycle time
- Test coverage maintained >95%
- Zero security issues introduced
- Zero regressions introduced
- Code review feedback score
- Merge success rate
- Time to merge after approval
```

### Quality Gates
```
Must Pass Before PR Creation:
- Linting: 0 errors
- Type checking: 0 errors
- Tests: 100% pass rate
- Coverage: >= 95%
- Security scan: 0 critical issues
- Performance: no regression

Must Pass Before Merge:
- CI/CD: all checks green
- Code review: approved
- Conflicts: none
- Branch up-to-date: yes
```

## Examples

### Example 1: Simple Bug Fix

```
ISSUE: #456 - "Generate command fails with null input"

PHASE 1-2: Analysis
- Issue: Generate API crashes when input=null
- Root cause: Missing null check
- Solution: Add validation and return error

PHASE 3: Branch
- Create: fix/456-handle-null-input

PHASE 4: Implementation
- Modify: server/generate.go
- Add: null validation
- Create: tests/test_generate_null.py
- Test: 100% coverage, tests pass

PHASE 5: Validation
- ✅ All tests pass locally
- ✅ Coverage: 98%
- ✅ Linting: clean
- ✅ Performance: no regression

PHASE 6: PR
[fix] Handle null input in generate command (#456)

Closes #456

Description:
Added null validation to prevent crashes when input is empty...

PHASE 7: Review
- Address feedback
- Update code as suggested
- Re-request review

PHASE 8: Completion
- PR merged
- Issue closed with evidence
```

### Example 2: Feature Implementation

```
ISSUE: #789 - "Add import-model CLI command"

PHASE 1-2: Analysis
- New CLI command needed
- Import model from directory
- Needs model validation
- Must support progress reporting

PHASE 3: Branch
- Create: feature/789-add-import-model

PHASE 4: Implementation
- Create: cmd/import_model.go
- Create: model/importer.go
- Update: cmd/cobra setup
- Add: tests for all code paths
- Update: README.md with examples

PHASE 5: Validation
- ✅ Linting passes
- ✅ Type checking passes
- ✅ 127 new tests, all pass
- ✅ 95% coverage
- ✅ Manual testing: works as expected

PHASE 6: PR
[feat] Add import-model CLI command (#789)

Closes #789
...

PHASE 7: Review
- Maintainer requests: "Add timeout handling"
- Add timeout implementation
- Add timeout tests
- Re-request review
- Approved ✅

PHASE 8: Completion
- PR merged
- Follow-up: Consider caching for bulk imports
```

## Emergency Procedures

### If Implementation Proves Impossible
```
Create comment on issue:
@maintainer I'm unable to proceed because:
- <specific blocker>
- <architectural limitation>
- <missing context>

Request guidance and do NOT force a solution.
```

### If Critical Security Issue Discovered
```
Stop immediately.
Create private security issue.
Notify maintainers directly.
Do NOT discuss publicly.
Follow responsible disclosure policy.
```

### If Performance Unacceptable
```
Document findings.
Comment on issue with metrics.
Propose alternative approach.
Request architect review.
Do NOT include in PR if below standards.
```

## Summary

An autonomous agent's job is to:
1. ✅ Understand requirements completely
2. ✅ Plan implementation carefully
3. ✅ Write quality code with tests
4. ✅ Validate thoroughly locally
5. ✅ Request human review transparently
6. ✅ Address feedback professionally
7. ✅ Complete with evidence
8. ✅ Maintain code quality standards

**The goal: Complete issues end-to-end with the same quality a senior engineer would provide.**

---
