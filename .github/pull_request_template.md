## Description

<!-- Brief description of the changes in this PR -->

## Type of Change

- [ ] ✨ New feature (non-breaking)
- [ ] 🐛 Bug fix (non-breaking)
- [ ] ♻️ Refactoring (no behavior change)
- [ ] ⚡ Performance improvement
- [ ] 📚 Documentation update
- [ ] 🔐 Security enhancement
- [ ] 🏗️ Infrastructure/DevOps change
- [ ] 🧪 Test addition/modification

## Motivation & Context

<!-- Why is this change needed? What problem does it solve? -->

## Related Issues

<!-- Link to related issues: Fixes #123, Relates to #456 -->
Fixes #

## Testing

### Test Coverage
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests passing: `pytest tests/ -v --cov=ollama`
- [ ] Coverage threshold met: ≥90%

### Manual Testing
<!-- Describe manual testing performed -->

### Performance Impact
<!-- Document any performance implications -->

## Checklist

### Code Quality
- [ ] Code follows project style guide (Black, Ruff, mypy --strict)
- [ ] All type hints in place (100% coverage)
- [ ] No unused imports or variables
- [ ] Error handling is explicit and comprehensive
- [ ] No `# type: ignore` without justification
- [ ] Docstrings added/updated

### Best Practices
- [ ] Single Responsibility Principle (SRP) followed
- [ ] Functions are pure (no unexpected side effects)
- [ ] Maximum function length respected (≤100 lines)
- [ ] Cyclomatic complexity ≤10
- [ ] No magic numbers (constants used instead)
- [ ] Proper logging with structlog
- [ ] No hardcoded credentials/secrets

### Filesystem Standards
- [ ] Directory structure follows Elite Standards
- [ ] Files organized according to naming conventions
- [ ] Test files mirror app structure
- [ ] No premature directory creation

### Git Hygiene
- [ ] Commits are atomic (one logical change per commit)
- [ ] Commit messages follow format: `type(scope): description`
- [ ] Branch name matches pattern: `{type}/{descriptive-name}`
- [ ] All commits are GPG signed
- [ ] No merge conflicts remaining
- [ ] Commit history is clean

### Testing
- [ ] Unit tests for all new functions
- [ ] Integration tests for new features
- [ ] Edge cases covered
- [ ] Error paths tested
- [ ] Coverage ≥90% for modified code

### Security
- [ ] No credentials in code
- [ ] Security audit clean: `pip-audit`
- [ ] OWASP principles followed
- [ ] Input validation in place
- [ ] Rate limiting considered

### Documentation
- [ ] Module docstrings updated
- [ ] Function docstrings with examples
- [ ] README.md updated (if applicable)
- [ ] API documentation updated (if applicable)
- [ ] Inline comments for complex logic

### Pre-Submit
- [ ] `pytest tests/ -v --cov=ollama --cov-report=term-missing` ✅
- [ ] `mypy ollama/ --strict` ✅
- [ ] `ruff check ollama/` ✅
- [ ] `pip-audit` ✅
- [ ] `black --line-length=100 .` ✅

## Deployment Considerations

<!-- Any special deployment steps or considerations? -->

## Breaking Changes

<!-- List any breaking changes and migration path if applicable -->

## Reviewers Notes

<!-- Any specific areas for reviewer attention? -->

---

**Version**: 1.0.0
**Last Updated**: January 13, 2026
**Template Author**: Ollama Elite Standards Team
