# Copilot Integration & Elite Standards

This directory contains all documentation and configuration to enforce elite standards through GitHub Copilot integration in VS Code.

## Quick Start

### 1. Set Up Git Hooks
```bash
bash scripts/setup-git-hooks.sh
```

This configures:
- **commit-msg-validate**: Enforces conventional commit format
- **pre-commit-elite**: Runs type checking, linting, formatting
- **pre-push-elite**: Validates branch naming and runs tests

### 2. Configure VS Code

Load enhanced settings:
```bash
cp .vscode/settings-elite.json .vscode/settings-backup.json
# Merge settings from settings-elite.json into settings.json
```

Install recommended extensions:
```bash
code --install-extension GitHub.copilot
code --install-extension GitHub.copilot-chat
code --install-extension ms-python.python
code --install-extension ms-python.black-formatter
code --install-extension charliermarsh.ruff
```

### 3. Configure Git Signing (GPG)

```bash
# List available GPG keys
gpg --list-secret-keys

# Configure git to use a key (replace KEY_ID with actual key)
git config user.signingkey KEY_ID
git config --global commit.gpgsign true

# All commits now require: git commit -S -m "message"
# Or use alias: git config alias.sc 'commit -S'
```

## File Structure

```
.github/
├── copilot-instructions.md    # Main standards document (56KB+)
├── pull_request_template.md   # PR template with standards
└── workflows/                 # CI/CD automation

.githooks/
├── commit-msg-validate        # Validates commit format
├── pre-commit-elite           # Runs code quality checks
└── pre-push-elite             # Validates before push

.vscode/
├── settings-elite.json        # Enhanced elite configuration
├── settings.json              # Current settings
├── extensions.json            # Recommended extensions
├── launch.json                # Debug configuration
└── tasks.json                 # VS Code tasks

docs/
├── ELITE_STANDARDS.md         # Quick reference
├── GIT_WORKFLOW.md            # Git best practices
├── TESTING.md                 # Test guidelines
└── COPILOT_GUIDE.md           # Using Copilot effectively
```

## Copilot Instructions

The main copilot-instructions.md file defines:

### Development Principles
- **Precision & Quality First**: Production-ready code, 100% type hints
- **Local Sovereignty**: All AI runs locally via Docker
- **Security & Privacy**: API keys, TLS 1.3+, no hardcoded credentials
- **Architecture Excellence**: FastAPI, PostgreSQL, Redis, Ollama

### Git Hygiene Mandates
- **Commit Format**: `type(scope): description`
- **Commit Frequency**: Minimum 1 per 30 minutes
- **Atomic Commits**: One logical unit per commit
- **Push Frequency**: At least every 4 hours
- **Branch Naming**: `{type}/{descriptive-name}`
- **Commit Signing**: All commits signed with GPG (`-S` flag)

### Code Quality Standards
- **Type Hints**: 100% coverage, `mypy --strict` passes
- **Test Coverage**: ≥90% overall, 100% for critical paths
- **Single Responsibility**: Functions ≤50 lines, max 4 parameters
- **Error Handling**: Custom exception hierarchy, no bare `except:`
- **Pure Functions**: Favor immutability, document side effects

### Deployment Architecture
- **GCP Load Balancer**: Single external entry point (https://elevatediq.ai/ollama)
- **Internal Docker Network**: All services on docker-compose network
- **Zero Direct Access**: Internal ports never exposed externally
- **Default Endpoints**: Development uses real IP (not localhost)

## Usage with Copilot

### Asking Copilot to Generate Code

Copilot will reference copilot-instructions.md for all code generation:

```
@copilot Create a new API endpoint /api/v1/custom following elite standards

Expected output:
✓ Type hints on all parameters and return types
✓ Docstring with examples
✓ Error handling with custom exceptions
✓ Test file in tests/unit/ mirroring structure
✓ Commit-ready code (passes all checks)
```

### Asking for Analysis

```
@copilot Analyze this function for compliance with elite standards

Checks:
- Type coverage and complexity
- Test coverage
- Docstring quality
- Error handling
- Performance impact
```

### Pre-Commit Validation

Before commit, hooks verify:
1. ✅ Commit message format (conventional)
2. ✅ Type checking passes (`mypy --strict`)
3. ✅ Linting passes (`ruff`)
4. ✅ Code formatted correctly (`black`)
5. ✅ Imports sorted (`isort`)
6. ✅ No debug statements
7. ✅ No TODOs in production code

### Pre-Push Validation

Before push, hooks verify:
1. ✅ Branch name matches pattern
2. ✅ All tests pass
3. ✅ Type checking passes
4. ✅ Linting passes

## Environment Setup

### Python Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements/dev.txt
```

### Development Endpoint (Real IP, Not Localhost)

```bash
# Get your local IP
export REAL_IP=$(hostname -I | awk '{print $1}')

# Or use DNS
export REAL_IP="dev-ollama.internal"

# Configure environment
export FASTAPI_HOST=0.0.0.0
export FASTAPI_PORT=8000
export PUBLIC_API_URL=http://$REAL_IP:8000

# Start server - accessible via real IP
uvicorn ollama.main:app --reload --host 0.0.0.0 --port 8000

# Access at: http://$REAL_IP:8000  (NOT localhost)
```

## CI/CD Integration

GitHub Actions runs all checks on push:
```yaml
- Type checking (mypy)
- Linting (ruff)
- Tests (pytest)
- Security audit (pip-audit)
- Build Docker image
```

## Troubleshooting

### Commit Hook Failing
```bash
# View what's failing
git commit --no-verify  # Bypass (not recommended)

# Fix and re-try
bash scripts/setup-git-hooks.sh
git commit -S -m "type(scope): description"
```

### Type Checking Issues
```bash
python3 -m mypy ollama/ --strict --show-error-codes
```

### Linting Issues
```bash
python3 -m ruff check ollama/ --fix
```

### Branch Name Wrong
```bash
git branch -m old-name feature/correct-name
```

## Resources

- **Main Standards**: [.github/copilot-instructions.md](.github/copilot-instructions.md)
- **VS Code Setup**: [.vscode/settings-elite.json](.vscode/settings-elite.json)
- **Git Hooks**: [.githooks/](../.githooks/)
- **Type Checking**: [mypy documentation](https://mypy.readthedocs.io/)
- **Code Style**: [Black formatter](https://black.readthedocs.io/)
- **Linting**: [Ruff](https://github.com/astral-sh/ruff)

## Version

- **Copilot Instructions**: v2.0.0
- **Elite Standards**: v2.0.0
- **Last Updated**: January 13, 2026

---

**For questions or updates, refer to copilot-instructions.md or contact the team.**
