# Development Setup Guide

This guide walks you through setting up your development environment for contributing to Ollama.

## Prerequisites

- **Python**: 3.11 or 3.12
- **Git**: With GPG signing capability
- **Docker & Docker Compose**: For local development stack
- **VS Code**: Recommended editor with extensions

## 🏗 Infrastructure Onboarding & Bootstrap

If you need to work on the underlying GCP infrastructure (GCP Landing Zone) or prepare the project for a new environment, please refer to the dedicated guides:

- 📖 [Infrastructure Onboarding Guide](../ONBOARDING_INFRA.md)
- 🚀 [Infrastructure Bootstrap (Pre-flight Validation)](#infrastructure-bootstrap)

## Initial Setup

### 1. Clone Repository

```bash
git clone https://github.com/kushin77/ollama.git
cd ollama
```

### 2. Configure Git

Ensure you have GPG signing configured (required for all commits):

#### 2.1 Generate GPG Key (if needed)

```bash
# Generate GPG key - interactive prompts will guide you
gpg --full-generate-key

# When prompted:
# - Select: RSA and RSA (option 1)
# - Key size: 4096 bits
# - Expiry: 3y (3 years)
# - Real name: Your Name
# - Email: your.email@company.com (must match git config)
# - Passphrase: Strong password (≥20 characters)
#
# Generation takes 1-2 minutes (move mouse, type to generate entropy)
```

#### 2.2 Configure Git to Use GPG

```bash
# List your GPG keys to get the KEY_ID
gpg --list-secret-keys --keyid-format LONG

# Output looks like:
# sec   rsa4096/ABCD1234EFGH5678 2026-01-13 [SC] [expires: 2029-01-13]
#                   ^^^^^^^^^^^^^^
#                   This is your KEY_ID

# Set your KEY_ID globally
git config --global user.signingkey ABCD1234EFGH5678

# Enable GPG signing for all commits
git config --global commit.gpgsign true

# (Optional) Enable GPG signing for tags
git config --global tag.gpgsign true

# Verify configuration
git config --global --list | grep -E "gpg|sign"
# Should show:
# commit.gpgsign=true
# user.signingkey=ABCD1234EFGH5678
```

#### 2.3 Test GPG Signing

```bash
# Create a test commit to verify GPG signing works
git add .gitignore  # or any file
git commit --allow-empty -m "test: verify gpg signing"

# Check if commit is signed (should show "gpg: Good signature")
git log --show-signature -1

# Verify GitHub shows "Verified" badge on commits at:
# https://github.com/kushin77/ollama/commits/main
```

#### 2.4 Troubleshooting GPG Issues

**Issue**: `error: gpg failed to sign the data`

```bash
# Solution: Add GPG_TTY environment variable
export GPG_TTY=$(tty)

# Make it permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export GPG_TTY=$(tty)' >> ~/.bashrc
source ~/.bashrc
```

**Issue**: `fatal: cannot run gpg: No such file or directory`

```bash
# Solution: Install gnupg
# On Ubuntu/Debian:
sudo apt-get install gnupg

# On macOS:
brew install gnupg
```

**Issue**: `Passphrase prompt doesn't appear`

```bash
# Solution: Use gpg-agent to cache passphrases
# Add to ~/.bashrc or ~/.zshrc:
export GPG_TTY=$(tty)
gpg-connect-agent updatestartuptty /bye >/dev/null 2>&1

# Restart terminal for changes to take effect
```

**Issue**: Wrong email in GPG key

```bash
# Generate a new key with correct email
gpg --full-generate-key

# Then update git config to use new key
git config --global user.signingkey YOUR_NEW_KEY_ID
```

#### 2.5 GPG Best Practices

- ✅ **DO**: Use strong passphrase (≥20 characters, mix of upper/lower/numbers/symbols)
- ✅ **DO**: Back up your private key in secure location:
  ```bash
  # Export private key (keep secure!)
  gpg --export-secret-keys YOUR_KEY_ID > private_key_backup.gpg
  ```
- ✅ **DO**: Test GPG signing on each new machine before committing
- ❌ **DON'T**: Share your GPG key or passphrase with anyone
- ❌ **DON'T**: Use an expiring key (set expiry to 3+ years)
- ❌ **DON'T**: Commit without GPG signing (hooks will enforce this)

### 3. Create Python Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -e ".[dev]"  # If using pyproject.toml with [dev] extras
# OR
pip install -r requirements.txt
pip install -r requirements/dev.txt  # If split requirements
```

### 4. Configure Environment

```bash
# Create .env from template
cp .env.example .env

# Fill in actual values
nano .env
# OR
code .env
```

**Key variables to configure**:

```env
DATABASE_URL=postgresql://ollama:password@localhost:5432/ollama
REDIS_URL=redis://localhost:6379/0
JWT_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
```

### 5. Start Development Stack

```bash
# Start Docker services (PostgreSQL, Redis, Qdrant, Ollama)
docker-compose up -d

# Initialize database
alembic upgrade head

# Start development server (in separate terminal)
source venv/bin/activate
uvicorn ollama.main:app --reload --host 0.0.0.0 --port 8000
```

Server will be available at: `http://localhost:8000`

## VS Code Setup

### 1. Install Recommended Extensions

When you open the workspace in VS Code, you'll see a notification to install recommended extensions. Click "Install All" or:

```bash
# Install from command line
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension charliermarsh.ruff
code --install-extension github.copilot
code --install-extension ms-azuretools.vscode-docker
code --install-extension eamodio.gitlens
```

### 2. Verify Python Interpreter

1. Open Command Palette: `Ctrl+Shift+P`
2. Search: "Python: Select Interpreter"
3. Choose: `./venv/bin/python`
4. VS Code will auto-configure settings based on `.vscode/settings.json`

### 3. Configure Git in VS Code

Git is already configured, but verify:

1. Open Settings: `Ctrl+,`
2. Search: `"git.enableCommitSigning"`
3. Should be enabled (✓)

## Running Tests

### Run All Tests

```bash
# From activated venv
pytest tests/ -v --cov=ollama --cov-report=html

# OR use VS Code task
Ctrl+Shift+P → "Tasks: Run Test Task"
```

### Run Specific Tests

```bash
# Run single test file
pytest tests/unit/test_auth.py -v

# Run single test
pytest tests/unit/test_auth.py::test_generate_api_key -v

# Run with coverage
pytest tests/ --cov=ollama --cov-report=term-missing
```

### View Coverage Report

```bash
# Generate and open HTML report
pytest tests/ --cov=ollama --cov-report=html
open htmlcov/index.html  # On macOS
# OR
xdg-open htmlcov/index.html  # On Linux
```

## Code Quality Checks

### Type Checking

```bash
# Run mypy in strict mode
mypy ollama/ --strict

# OR use VS Code task
Ctrl+Shift+P → "Tasks: Run Type Checking"
```

### Linting

```bash
# Run ruff
ruff check ollama/

# With automatic fixes
ruff check ollama/ --fix

# OR use VS Code task
Ctrl+Shift+P → "Tasks: Run Linting"
```

### Code Formatting

```bash
# Format code with black
black ollama/ tests/ --line-length=100

# OR use VS Code task
Ctrl+Shift+P → "Tasks: Format Code"
```

### Security Audit

```bash
# Check for vulnerable dependencies
pip-audit

# OR use VS Code task
Ctrl+Shift+P → "Tasks: Security Audit"
```

### Run All Checks

```bash
# One command to run all checks
Ctrl+Shift+P → "Tasks: Run All Checks"

# OR manually:
mypy ollama/ --strict && \
ruff check ollama/ && \
black ollama/ tests/ --check && \
pip-audit && \
pytest tests/ -v --cov=ollama
```

## Infrastructure Bootstrap

Before deploying to any new GCP environment, you must validate the project's compliance with the **GCP Landing Zone** standards. This includes verifying the [24-label mandate](../../pmo.yaml) and testing the Terraform build.

### 1. Validate pmo.yaml

The project uses a `pmo.yaml` file at the root to store essential business and technical metadata. All 24 labels must be present and correctly formatted.

### 2. Run Pre-flight Validation

Execute the bootstrap script to verify infrastructure readiness:

```bash
# Run the validation script
./scripts/infra-bootstrap.sh
```

This script will:

- Check for all 24 mandatory labels in `pmo.yaml`.
- Generate temporary Terraform variables.
- Initialize and run a `terraform plan` to ensure the configuration is valid.

### 3. Terraform Components

The project's infrastructure is managed under the `terraform/` directory:

- `terraform/00-bootstrap/`: Main project initialization (State, KMS, Service Accounts).

## Development Workflow

### 1. Create Feature Branch

```bash
# Create and switch to branch
git checkout -b feature/your-feature-name

# Branch naming convention:
# - feature/: New features
# - bugfix/: Bug fixes
# - refactor/: Code refactoring
# - perf/: Performance improvements
# - docs/: Documentation updates
# - test/: Test additions/improvements
```

### 2. Make Changes

- Write code with proper type hints
- Include docstrings (Google-style)
- Write tests for new functionality
- Keep functions focused and small

### 3. Run Checks Before Committing

```bash
# Format code
black ollama/ tests/

# Check linting
ruff check ollama/ --fix

# Run tests
pytest tests/ -v --cov=ollama

# Type check
mypy ollama/ --strict

# Security audit
pip-audit
```

### 4. Commit Changes

```bash
# Stage files
git add .

# Commit with conventional format
# (Uses .gitmessage template automatically)
git commit

# VS Code will prompt for message, or use:
# feat(scope): short description
#
# Longer explanation of changes
#
# Fixes #123
```

**Commit Types**:

- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `test`: Test additions/modifications
- `docs`: Documentation changes
- `infra`: Infrastructure/CI-CD changes
- `security`: Security-related changes
- `chore`: Maintenance tasks

### 5. Push and Create Pull Request

```bash
# Push to origin
git push origin feature/your-feature-name

# Create PR on GitHub
# - Use clear title and description
# - Link related issues
# - Ensure all checks pass
```

## Database Migrations

### Create New Migration

```bash
# Create migration with description
alembic revision -m "description of changes"

# Edit autogenerated migration in alembic/versions/
# Write up() and down() functions

# Test migration locally
alembic upgrade head  # Apply
alembic downgrade -1  # Rollback
alembic upgrade head  # Reapply
```

### Apply Migrations

```bash
# Apply all pending migrations
alembic upgrade head

# Apply specific migration
alembic upgrade abc123def456

# Rollback to previous
alembic downgrade -1

# View migration history
alembic history
```

## Common Tasks

### Add New Dependency

```bash
# Add to pyproject.toml or requirements.txt
pip install package_name

# Verify no security issues
pip-audit

# Update lock file (if using poetry)
poetry lock
```

### Run Specific Test Suite

```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# With coverage
pytest tests/ --cov=ollama --cov-report=html
```

### Debug Code

#### Using VS Code Debugger

1. Open file to debug
2. Set breakpoint: Click left of line number
3. Run: `Ctrl+Shift+D` → Select "Python: FastAPI" → Click play
4. Access app at `http://localhost:8000`
5. Step through code, inspect variables

#### Using Python REPL

```bash
# Open Python REPL
python

# Import and test
from ollama.services.cache import CacheManager
cache = CacheManager()
# ...

# Exit
exit()
```

### Check API Documentation

```bash
# With server running (http://localhost:8000)
# Visit these endpoints:

# OpenAPI/Swagger UI
http://localhost:8000/docs

# ReDoc documentation
http://localhost:8000/redoc

# OpenAPI JSON schema
http://localhost:8000/openapi.json
```

### Monitor Services

```bash
# Check running containers
docker ps

# View logs
docker logs container_name
docker logs -f container_name  # Follow logs

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Troubleshooting

### Poetry/Pip Lock Issues

```bash
# Clear pip cache
pip cache purge

# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

### Database Connection Issues

```bash
# Check PostgreSQL is running
docker ps | grep postgres

# Reset database
docker-compose down -v
docker-compose up -d

# Rerun migrations
alembic upgrade head
```

### Type Checking Issues

```bash
# Clear mypy cache
rm -rf .mypy_cache

# Clear ruff cache
rm -rf .ruff_cache

# Rerun checks
mypy ollama/ --strict
ruff check ollama/
```

### Test Failures

```bash
# Run with verbose output
pytest tests/ -vv

# Run with print statements visible
pytest tests/ -s

# Run specific test
pytest tests/unit/test_file.py::test_function -vv
```

## Tools & Extensions Reference

| Tool           | Purpose             | Command                            |
| :------------- | :------------------ | :--------------------------------- |
| pytest         | Testing             | `pytest tests/`                    |
| mypy           | Type checking       | `mypy ollama/ --strict`            |
| ruff           | Linting             | `ruff check ollama/`               |
| black          | Code formatting     | `black ollama/`                    |
| pip-audit      | Security scanning   | `pip-audit`                        |
| alembic        | Database migrations | `alembic upgrade head`             |
| uvicorn        | Dev server          | `uvicorn ollama.main:app --reload` |
| docker-compose | Local stack         | `docker-compose up -d`             |

## Additional Resources

- [Architecture](docs/architecture.md) - System design
- [Contributing](CONTRIBUTING.md) - Contribution guidelines
- [Copilot Instructions](..copilot-instructions) - Development standards
- [Compliance Report](COPILOT_COMPLIANCE_REPORT.md) - Standards compliance

## Getting Help

- **GitHub Issues**: Create issue for bugs or feature requests
- **Discussions**: Start discussion for questions
- **Code Review**: Review PRs and provide feedback
- **Documentation**: Update docs if standards unclear

---

**Last Updated**: January 13, 2026
**Maintained By**: kushin77
**Status**: ✅ Production-Ready
