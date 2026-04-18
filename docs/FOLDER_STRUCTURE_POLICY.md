# Elite Filesystem Standards - Enforcement Policy

**Last Updated**: January 13, 2026
**Status**: ENFORCED via pre-commit hooks

---

## Overview

This document defines the mandatory folder structure for the Ollama repository. The structure is **automatically enforced** via git hooks and validation scripts.

---

## Root Directory - Strict Whitelist

**ONLY the following files are allowed at root:**

### Essential Project Files
- `README.md` - Project overview and quick start
- `CONTRIBUTING.md` - Contribution guidelines
- `LICENSE` - Project license
- `CHANGELOG.md` - Version history

### Configuration Files
- `pyproject.toml` - Python project configuration
- `setup.py` - Python package setup
- `.gitignore` - Git ignore rules
- `.pre-commit-config.yaml` - Pre-commit hooks
- `alembic.ini` - Database migration config

### Deployment Files
- `Dockerfile` - Docker image definition
- `docker-compose.yml` - Main deployment
- `docker-compose.prod.yml` - Production deployment
- `docker-compose.minimal.yml` - Minimal deployment
- `docker-compose.elite.yml` - Elite deployment

### Utility Scripts
- `test_server.py` - Server testing utility
- `verify-completion.sh` - Completion verification

### Summary Documents (Elite Standards)
- `IMPLEMENTATION_COMPLETE.md` - Implementation status
- `ELITE_STANDARDS_EXECUTIVE_SUMMARY.md` - Executive overview
- `ELITE_STANDARDS_IMPLEMENTATION_COMPLETE.md` - Technical report
- `DEEP_SCAN_ELITE_STANDARDS_REPORT.md` - Deep scan results

### Hidden Files (Allowed)
- `.copilot-instructions` - Symlink to instructions
- `.coverage` - Test coverage data

---

## Required Directory Structure

```
ollama/
├── README.md                           # ✅ Root essential
├── CONTRIBUTING.md                     # ✅ Root essential
├── LICENSE                            # ✅ Root essential
├── pyproject.toml                     # ✅ Root config
├── docker-compose.yml                 # ✅ Root deployment
├──
├── ollama/                            # ✅ Application code
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── api/                           # API routes
│   │   ├── routes/
│   │   ├── schemas/
│   │   └── ...
│   ├── services/                      # Business logic
│   ├── repositories/                  # Data access
│   ├── models/                        # ORM models
│   └── monitoring/                    # Observability
│
├── tests/                             # ✅ Test suite
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── docs/                              # ✅ Documentation
│   ├── ELITE_STANDARDS_REFERENCE.md  # Quick reference
│   ├── DEPLOYMENT.md                  # Deployment guide
│   ├── reports/                       # ✅ Archived reports
│   │   ├── INDEX.md
│   │   ├── DEPLOYMENT_*.md
│   │   ├── PHASE_4_*.md
│   │   └── ...
│   └── archive/                       # Historical docs
│
├── scripts/                           # ✅ Automation scripts
│   ├── setup-git-hooks.sh
│   ├── verify-elite-setup.sh
│   ├── cleanup-root-directory.sh
│   └── validate-folder-structure.sh
│
├── .github/                           # ✅ GitHub config
│   ├── copilot-instructions.md        # Elite standards
│   ├── COPILOT_INTEGRATION.md
│   ├── workflows/                     # CI/CD
│   └── ISSUE_TEMPLATE/
│
├── .githooks/                         # ✅ Git hooks
│   ├── commit-msg-validate
│   ├── pre-commit-elite
│   └── pre-push-elite
│
├── .vscode/                           # ✅ VS Code config
│   ├── settings.json
│   ├── settings-elite.json
│   ├── extensions.json
│   └── tasks.json
│
├── config/                            # ✅ Configuration
│   ├── development.yaml
│   ├── production.yaml
│   └── ...
│
├── docker/                            # ✅ Docker configs
│   ├── nginx/
│   ├── postgres/
│   └── ...
│
├── k8s/                              # ✅ Kubernetes
│   ├── base/
│   └── overlays/
│
├── monitoring/                        # ✅ Observability
│   ├── prometheus.yml
│   ├── grafana/
│   └── ...
│
├── alembic/                          # ✅ DB migrations
│   └── versions/
│
└── requirements/                      # ✅ Dependencies
    ├── base.txt
    ├── dev.txt
    └── prod.txt
```

---

## File Placement Rules

### ❌ NEVER at Root
- Status reports (→ `docs/reports/`)
- Temporary documentation (→ `docs/` or `docs/archive/`)
- Shell scripts (→ `scripts/`)
- Python scripts (→ `scripts/` or `ollama/`)
- Configuration files (→ `config/`)
- Test files (→ `tests/`)

### ✅ Where Files Belong

| File Type | Destination | Example |
|-----------|-------------|---------|
| Status reports | `docs/reports/` | `DEPLOYMENT_STATUS.md` |
| Documentation | `docs/` | `API_DESIGN.md` |
| Shell scripts | `scripts/` | `backup.sh` |
| Python code | `ollama/` | `auth.py` |
| Tests | `tests/unit/` | `test_auth.py` |
| Configuration | `config/` | `logging.yaml` |
| Deployment | `docker/`, `k8s/` | `Dockerfile.api` |

---

## Enforcement Mechanisms

### 1. Pre-Commit Hook ✅

**File**: `.githooks/pre-commit-elite`

Automatically checks folder structure before every commit:
```bash
# Validates:
- No loose .md or .txt files at root (except whitelisted)
- No loose scripts at root
- All files in proper directories
```

**Result**: Commit is **blocked** if violations found.

### 2. Validation Script ✅

**File**: `scripts/validate-folder-structure.sh`

Manual validation:
```bash
bash scripts/validate-folder-structure.sh
```

Checks:
- Required directories present
- No loose files at root
- Proper file organization

### 3. Cleanup Script ✅

**File**: `scripts/cleanup-root-directory.sh`

Auto-organize loose files:
```bash
bash scripts/cleanup-root-directory.sh
```

Moves:
- Status reports → `docs/reports/`
- Creates index in `docs/reports/INDEX.md`

---

## Handling Violations

### If Pre-Commit Hook Blocks Your Commit

```bash
# 1. See what's wrong
git status

# 2. Auto-cleanup (if status reports)
bash scripts/cleanup-root-directory.sh

# 3. Or manually move files
mv SOME_REPORT.md docs/reports/

# 4. Stage changes
git add docs/reports/

# 5. Retry commit
git commit -m "docs: reorganize status reports"
```

### Creating New Files

**ALWAYS create files in proper directories:**

```bash
# ❌ WRONG - Creates loose file at root
echo "# New Doc" > NEW_FEATURE.md

# ✅ CORRECT - Creates in docs/
echo "# New Doc" > docs/NEW_FEATURE.md
```

---

## Exceptions & Special Cases

### Allowed Root Documents

**Elite Standards Summaries** (from deep scan):
- `IMPLEMENTATION_COMPLETE.md`
- `ELITE_STANDARDS_EXECUTIVE_SUMMARY.md`
- `ELITE_STANDARDS_IMPLEMENTATION_COMPLETE.md`
- `DEEP_SCAN_ELITE_STANDARDS_REPORT.md`

**Rationale**: High-visibility executive summaries that need to be immediately visible.

### Temporary Files

**Build artifacts, logs, coverage** → Already in `.gitignore`
- `.coverage`, `htmlcov/`, `*.log`
- Never committed, so not enforced

---

## Verification Checklist

Before committing:
- [ ] No loose `.md` files at root (except whitelisted)
- [ ] No loose `.txt` files at root
- [ ] No loose scripts at root
- [ ] Status reports in `docs/reports/`
- [ ] Documentation in `docs/`
- [ ] Scripts in `scripts/`
- [ ] Python code in `ollama/`
- [ ] Tests in `tests/`

---

## Team Guidelines

### For All Developers

1. **Never create loose files at root**
   - Use proper subdirectories from start

2. **Run validation before push**
   ```bash
   bash scripts/validate-folder-structure.sh
   ```

3. **Archive old reports immediately**
   ```bash
   mv OLD_REPORT.md docs/reports/
   ```

### For Code Reviewers

Check PRs for:
- No loose files added to root
- Files in correct directories
- Proper organization maintained

### For CI/CD

Add validation to pipeline:
```yaml
- name: Validate Folder Structure
  run: bash scripts/validate-folder-structure.sh
```

---

## Rationale

**Why enforce strict folder structure?**

✅ **Discoverability**: Files are where you expect them
✅ **Maintainability**: Clear organization over time
✅ **Scalability**: Structure supports growth
✅ **Professionalism**: Elite-level project organization
✅ **Automation**: Tools know where files are
✅ **Onboarding**: New developers find things easily

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0 | 2026-01-13 | Pre-commit enforcement added |
| 1.0.0 | 2026-01-13 | Initial enforcement policy |

---

## References

- **Main Standards**: `.github/copilot-instructions.md`
- **Elite Reference**: `docs/ELITE_STANDARDS_REFERENCE.md`
- **Validation Script**: `scripts/validate-folder-structure.sh`
- **Cleanup Script**: `scripts/cleanup-root-directory.sh`

---

**Status**: ✅ ENFORCED
**Automated**: Pre-commit hooks + validation scripts
**Compliance**: MANDATORY for all commits
