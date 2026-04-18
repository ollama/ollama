#!/bin/bash
# Setup verification script
# Verifies that all elite standards configurations are in place
# Usage: bash scripts/verify-elite-setup.sh

set -e

echo "🔍 Verifying Elite Standards Setup..."
echo ""

checks_passed=0
checks_failed=0

# Helper functions
check_pass() {
    echo "  ✅ $1"
    ((checks_passed++))
}

check_fail() {
    echo "  ❌ $1"
    ((checks_failed++))
}

check_warn() {
    echo "  ⚠️  $1"
}

# ============================================================
# Git Configuration Checks
# ============================================================
echo "📝 Git Configuration Checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check git hooks path
if git config core.hooksPath | grep -q "\.githooks"; then
    check_pass "Git hooks path configured (.githooks)"
else
    check_warn "Git hooks path not configured (run: bash scripts/setup-git-hooks.sh)"
fi

# Check commit signing
if git config commit.gpgsign | grep -q "true"; then
    check_pass "Commit signing enabled (commit.gpgsign)"
else
    check_warn "Commit signing not enabled (run: git config user.signingkey <KEY>)"
fi

# Check for GPG key
if git config user.signingkey > /dev/null 2>&1; then
    check_pass "GPG key configured"
else
    check_warn "No GPG key configured (optional but recommended)"
fi

echo ""

# ============================================================
# Hook Files Checks
# ============================================================
echo "🪝 Git Hooks Checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check hook files exist
if [ -x .githooks/commit-msg-validate ]; then
    check_pass "commit-msg-validate hook exists and executable"
else
    check_fail "commit-msg-validate hook missing or not executable"
fi

if [ -x .githooks/pre-commit-elite ]; then
    check_pass "pre-commit-elite hook exists and executable"
else
    check_fail "pre-commit-elite hook missing or not executable"
fi

if [ -x .githooks/pre-push-elite ]; then
    check_pass "pre-push-elite hook exists and executable"
else
    check_fail "pre-push-elite hook missing or not executable"
fi

echo ""

# ============================================================
# Configuration Files Checks
# ============================================================
echo "⚙️  Configuration Files Checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check pre-commit config
if [ -f .pre-commit-config.yaml ]; then
    if grep -q "black\|ruff\|mypy" .pre-commit-config.yaml; then
        check_pass ".pre-commit-config.yaml exists with quality tools"
    else
        check_fail ".pre-commit-config.yaml missing quality tools"
    fi
else
    check_fail ".pre-commit-config.yaml not found"
fi

# Check .gitignore
if [ -f .gitignore ]; then
    check_pass ".gitignore exists"
else
    check_fail ".gitignore not found"
fi

# Check VS Code settings
if [ -f .vscode/settings.json ]; then
    if grep -q "python.analysis.typeCheckingMode" .vscode/settings.json; then
        check_pass ".vscode/settings.json configured for strict type checking"
    else
        check_warn ".vscode/settings.json may not have strict type checking"
    fi
else
    check_fail ".vscode/settings.json not found"
fi

echo ""

# ============================================================
# Python Tools Checks
# ============================================================
echo "🐍 Python Tools Checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check python installation
if command -v python3 &> /dev/null; then
    check_pass "Python 3 installed"
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    echo "    Version: $python_version"
else
    check_fail "Python 3 not found"
fi

# Check mypy
if python3 -m mypy --version &> /dev/null; then
    check_pass "mypy installed (type checker)"
else
    check_warn "mypy not installed (run: pip install mypy)"
fi

# Check ruff
if python3 -m ruff --version &> /dev/null; then
    check_pass "ruff installed (linter)"
else
    check_warn "ruff not installed (run: pip install ruff)"
fi

# Check black
if python3 -m black --version &> /dev/null; then
    check_pass "black installed (formatter)"
else
    check_warn "black not installed (run: pip install black)"
fi

# Check pytest
if python3 -m pytest --version &> /dev/null; then
    check_pass "pytest installed (test runner)"
else
    check_warn "pytest not installed (run: pip install pytest)"
fi

# Check pytest-cov
if python3 -c "import pytest_cov" 2> /dev/null; then
    check_pass "pytest-cov installed (coverage reporting)"
else
    check_warn "pytest-cov not installed (run: pip install pytest-cov)"
fi

echo ""

# ============================================================
# Documentation Checks
# ============================================================
echo "📚 Documentation Checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check copilot instructions
if [ -f .github/copilot-instructions.md ]; then
    size=$(wc -c < .github/copilot-instructions.md)
    if [ $size -gt 50000 ]; then
        check_pass ".github/copilot-instructions.md exists ($(( size / 1024 ))KB)"
    else
        check_warn ".github/copilot-instructions.md may be incomplete"
    fi
else
    check_fail ".github/copilot-instructions.md not found"
fi

# Check integration guide
if [ -f .github/COPILOT_INTEGRATION.md ]; then
    check_pass ".github/COPILOT_INTEGRATION.md exists (setup guide)"
else
    check_fail ".github/COPILOT_INTEGRATION.md not found"
fi

# Check contributing guide
if [ -f CONTRIBUTING.md ]; then
    check_pass "CONTRIBUTING.md exists"
else
    check_warn "CONTRIBUTING.md not found"
fi

# Check elite standards reference
if [ -f docs/ELITE_STANDARDS_REFERENCE.md ]; then
    check_pass "docs/ELITE_STANDARDS_REFERENCE.md exists (quick reference)"
else
    check_fail "docs/ELITE_STANDARDS_REFERENCE.md not found"
fi

echo ""

# ============================================================
# Project Structure Checks
# ============================================================
echo "📁 Project Structure Checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check directories
[ -d ollama ] && check_pass "ollama/ directory exists" || check_fail "ollama/ not found"
[ -d tests ] && check_pass "tests/ directory exists" || check_fail "tests/ not found"
[ -d .github ] && check_pass ".github/ directory exists" || check_fail ".github/ not found"
[ -d .githooks ] && check_pass ".githooks/ directory exists" || check_fail ".githooks/ not found"
[ -d .vscode ] && check_pass ".vscode/ directory exists" || check_fail ".vscode/ not found"
[ -d docs ] && check_pass "docs/ directory exists" || check_fail "docs/ not found"
[ -d scripts ] && check_pass "scripts/ directory exists" || check_fail "scripts/ not found"

echo ""

# ============================================================
# Summary
# ============================================================
echo "🎯 Verification Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

total=$((checks_passed + checks_failed))
if [ $total -gt 0 ]; then
    percentage=$((checks_passed * 100 / total))
    echo "Passed: $checks_passed / $total ($percentage%)"
fi

echo ""

if [ $checks_failed -eq 0 ]; then
    echo "✅ SETUP COMPLETE - All checks passed!"
    echo ""
    echo "Next steps:"
    echo "  1. Install VS Code extensions (Copilot, Python, Ruff)"
    echo "  2. Configure GPG signing if not done"
    echo "  3. Make your first commit with elite standards"
    echo ""
    exit 0
else
    echo "⚠️  SETUP INCOMPLETE - $checks_failed checks failed"
    echo ""
    echo "Run the following to complete setup:"
    echo "  bash scripts/setup-git-hooks.sh"
    echo "  pip install -r requirements/dev.txt"
    echo "  git config user.signingkey <YOUR_GPG_KEY>"
    echo ""
    exit 1
fi
