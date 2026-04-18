#!/bin/bash
# =============================================================================
# Run All Quality Checks
# Runs pytest, mypy, ruff, black, and pip-audit in sequence
# MUST pass before any commit
# =============================================================================

set -e  # Exit on first error

echo "🔬 Running all quality checks..."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Check 1: Type checking
echo "${YELLOW}[1/5]${NC} Running mypy (type checking)..."
if python -m mypy ollama/ --strict; then
    echo "${GREEN}✅ Type checking passed${NC}"
else
    echo "${RED}❌ Type checking failed${NC}"
    exit 1
fi
echo ""

# Check 2: Linting
echo "${YELLOW}[2/5]${NC} Running ruff (linting)..."
if python -m ruff check ollama/ tests/; then
    echo "${GREEN}✅ Linting passed${NC}"
else
    echo "${RED}❌ Linting failed${NC}"
    exit 1
fi
echo ""

# Check 3: Code formatting
echo "${YELLOW}[3/5]${NC} Checking black (code formatting)..."
if python -m black --check ollama/ tests/ --line-length=100; then
    echo "${GREEN}✅ Code formatting check passed${NC}"
else
    echo "${RED}❌ Code formatting check failed${NC}"
    echo "Run: black ollama/ tests/ --line-length=100"
    exit 1
fi
echo ""

# Check 4: Tests with coverage
echo "${YELLOW}[4/5]${NC} Running pytest (tests with coverage)..."
if python -m pytest tests/ -v --cov=ollama --cov-report=term-missing --cov-report=html --cov-fail-under=90; then
    echo "${GREEN}✅ Tests passed (≥90% coverage)${NC}"
else
    echo "${RED}❌ Tests failed or coverage < 90%${NC}"
    exit 1
fi
echo ""

# Check 5: Security audit
echo "${YELLOW}[5/5]${NC} Running pip-audit (security audit)..."
if pip-audit; then
    echo "${GREEN}✅ Security audit passed${NC}"
else
    echo "${RED}❌ Security audit found vulnerabilities${NC}"
    exit 1
fi
echo ""

echo "${GREEN}🎉 All checks passed! Code is production-ready.${NC}"
