#!/bin/bash
# Elite Push Script - Enforces high standards for committing and pushing to main
# Mandatory Checks: Folder Structure, Type Checking, Linting, Security Audit, Tests (90%+ coverage)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Set Python command
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${PROJECT_ROOT}/venv/bin/python"
VENV_PYTEST="${PROJECT_ROOT}/venv/bin/pytest"
VENV_MYPY="${PROJECT_ROOT}/venv/bin/mypy"
VENV_RUFF="${PROJECT_ROOT}/venv/bin/ruff"
VENV_PIPAUDIT="${PROJECT_ROOT}/venv/bin/pip-audit"

echo -e "${BLUE}🚀 Starting Elite Push process...${NC}"

# 1. Folder Structure Validation
echo -e "${YELLOW}🔍 Step 1: Validating folder structure...${NC}"
$VENV_PYTHON "$PROJECT_ROOT/scripts/validate_folder_structure.py" --strict
echo -e "${GREEN}✅ Folder structure is compliant.${NC}"

# 2. Type Checking
echo -e "${YELLOW}🔍 Step 2: Running type checking (mypy)...${NC}"
$VENV_MYPY "$PROJECT_ROOT/ollama/" --strict
echo -e "${GREEN}✅ Type checking passed.${NC}"

# 3. Linting
echo -e "${YELLOW}🔍 Step 3: Running linting (ruff)...${NC}"
$VENV_RUFF check "$PROJECT_ROOT/ollama/"
echo -e "${GREEN}✅ Linting passed.${NC}"

# 4. Security Audit
echo -e "${YELLOW}🔍 Step 4: Running security audit (pip-audit)...${NC}"
$VENV_PIPAUDIT
echo -e "${GREEN}✅ Security audit passed.${NC}"

# 5. Tests and Coverage
echo -e "${YELLOW}🔍 Step 5: Running tests and checking coverage (pytest)...${NC}"
$VENV_PYTEST "$PROJECT_ROOT/tests/" -v --cov=ollama --cov-report=term-missing --cov-fail-under=90
echo -e "${GREEN}✅ All tests passed with >=90% coverage.${NC}"

# 6. Git Status and Staging
echo -e "${YELLOW}📦 Step 6: Staging all changes...${NC}"
git add .
STAGED_CHANGES=$(git diff --cached --name-only)

if [ -z "$STAGED_CHANGES" ]; then
    echo -e "${RED}❌ No changes to commit.${NC}"
    exit 0
fi

echo -e "${GREEN}✅ Changes staged for commit.${NC}"

# 7. Commit with Mandatory Format
echo -e "${YELLOW}📝 Step 7: Committing changes...${NC}"
# Use the mandatory format type(scope): description
# Since this is an automated script, we might need a way to pass the message.
# For now, we'll try to determine it or ask the user, but since I'm the agent,
# I'll use a standard "feat(infra): apply elite standards and restructure" or similar.

if [ -z "$1" ]; then
    COMMIT_MSG="feat(infra): enforce elite standards and complete filesystem restructure"
else
    COMMIT_MSG="$1"
fi

echo -e "${BLUE}Commit message: $COMMIT_MSG${NC}"

# Mandatory GPG Signing
git commit -S -m "$COMMIT_MSG"
echo -e "${GREEN}✅ Changes committed with GPG signature.${NC}"

# 8. Push to main
echo -e "${YELLOW}📤 Step 8: Pushing to main...${NC}"
git push origin main
echo -e "${GREEN}✅ Successfully pushed to main!${NC}"

echo -e "${BLUE}✨ Elite Push process completed successfully.${NC}"
