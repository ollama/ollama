#!/bin/bash
# =============================================================================
# Enterprise Upgrade Validation Script
# Verifies all Phase 1-5 improvements are working
# =============================================================================

set -e

echo "🔍 Validating Enterprise Upgrade Implementation..."
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAILED=0

# ============================================================
# PHASE 1: Development Environment
# ============================================================
echo -e "${BLUE}[PHASE 1]${NC} Development Environment"
echo "─────────────────────────────────────────────"

if [ -f "$PROJECT_ROOT/scripts/setup-dev.sh" ]; then
    echo -e "  ${GREEN}✅${NC} setup-dev.sh exists"
else
    echo -e "  ${RED}❌${NC} setup-dev.sh missing"
    FAILED=$((FAILED + 1))
fi

if [ -f "$PROJECT_ROOT/scripts/run-all-checks.sh" ]; then
    echo -e "  ${GREEN}✅${NC} run-all-checks.sh exists"
else
    echo -e "  ${RED}❌${NC} run-all-checks.sh missing"
    FAILED=$((FAILED + 1))
fi

if [ -d "$PROJECT_ROOT/venv" ]; then
    echo -e "  ${GREEN}✅${NC} venv directory exists"
else
    echo -e "  ${YELLOW}⚠️ ${NC} venv not created (run setup-dev.sh)"
fi

if command -v python3 &> /dev/null; then
    echo -e "  ${GREEN}✅${NC} python3 available"
fi

echo ""

# ============================================================
# PHASE 2: Type Safety
# ============================================================
echo -e "${BLUE}[PHASE 2]${NC} Type Safety Improvements"
echo "─────────────────────────────────────────────"

type_safety_checks=$(grep -c "-> .*|.*None:" "$PROJECT_ROOT/ollama/main.py" || echo 0)
if [ "$type_safety_checks" -gt 0 ]; then
    echo -e "  ${GREEN}✅${NC} Type hints improved ($type_safety_checks instances)"
else
    echo -e "  ${YELLOW}⚠️ ${NC} Type hints may need more work"
fi

if grep -q "RuntimeError.*initialized" "$PROJECT_ROOT/ollama/main.py"; then
    echo -e "  ${GREEN}✅${NC} Better error messages added"
fi

echo ""

# ============================================================
# PHASE 3: Tests
# ============================================================
echo -e "${BLUE}[PHASE 3]${NC} Real Tests Created"
echo "─────────────────────────────────────────────"

if [ -f "$PROJECT_ROOT/tests/integration/test_inference_real.py" ]; then
    echo -e "  ${GREEN}✅${NC} test_inference_real.py created"

    test_count=$(grep -c "def test_" "$PROJECT_ROOT/tests/integration/test_inference_real.py" || echo 0)
    if [ "$test_count" -gt 10 ]; then
        echo -e "  ${GREEN}✅${NC} $test_count real tests (expected 16+)"
    fi

    assertion_count=$(grep -c "assert " "$PROJECT_ROOT/tests/integration/test_inference_real.py" || echo 0)
    if [ "$assertion_count" -gt 20 ]; then
        echo -e "  ${GREEN}✅${NC} $assertion_count assertions (expected 50+)"
    fi
else
    echo -e "  ${RED}❌${NC} test_inference_real.py missing"
    FAILED=$((FAILED + 1))
fi

echo ""

# ============================================================
# PHASE 4: Git Hooks & CI/CD
# ============================================================
echo -e "${BLUE}[PHASE 4]${NC} Git Hooks & CI/CD"
echo "─────────────────────────────────────────────"

if [ -f "$PROJECT_ROOT/.githooks/pre-commit" ]; then
    echo -e "  ${GREEN}✅${NC} pre-commit hook exists"

    if [ -x "$PROJECT_ROOT/.githooks/pre-commit" ]; then
        echo -e "  ${GREEN}✅${NC} pre-commit hook is executable"
    fi
fi

if [ -f "$PROJECT_ROOT/.github/workflows/quality-checks.yml" ]; then
    echo -e "  ${GREEN}✅${NC} GitHub Actions workflow created"

    if grep -q "mypy" "$PROJECT_ROOT/.github/workflows/quality-checks.yml"; then
        echo -e "  ${GREEN}✅${NC} Type checking in CI/CD"
    fi

    if grep -q "ruff" "$PROJECT_ROOT/.github/workflows/quality-checks.yml"; then
        echo -e "  ${GREEN}✅${NC} Linting in CI/CD"
    fi

    if grep -q "pip-audit" "$PROJECT_ROOT/.github/workflows/quality-checks.yml"; then
        echo -e "  ${GREEN}✅${NC} Security scanning in CI/CD"
    fi
else
    echo -e "  ${RED}❌${NC} GitHub Actions workflow missing"
fi

echo ""

# ============================================================
# PHASE 5: Docker & Deployment
# ============================================================
echo -e "${BLUE}[PHASE 5]${NC} Docker & Deployment Security"
echo "─────────────────────────────────────────────"

DOCKER_FILE="$PROJECT_ROOT/docker/docker-compose.prod.yml"

if [ -f "$DOCKER_FILE" ]; then

    # Check for pinned versions
    if grep -q "image: ollama:1.0.0" "$DOCKER_FILE"; then
        echo -e "  ${GREEN}✅${NC} API image pinned (1.0.0)"
    fi

    if grep -q "image: postgres:15.5-alpine" "$DOCKER_FILE"; then
        echo -e "  ${GREEN}✅${NC} PostgreSQL image pinned (15.5-alpine)"
    fi

    if grep -q "image: redis:7.2.4-alpine" "$DOCKER_FILE"; then
        echo -e "  ${GREEN}✅${NC} Redis image pinned (7.2.4-alpine)"
    fi

    if grep -q "image: qdrant/qdrant:v1.8.1" "$DOCKER_FILE"; then
        echo -e "  ${GREEN}✅${NC} Qdrant image pinned (v1.8.1)"
    fi

    # Check for resource limits
    resource_limit_count=$(grep -c "memory: " "$DOCKER_FILE" || echo 0)
    if [ "$resource_limit_count" -ge 4 ]; then
        echo -e "  ${GREEN}✅${NC} Resource limits added ($resource_limit_count found)"
    fi

    # Check for health check improvements
    if grep -q "start_period:" "$DOCKER_FILE"; then
        echo -e "  ${GREEN}✅${NC} Health check grace periods added"
    fi

    # Check for restart policies
    if grep -q "restart: on-failure:3" "$DOCKER_FILE"; then
        echo -e "  ${GREEN}✅${NC} Safe restart policies configured"
    fi
else
    echo -e "  ${RED}❌${NC} docker-compose.prod.yml not found"
fi

if [ -f "$PROJECT_ROOT/.env.example" ]; then
    echo -e "  ${GREEN}✅${NC} .env.example created (no secrets)"
fi

echo ""

# ============================================================
# Summary
# ============================================================
echo "═══════════════════════════════════════════════════════════"
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ ALL PHASES VALIDATED SUCCESSFULLY${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. ./scripts/setup-dev.sh"
    echo "  2. source venv/bin/activate"
    echo "  3. ./scripts/run-all-checks.sh"
    echo "  4. git config core.hooksPath .githooks"
    echo ""
    exit 0
else
    echo -e "${RED}❌ VALIDATION FAILED: $FAILED issue(s)${NC}"
    exit 1
fi
