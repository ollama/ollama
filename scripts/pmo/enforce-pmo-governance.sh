#!/bin/bash
# PMO Governance Enforcement Script
# Migrated from: gcp-landing-zone/scripts/pmo/enforce-pmo-governance.sh
# Purpose: Enforce PMO governance standards across repositories

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PMO_YAML="${REPO_ROOT}/pmo.yaml"
REQUIRED_LABELS=24
MIN_LABELS=20

echo "🔍 PMO Governance Enforcement"
echo "Repository: ${REPO_ROOT}"
echo "---"

# Check 1: pmo.yaml exists
check_pmo_yaml_exists() {
    echo -n "Checking pmo.yaml exists... "
    if [ ! -f "${PMO_YAML}" ]; then
        echo -e "${RED}FAIL${NC}"
        echo "  ❌ pmo.yaml not found"
        echo "  💡 Run: ollama-pmo onboard $(basename ${REPO_ROOT})"
        return 1
    fi
    echo -e "${GREEN}PASS${NC}"
    return 0
}

# Check 2: pmo.yaml has all mandatory labels
check_mandatory_labels() {
    echo -n "Checking mandatory labels... "

    local label_count=0

    # Count non-empty labels
    for key in environment cost_center team managed_by \
               created_by created_date lifecycle_state teardown_date retention_days \
               product component tier compliance \
               version stack backup_strategy monitoring_enabled \
               budget_owner project_code monthly_budget_usd chargeback_unit \
               git_repository git_branch auto_delete; do
        if grep -q "^${key}:" "${PMO_YAML}" && ! grep -q "^${key}:\s*$" "${PMO_YAML}"; then
            ((label_count++))
        fi
    done

    if [ ${label_count} -lt ${MIN_LABELS} ]; then
        echo -e "${RED}FAIL${NC}"
        echo "  ❌ Only ${label_count}/${REQUIRED_LABELS} labels populated"
        echo "  💡 Run: scripts/pmo/validate-pmo-metadata.sh"
        return 1
    fi

    echo -e "${GREEN}PASS${NC} (${label_count}/${REQUIRED_LABELS})"
    return 0
}

# Check 3: GitHub labels configured
check_github_labels() {
    echo -n "Checking GitHub labels... "

    # Check if running in GitHub Actions
    if [ -n "${GITHUB_REPOSITORY:-}" ]; then
        # Count required labels via GitHub API
        local label_count=$(gh label list --json name | jq '. | length')

        if [ ${label_count} -lt 20 ]; then
            echo -e "${YELLOW}WARN${NC}"
            echo "  ⚠️ Only ${label_count} labels configured"
            echo "  💡 Run: scripts/pmo/setup-labels.sh"
            return 0
        fi

        echo -e "${GREEN}PASS${NC} (${label_count} labels)"
    else
        echo -e "${YELLOW}SKIP${NC} (not in GitHub environment)"
    fi

    return 0
}

# Check 4: Required workflows exist
check_workflows() {
    echo -n "Checking PMO workflows... "

    local workflow_dir="${REPO_ROOT}/.github/workflows"
    local required_workflows=(
        "pmo-validation.yml"
        "compliance-check.yml"
    )

    local missing=0
    for workflow in "${required_workflows[@]}"; do
        if [ ! -f "${workflow_dir}/${workflow}" ]; then
            ((missing++))
        fi
    done

    if [ ${missing} -gt 0 ]; then
        echo -e "${YELLOW}WARN${NC}"
        echo "  ⚠️ ${missing} required workflows missing"
        echo "  💡 Copy from templates/pmo/workflows/"
        return 0
    fi

    echo -e "${GREEN}PASS${NC}"
    return 0
}

# Check 5: GPG signing enabled
check_gpg_signing() {
    echo -n "Checking GPG commit signing... "

    if ! git config --get commit.gpgsign | grep -q "true"; then
        echo -e "${YELLOW}WARN${NC}"
        echo "  ⚠️ GPG signing not enabled"
        echo "  💡 Run: git config commit.gpgsign true"
        return 0
    fi

    echo -e "${GREEN}PASS${NC}"
    return 0
}

# Check 6: Pre-commit hooks installed
check_pre_commit_hooks() {
    echo -n "Checking pre-commit hooks... "

    if [ ! -f "${REPO_ROOT}/.git/hooks/pre-commit" ]; then
        echo -e "${YELLOW}WARN${NC}"
        echo "  ⚠️ Pre-commit hooks not installed"
        echo "  💡 Run: cp templates/pmo/hooks/pre-commit .git/hooks/"
        return 0
    fi

    echo -e "${GREEN}PASS${NC}"
    return 0
}

# Main enforcement function
enforce_governance() {
    local failed=0

    check_pmo_yaml_exists || ((failed++))
    check_mandatory_labels || ((failed++))
    check_github_labels || ((failed++))
    check_workflows || ((failed++))
    check_gpg_signing || ((failed++))
    check_pre_commit_hooks || ((failed++))

    echo "---"

    if [ ${failed} -gt 0 ]; then
        echo -e "${RED}❌ Governance check failed: ${failed} critical issues${NC}"
        echo ""
        echo "Quick fix:"
        echo "  1. Run: ollama-pmo onboard $(basename ${REPO_ROOT})"
        echo "  2. Review and commit pmo.yaml"
        echo "  3. Re-run: $0"
        return 1
    else
        echo -e "${GREEN}✅ All governance checks passed${NC}"
        return 0
    fi
}

# Run enforcement
enforce_governance
exit $?
