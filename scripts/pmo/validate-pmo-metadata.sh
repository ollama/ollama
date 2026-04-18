#!/bin/bash
# PMO Metadata Validation Script
# Migrated from: gcp-landing-zone/scripts/pmo/validate-pmo-metadata.sh
# Purpose: Validate pmo.yaml against schema

set -euo pipefail

# Configuration
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PMO_YAML="${REPO_ROOT}/pmo.yaml"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "🔍 PMO Metadata Validation"
echo "File: ${PMO_YAML}"
echo "---"

# Check if pmo.yaml exists
if [ ! -f "${PMO_YAML}" ]; then
    echo -e "${RED}❌ ERROR: pmo.yaml not found${NC}"
    exit 1
fi

# Validate YAML syntax
echo -n "Validating YAML syntax... "
if ! python3 -c "import yaml; yaml.safe_load(open('${PMO_YAML}'))" 2>/dev/null; then
    echo -e "${RED}FAIL${NC}"
    echo "  ❌ Invalid YAML syntax"
    exit 1
fi
echo -e "${GREEN}PASS${NC}"

# Validate mandatory labels
errors=0
warnings=0

validate_label() {
    local category=$1
    local label=$2
    local required=${3:-true}

    local value=$(grep "^${label}:" "${PMO_YAML}" | cut -d: -f2- | xargs || echo "")

    if [ -z "${value}" ]; then
        if [ "${required}" = "true" ]; then
            echo -e "  ${RED}❌ ${label}: MISSING (required)${NC}"
            ((errors++))
        else
            echo -e "  ${YELLOW}⚠️ ${label}: empty (optional)${NC}"
            ((warnings++))
        fi
        return 1
    fi

    echo -e "  ${GREEN}✓${NC} ${label}: ${value}"
    return 0
}

# Organizational labels (4 required)
echo ""
echo "📋 Organizational Labels:"
validate_label "org" "environment" true
validate_label "org" "cost_center" true
validate_label "org" "team" true
validate_label "org" "managed_by" true

# Lifecycle labels (5 required)
echo ""
echo "♻️ Lifecycle Labels:"
validate_label "lifecycle" "created_by" true
validate_label "lifecycle" "created_date" true
validate_label "lifecycle" "lifecycle_state" true
validate_label "lifecycle" "teardown_date" false
validate_label "lifecycle" "retention_days" true

# Business labels (4 required)
echo ""
echo "💼 Business Labels:"
validate_label "business" "product" true
validate_label "business" "component" true
validate_label "business" "tier" true
validate_label "business" "compliance" false

# Technical labels (4 required)
echo ""
echo "🔧 Technical Labels:"
validate_label "technical" "version" true
validate_label "technical" "stack" true
validate_label "technical" "backup_strategy" true
validate_label "technical" "monitoring_enabled" true

# Financial labels (4 required)
echo ""
echo "💰 Financial Labels:"
validate_label "financial" "budget_owner" true
validate_label "financial" "project_code" true
validate_label "financial" "monthly_budget_usd" true
validate_label "financial" "chargeback_unit" true

# Git labels (3 required)
echo ""
echo "🔀 Git Labels:"
validate_label "git" "git_repository" true
validate_label "git" "git_branch" true
validate_label "git" "auto_delete" true

# Summary
echo ""
echo "---"
echo "📊 Validation Summary:"
echo "  Total labels: 24 mandatory"
echo "  Errors: ${errors}"
echo "  Warnings: ${warnings}"

if [ ${errors} -gt 0 ]; then
    echo ""
    echo -e "${RED}❌ Validation failed with ${errors} error(s)${NC}"
    echo ""
    echo "💡 Fix missing labels in pmo.yaml or run:"
    echo "   ollama-pmo onboard $(basename ${REPO_ROOT}) --regenerate"
    exit 1
elif [ ${warnings} -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}⚠️ Validation passed with ${warnings} warning(s)${NC}"
    exit 0
else
    echo ""
    echo -e "${GREEN}✅ All validations passed!${NC}"
    exit 0
fi
