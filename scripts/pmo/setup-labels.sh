#!/bin/bash
# Setup GitHub Labels Script
# Migrated from: gcp-landing-zone/scripts/pmo/setup-labels.sh
# Purpose: Configure standardized GitHub labels for PMO compliance

set -euo pipefail

# Configuration
REPO="${1:-}"
if [ -z "${REPO}" ]; then
    echo "Usage: $0 <owner/repo>"
    echo "Example: $0 kushin77/ollama"
    exit 1
fi

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "🏷️  GitHub Label Setup"
echo "Repository: ${REPO}"
echo "---"

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) not installed"
    echo "Install: https://cli.github.com/"
    exit 1
fi

# Function to create or update label
create_label() {
    local name=$1
    local color=$2
    local description=$3

    if gh label list --repo "${REPO}" --json name | jq -e ".[] | select(.name == \"${name}\")" > /dev/null 2>&1; then
        echo -e "  ${YELLOW}↻${NC} ${name} (already exists)"
    else
        gh label create "${name}" \
            --repo "${REPO}" \
            --color "${color}" \
            --description "${description}" \
            --force
        echo -e "  ${GREEN}✓${NC} ${name}"
    fi
}

# Type labels
echo ""
echo "📋 Creating type labels..."
create_label "task" "0E8A16" "Work item or feature to implement"
create_label "epic" "5319E7" "Large multi-issue initiative"
create_label "bug" "D73A4A" "Something is broken or not working"
create_label "security" "B60205" "Security vulnerability or concern"
create_label "docs" "0075CA" "Documentation update or addition"
create_label "refactor" "FBCA04" "Code refactoring or cleanup"
create_label "perf" "D4C5F9" "Performance improvement"

# Priority labels
echo ""
echo "🔥 Creating priority labels..."
create_label "priority-p0" "B60205" "Critical - Production down, immediate fix"
create_label "priority-p1" "D93F0B" "High - Major issue, fix within 24h"
create_label "priority-p2" "FBCA04" "Medium - Fix within 1 week"
create_label "priority-p3" "0E8A16" "Low - Nice to have, backlog"

# Component labels
echo ""
echo "🔧 Creating component labels..."
create_label "api" "1D76DB" "API layer changes"
create_label "database" "C5DEF5" "Database or data layer"
create_label "frontend" "D4C5F9" "Frontend or UI changes"
create_label "auth" "FBCA04" "Authentication or authorization"
create_label "monitoring" "0E8A16" "Observability, logging, metrics"
create_label "deployment" "D93F0B" "Deployment or infrastructure"
create_label "backend" "5319E7" "Backend services"

# Effort labels
echo ""
echo "📏 Creating effort labels..."
create_label "trivial-effort" "C2E0C6" "< 2 hours (1 SP)"
create_label "small-effort" "BFD4F2" "2-4 hours (2 SP)"
create_label "medium-effort" "FBCA04" "4-8 hours (3-5 SP)"
create_label "large-effort" "D93F0B" "1-2 days (8 SP)"
create_label "epic-effort" "B60205" "1+ weeks (13+ SP)"

# PMO labels
echo ""
echo "📊 Creating PMO labels..."
create_label "pmo" "1D76DB" "PMO governance related"
create_label "compliance" "FFA500" "Compliance or audit related"
create_label "cost-tracking" "FBCA04" "Cost attribution or optimization"
create_label "governance" "5319E7" "Governance policy related"

# Phase labels
echo ""
echo "🎯 Creating phase labels..."
create_label "phase-1" "0E8A16" "Phase 1 - Foundation"
create_label "phase-2" "1D76DB" "Phase 2 - AI Intelligence"
create_label "phase-3" "5319E7" "Phase 3 - Executive Ops"
create_label "phase-4" "D93F0B" "Phase 4 - Advanced Automation"

# Status labels
echo ""
echo "📍 Creating status labels..."
create_label "in-progress" "FBCA04" "Currently being worked on"
create_label "blocked" "D73A4A" "Blocked by dependency or issue"
create_label "ready-for-review" "0E8A16" "Ready for code review"
create_label "needs-info" "FFA500" "Needs more information"

# Summary
echo ""
echo "---"
echo -e "${GREEN}✅ Label setup complete!${NC}"
echo ""
echo "Total labels created: 35+"
echo "View labels: gh label list --repo ${REPO}"
