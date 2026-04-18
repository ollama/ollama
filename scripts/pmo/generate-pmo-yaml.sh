#!/bin/bash
# Generate PMO YAML Script
# Migrated from: gcp-landing-zone/scripts/pmo/generate-pmo-yaml.sh
# Purpose: Generate pmo.yaml with intelligent defaults

set -euo pipefail

# Configuration
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PMO_YAML="${REPO_ROOT}/pmo.yaml"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "📝 PMO YAML Generator"
echo "Repository: $(basename "${REPO_ROOT}")"
echo "---"

# Check if pmo.yaml already exists
if [ -f "${PMO_YAML}" ]; then
    echo -e "${YELLOW}⚠️  pmo.yaml already exists${NC}"
    read -p "Overwrite? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Detect values from repository
detect_git_repo() {
    git config --get remote.origin.url | sed 's/\.git$//' | sed 's|https://||' | sed 's|git@github.com:|github.com/|'
}

detect_git_branch() {
    git branch --show-current || echo "main"
}

detect_created_by() {
    git log --reverse --format="%ae" | head -n1 || echo "$(git config user.email)"
}

detect_created_date() {
    git log --reverse --format="%ad" --date=short | head -n1 || date +%Y-%m-%d
}

detect_stack() {
    if [ -f "pyproject.toml" ]; then
        python_version=$(grep -oP 'python = "\^?\K[0-9.]+' pyproject.toml | head -n1)
        if grep -q "fastapi" pyproject.toml; then
            echo "python-${python_version}-fastapi"
        else
            echo "python-${python_version}"
        fi
    elif [ -f "package.json" ]; then
        node_version=$(grep -oP '"node": "\^?\K[0-9.]+' package.json | head -n1 || echo "20")
        echo "nodejs-${node_version}"
    elif [ -f "go.mod" ]; then
        go_version=$(grep -oP 'go \K[0-9.]+' go.mod | head -n1)
        echo "golang-${go_version}"
    else
        echo "unknown"
    fi
}

detect_version() {
    if [ -f "pyproject.toml" ]; then
        grep -oP 'version = "\K[^"]+' pyproject.toml | head -n1 || echo "0.1.0"
    elif [ -f "package.json" ]; then
        grep -oP '"version": "\K[^"]+' package.json | head -n1 || echo "0.1.0"
    else
        echo "0.1.0"
    fi
}

# Generate pmo.yaml
echo -e "${BLUE}Generating pmo.yaml...${NC}"

cat > "${PMO_YAML}" << EOF
# PMO Metadata
# Auto-generated: $(date +%Y-%m-%d)
# Repository: $(basename "${REPO_ROOT}")

# Organizational (4 required)
environment: "development"  # production|staging|development|sandbox
cost_center: "engineering"  # engineering|ai-ml|data|infra
team: "platform-engineering"  # Your team name
managed_by: "terraform"  # terraform|manual|cloudformation

# Lifecycle (5 required)
created_by: "$(detect_created_by)"
created_date: "$(detect_created_date)"
lifecycle_state: "active"  # active|maintenance|sunset|archived
teardown_date: "none"  # YYYY-MM-DD or none
retention_days: "3650"  # 365|3650|7300 (1yr|10yr|20yr)

# Business (4 required)
product: "$(basename "${REPO_ROOT}")"  # Product name
component: "api-server"  # api|database|frontend|auth|monitoring
tier: "high"  # critical|high|medium|low
compliance: "none"  # sox|hipaa|pci|gdpr|none

# Technical (4 required)
version: "$(detect_version)"
stack: "$(detect_stack)"
backup_strategy: "daily"  # daily|weekly|monthly|none
monitoring_enabled: "true"  # true|false

# Financial (4 required)
budget_owner: "$(detect_created_by)"
project_code: "$(echo $(basename "${REPO_ROOT}") | tr '[:lower:]' '[:upper:]')-$(date +%Y)-001"
monthly_budget_usd: "500"  # Estimated monthly cost (USD)
chargeback_unit: "engineering"  # Team or cost center for chargeback

# Git (3 required)
git_repository: "$(detect_git_repo)"
git_branch: "$(detect_git_branch)"
auto_delete: "false"  # true|false

# Notes:
# - Review and adjust all values above
# - Ensure budget_owner email is correct
# - Update monthly_budget_usd with realistic estimate
# - Set compliance if applicable (sox, hipaa, pci, gdpr)
# - Adjust tier based on production criticality
EOF

echo -e "${GREEN}✅ pmo.yaml generated${NC}"
echo ""
echo "📋 Next steps:"
echo "  1. Review ${PMO_YAML}"
echo "  2. Adjust values as needed"
echo "  3. Validate: ./scripts/pmo/validate-pmo-metadata.sh"
echo "  4. Commit: git add pmo.yaml && git commit -S -m 'chore(pmo): Add PMO metadata'"
echo ""
echo "💡 Tip: Run 'ollama-pmo onboard --auto' for AI-powered generation"
