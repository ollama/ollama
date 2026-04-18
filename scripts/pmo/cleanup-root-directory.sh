#!/usr/bin/env bash
#
# cleanup-root-directory.sh - Clean up root directory files (PMO Mandate)
#
# Migrated from: gcp-landing-zone/scripts/pmo/cleanup-root-directory.sh
# Purpose: Enforce "No Root Chaos" mandate by organizing loose files
#
# Elite FAANG Standards:
# - NO files at root except mandatory configs
# - Everything organized into Level 2+ subdirectories
# - Automated cleanup with safe fallbacks
#
# Usage:
#   ./scripts/pmo/cleanup-root-directory.sh [--dry-run]
#   ./scripts/pmo/cleanup-root-directory.sh --execute

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
FILES_MOVED=0
FILES_SKIPPED=0
ERRORS=0

# Dry run mode (default: true for safety)
DRY_RUN=true

# Parse arguments
for arg in "$@"; do
    case $arg in
        --execute)
            DRY_RUN=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown argument: $arg${NC}"
            echo "Usage: $0 [--dry-run|--execute]"
            exit 1
            ;;
    esac
done

# Header
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}🧹 PMO Root Directory Cleanup${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}⚠️  DRY RUN MODE - No files will be moved${NC}"
    echo -e "${YELLOW}   Use --execute to perform actual cleanup${NC}"
else
    echo -e "${GREEN}✅ EXECUTE MODE - Files will be moved${NC}"
fi
echo ""

# Allowed files at root (mandatory configs only)
ALLOWED_ROOT_FILES=(
    ".gitignore"
    ".gitattributes"
    ".dockerignore"
    ".editorconfig"
    "README.md"
    "LICENSE"
    "CONTRIBUTING.md"
    "CHANGELOG.md"
    "pmo.yaml"
    "pyproject.toml"
    "package.json"
    "go.mod"
    "go.sum"
    "Cargo.toml"
    "docker-compose.yml"
    "docker-compose.yaml"
    "Makefile"
    ".env.example"
    "requirements.txt"
    "setup.py"
    "setup.cfg"
    "poetry.lock"
    "Pipfile"
    "Pipfile.lock"
    "mypy.ini"
    "pytest.ini"
    ".pre-commit-config.yaml"
    "mkdocs.yml"
)

# Categorization rules: extension → target directory
declare -A EXTENSION_MAP
EXTENSION_MAP=(
    [".md"]="docs/"
    [".txt"]="docs/"
    [".pdf"]="docs/"
    [".rst"]="docs/"
    [".yml"]="config/"
    [".yaml"]="config/"
    [".json"]="config/"
    [".toml"]="config/"
    [".ini"]="config/"
    [".cfg"]="config/"
    [".sh"]="scripts/"
    [".bash"]="scripts/"
    [".py"]="scripts/"
    [".js"]="scripts/"
    [".ts"]="scripts/"
    [".sql"]="scripts/"
    [".env"]="config/secrets/"
    [".key"]="config/secrets/"
    [".pem"]="config/secrets/"
    [".crt"]="config/secrets/"
)

# Categorization rules: filename pattern → target directory
declare -A FILENAME_MAP
FILENAME_MAP=(
    ["*SUMMARY*"]="docs/"
    ["*REPORT*"]="docs/"
    ["*MANIFEST*"]="docs/"
    ["*CHECKLIST*"]="docs/"
    ["*APPROVAL*"]="docs/"
    ["*VERIFICATION*"]="docs/"
    ["*COMPLETION*"]="docs/"
    ["*DEPLOYMENT*"]="docs/"
    ["*DELIVERABLES*"]="docs/"
    ["*AUDIT*"]="docs/"
    ["*INDEX*"]="docs/"
    ["*ROADMAP*"]="docs/"
    ["*MANDATE*"]="docs/"
    ["*AUTHORIZATION*"]="docs/"
    ["*CLOSURE*"]="docs/"
    ["docker-compose*"]="docker/"
    ["Dockerfile*"]="docker/"
    [".env*"]="config/"
    ["*.example"]="config/"
)

# Function: Check if file is allowed at root
is_allowed_at_root() {
    local file="$1"
    for allowed in "${ALLOWED_ROOT_FILES[@]}"; do
        if [ "$file" = "$allowed" ]; then
            return 0
        fi
    done
    return 1
}

# Function: Get target directory for file
get_target_directory() {
    local file="$1"
    local extension="${file##*.}"

    # Check filename patterns first (more specific)
    for pattern in "${!FILENAME_MAP[@]}"; do
        if [[ "$file" == $pattern ]]; then
            echo "${FILENAME_MAP[$pattern]}"
            return 0
        fi
    done

    # Check extension mapping
    if [ -n "${EXTENSION_MAP[.$extension]+isset}" ]; then
        echo "${EXTENSION_MAP[.$extension]}"
        return 0
    fi

    # Default: move to docs/ if uncertain
    echo "docs/"
    return 0
}

# Function: Move file to target directory
move_file() {
    local file="$1"
    local target_dir="$2"

    # Ensure target directory exists
    if [ "$DRY_RUN" = false ]; then
        mkdir -p "$target_dir"
    fi

    # Check if target file already exists
    if [ -f "${target_dir}${file}" ]; then
        echo -e "  ${YELLOW}⚠️  Skip: ${file} (already exists in ${target_dir})${NC}"
        ((FILES_SKIPPED++))
        return 1
    fi

    # Move file
    if [ "$DRY_RUN" = true ]; then
        echo -e "  ${BLUE}🔵 Would move: ${file} → ${target_dir}${NC}"
        ((FILES_MOVED++))
    else
        if mv "$file" "${target_dir}${file}"; then
            echo -e "  ${GREEN}✅ Moved: ${file} → ${target_dir}${NC}"
            ((FILES_MOVED++))
        else
            echo -e "  ${RED}❌ Error moving: ${file}${NC}"
            ((ERRORS++))
        fi
    fi
}

# Main cleanup logic
echo "🔍 Scanning root directory..."
echo ""

# Get all files in root (excluding directories and hidden files)
for file in *; do
    # Skip directories
    if [ -d "$file" ]; then
        continue
    fi

    # Skip hidden files (already handled by .gitignore)
    if [[ "$file" == .* ]]; then
        continue
    fi

    # Check if file is allowed at root
    if is_allowed_at_root "$file"; then
        echo -e "  ${GREEN}✅ Keep: ${file} (allowed at root)${NC}"
        continue
    fi

    # Determine target directory
    target_dir=$(get_target_directory "$file")

    # Move file
    move_file "$file" "$target_dir"
done

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}📊 Cleanup Summary${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "Files moved: ${GREEN}${FILES_MOVED}${NC}"
echo -e "Files skipped: ${YELLOW}${FILES_SKIPPED}${NC}"
echo -e "Errors: ${RED}${ERRORS}${NC}"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}⚠️  This was a dry run. Use --execute to apply changes.${NC}"
    exit 0
else
    if [ "$ERRORS" -gt 0 ]; then
        echo -e "${RED}❌ Cleanup completed with errors${NC}"
        exit 1
    else
        echo -e "${GREEN}✅ Cleanup completed successfully${NC}"
        exit 0
    fi
fi
