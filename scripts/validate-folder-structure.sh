#!/bin/bash
# Folder structure validation script
# Enforces clean root directory with no loose status/report files
# Usage: bash scripts/validate-folder-structure.sh

set -e

echo "📁 Validating Repository Folder Structure..."
echo ""

# Define allowed files at root (whitelist)
allowed_root_files=(
    "README.md"
    "CONTRIBUTING.md"
    "LICENSE"
    "CHANGELOG.md"
    "pyproject.toml"
    "setup.py"
    "Dockerfile"
    "docker-compose.yml"
    "docker-compose.prod.yml"
    "docker-compose.minimal.yml"
    "docker-compose.elite.yml"
    ".gitignore"
    ".pre-commit-config.yaml"
    "alembic.ini"
    "test_server.py"
    "verify-completion.sh"
    ".copilot-instructions"
    ".coverage"
    "IMPLEMENTATION_COMPLETE.md"
    "ELITE_STANDARDS_EXECUTIVE_SUMMARY.md"
    "ELITE_STANDARDS_IMPLEMENTATION_COMPLETE.md"
    "DEEP_SCAN_ELITE_STANDARDS_REPORT.md"
    "FOLDER_STRUCTURE_ENFORCEMENT_COMPLETE.md"
)

# Check for required directories
required_dirs=(
    "ollama"
    "tests"
    "docs"
    "scripts"
    ".github"
    ".githooks"
    ".vscode"
    "config"
    "alembic"
)

violations=0

# ============================================================
# Check Required Directories Exist
# ============================================================
echo "🔍 Checking required directories..."
for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "  ✅ $dir/"
    else
        echo "  ❌ $dir/ - MISSING"
        ((violations++))
    fi
done

echo ""

# ============================================================
# Check for Loose Files at Root
# ============================================================
echo "🔍 Checking for loose files at root..."

# Get all files in root (excluding hidden files and directories)
root_files=$(find . -maxdepth 1 -type f ! -name ".*" -printf "%f\n" 2>/dev/null)

loose_files=()
while IFS= read -r file; do
    # Skip if empty
    [ -z "$file" ] && continue

    # Check if file is in allowed list
    allowed=false
    for allowed_file in "${allowed_root_files[@]}"; do
        if [[ "$file" == "$allowed_file" ]]; then
            allowed=true
            break
        fi
    done

    # If not allowed and is a documentation/report file, flag it
    if ! $allowed; then
        if [[ "$file" =~ \.(md|txt)$ ]]; then
            loose_files+=("$file")
        elif [[ "$file" =~ \.(py|sh|yaml|yml|json)$ ]]; then
            # Scripts and configs should be in subdirectories
            loose_files+=("$file")
        fi
    fi
done <<< "$root_files"

if [ ${#loose_files[@]} -gt 0 ]; then
    echo "  ❌ Found ${#loose_files[@]} loose file(s):"
    for file in "${loose_files[@]}"; do
        echo "     - $file"
    done
    ((violations++))
else
    echo "  ✅ No loose files (all properly organized)"
fi

echo ""

# ============================================================
# Check docs/ Structure
# ============================================================
echo "🔍 Checking docs/ structure..."

if [ -d "docs" ]; then
    if [ -d "docs/reports" ]; then
        echo "  ✅ docs/reports/ exists (for archived status reports)"
    else
        echo "  ⚠️  docs/reports/ missing (recommended for status reports)"
    fi

    if [ -d "docs/archive" ]; then
        echo "  ✅ docs/archive/ exists (for historical documentation)"
    fi
else
    echo "  ❌ docs/ directory missing"
    ((violations++))
fi

echo ""

# ============================================================
# Check for Proper Organization
# ============================================================
echo "🔍 Checking file organization..."

# Check if Python code is in ollama/
if [ -d "ollama" ]; then
    py_files=$(find ollama -name "*.py" 2>/dev/null | wc -l)
    echo "  ✅ ollama/ contains $py_files Python files"
else
    echo "  ❌ ollama/ directory missing"
    ((violations++))
fi

# Check if tests are in tests/
if [ -d "tests" ]; then
    test_files=$(find tests -name "test_*.py" 2>/dev/null | wc -l)
    echo "  ✅ tests/ contains $test_files test files"
else
    echo "  ❌ tests/ directory missing"
    ((violations++))
fi

# Check if scripts are in scripts/
if [ -d "scripts" ]; then
    script_files=$(find scripts -name "*.sh" 2>/dev/null | wc -l)
    echo "  ✅ scripts/ contains $script_files shell scripts"
else
    echo "  ⚠️  scripts/ directory missing"
fi

echo ""

# ============================================================
# Summary
# ============================================================
echo "═══════════════════════════════════════════════════════════"

if [ $violations -eq 0 ]; then
    echo "✅ FOLDER STRUCTURE VALID"
    echo ""
    echo "All checks passed:"
    echo "  ✓ Required directories present"
    echo "  ✓ No loose files at root"
    echo "  ✓ Files properly organized"
    echo ""
    exit 0
else
    echo "❌ FOLDER STRUCTURE VIOLATIONS: $violations issue(s) found"
    echo ""
    echo "To fix loose files at root, run:"
    echo "  bash scripts/cleanup-root-directory.sh"
    echo ""
    echo "Elite Standards require:"
    echo "  - Essential files only at root (README, LICENSE, etc.)"
    echo "  - Status reports in docs/reports/"
    echo "  - Documentation in docs/"
    echo "  - Scripts in scripts/"
    echo "  - Python code in ollama/"
    echo "  - Tests in tests/"
    echo ""
    exit 1
fi
