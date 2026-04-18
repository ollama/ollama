#!/bin/bash
# =============================================================================
# Development Environment Setup Script
# Sets up virtual environment and installs all dependencies
# Usage: ./scripts/setup-dev.sh
# =============================================================================

set -e  # Exit on first error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🔧 Setting up Ollama development environment..."
echo "Project root: $PROJECT_ROOT"

# Step 1: Remove existing venv if present
if [ -d "$PROJECT_ROOT/venv" ]; then
    echo "⚠️  Removing existing venv..."
    rm -rf "$PROJECT_ROOT/venv"
fi

# Step 2: Create virtual environment
echo "📦 Creating Python 3.11+ virtual environment..."
python3 -m venv "$PROJECT_ROOT/venv"

# Step 3: Activate venv
source "$PROJECT_ROOT/venv/bin/activate"
echo "✅ Virtual environment activated"

# Step 4: Upgrade pip, setuptools, wheel
echo "🚀 Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

# Step 5: Install project with dev dependencies
echo "📚 Installing project with dev dependencies..."
cd "$PROJECT_ROOT"
pip install -e ".[dev]"

# Step 6: Verify all tools are installed
echo "🔍 Verifying installation..."
tools=("python" "pytest" "mypy" "ruff" "black" "pip-audit")
for tool in "${tools[@]}"; do
    if command -v "$tool" &> /dev/null; then
        version=$($tool --version 2>&1 | head -1)
        echo "  ✅ $tool: $version"
    else
        echo "  ❌ $tool: NOT FOUND"
        exit 1
    fi
done

# Step 7: Setup git hooks
echo "🎣 Setting up git hooks..."
git config core.hooksPath .githooks
chmod +x .githooks/*
echo "  ✅ Git hooks configured"

# Step 8: Show next steps
echo ""
echo "✅ Development environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Run all checks: ./scripts/run-all-checks.sh"
echo "  3. Start dev server: python -m uvicorn ollama.main:app --reload"
echo ""
echo "💡 Key commands:"
echo "  pytest tests/ -v --cov=ollama           # Run tests with coverage"
echo "  mypy ollama/ --strict                   # Type check"
echo "  ruff check ollama/                      # Lint check"
echo "  black ollama/ tests/                    # Format code"
echo "  pip-audit                               # Security audit"
