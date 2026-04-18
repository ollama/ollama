#!/bin/bash

set -euo pipefail

# FAANG-Elite Development Environment Setup
# Version: 3.0.0
# Purpose: Configure development environment to Top 0.01% standards

echo "=========================================="
echo "FAANG-Elite Development Setup (v3.0.0)"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Check Python version (3.11+)
echo -e "${BLUE}[1/8] Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
MIN_VERSION="3.11"

if [ "$(printf '%s\n' "$MIN_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$MIN_VERSION" ]; then
    echo -e "${GREEN}✓ Python $PYTHON_VERSION (required: 3.11+)${NC}"
else
    echo -e "${YELLOW}⚠ Python $PYTHON_VERSION detected. Recommended: 3.11+${NC}"
fi

# 2. Create virtual environment
echo ""
echo -e "${BLUE}[2/8] Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate venv
source venv/bin/activate

# 3. Upgrade pip and install wheel
echo ""
echo -e "${BLUE}[3/8] Upgrading pip and installing build tools...${NC}"
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo -e "${GREEN}✓ Build tools installed${NC}"

# 4. Install development dependencies
echo ""
echo -e "${BLUE}[4/8] Installing development dependencies...${NC}"
if [ -f "requirements/dev.txt" ]; then
    pip install -r requirements/dev.txt > /dev/null 2>&1
    echo -e "${GREEN}✓ Dev dependencies installed${NC}"
elif [ -f "requirements.txt" ]; then
    pip install -r requirements.txt > /dev/null 2>&1
    echo -e "${GREEN}✓ Dependencies installed${NC}"
fi

# 5. Install pre-commit hooks
echo ""
echo -e "${BLUE}[5/8] Setting up pre-commit hooks...${NC}"
if [ -f ".pre-commit-config.yaml" ]; then
    pre-commit install > /dev/null 2>&1
    pre-commit run --all-files 2>/dev/null || true
    echo -e "${GREEN}✓ Pre-commit hooks configured${NC}"
else
    echo -e "${YELLOW}⚠ No .pre-commit-config.yaml found${NC}"
fi

# 6. Configure Git for GPG signing (optional)
echo ""
echo -e "${BLUE}[6/8] Git configuration...${NC}"
echo -e "${YELLOW}Note: To enable GPG commit signing, run:${NC}"
echo "  git config --global commit.gpgSign true"
echo "  git config --global user.signingkey YOUR_KEY_ID"

# 7. Run initial checks
echo ""
echo -e "${BLUE}[7/8] Running initial code quality checks...${NC}"
if command -v mypy &> /dev/null; then
    echo "  → Running type checking (mypy)..."
    mypy ollama/ --strict 2>/dev/null || echo "    (initial issues may exist)"
    echo -e "${GREEN}  ✓ Type checking configured${NC}"
fi

if command -v ruff &> /dev/null; then
    echo "  → Running linting (ruff)..."
    ruff check ollama/ --exit-zero 2>/dev/null
    echo -e "${GREEN}  ✓ Linting configured${NC}"
fi

# 8. Display summary
echo ""
echo -e "${BLUE}[8/8] Setup complete!${NC}"
echo ""
echo -e "${GREEN}=========================================="
echo "Your environment is ready for development"
echo "==========================================${NC}"
echo ""
echo "Quick Start Commands:"
echo "  1. Activate venv: source venv/bin/activate"
echo "  2. Run tests: pytest tests/ -v --cov=ollama"
echo "  3. Run type check: mypy ollama/ --strict"
echo "  4. Run linting: ruff check ollama/"
echo "  5. Format code: black ollama/ tests/"
echo "  6. Start server: uvicorn ollama.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "Important:"
echo "  • Use REAL_IP (not localhost) for development"
echo "  • All commits must be signed (-S flag)"
echo "  • Pre-commit hooks will enforce standards automatically"
echo "  • 95%+ test coverage required for new code"
echo ""
echo "Documentation:"
echo "  • Standards: .github/FAANG-ELITE-STANDARDS.md"
echo "  • Folder Structure: .github/FOLDER-STRUCTURE-STANDARDS.md"
echo "  • Copilot Instructions: .github/copilot-instructions.md"
echo ""
