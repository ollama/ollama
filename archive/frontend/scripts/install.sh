#!/bin/bash
# Frontend Installation Script
# Installs dependencies and sets up development environment

set -e

echo "🚀 Ollama Frontend Installation"
echo "================================"

# Check Node.js version
NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "❌ Error: Node.js 18+ required (found: $(node -v))"
    exit 1
fi

echo "✅ Node.js version: $(node -v)"
echo "✅ npm version: $(npm -v)"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
npm ci

# Create .env.local if not exists
if [ ! -f .env.local ]; then
    echo ""
    echo "📝 Creating .env.local from template..."
    cp .env.example .env.local
    echo "⚠️  WARNING: Update .env.local with your Firebase credentials!"
fi

# Validate TypeScript configuration
echo ""
echo "🔍 Validating TypeScript configuration..."
npm run type-check || echo "⚠️  Type checking found issues (expected for initial setup)"

# Run linting
echo ""
echo "🔍 Running linter..."
npm run lint || echo "⚠️  Linting found issues (expected for initial setup)"

echo ""
echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "  1. Update .env.local with your Firebase credentials"
echo "  2. Run 'npm run dev' to start development server"
echo "  3. Open http://localhost:3000 in your browser"
echo ""
echo "Documentation: https://github.com/kushin77/ollama/tree/main/frontend"
