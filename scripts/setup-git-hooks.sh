#!/bin/bash
# Setup git hooks for elite standards enforcement
# Usage: bash scripts/setup-git-hooks.sh

set -e

echo "🔧 Setting up git hooks for elite standards..."
echo ""

# Create .githooks directory if it doesn't exist
mkdir -p .githooks

# Make hooks executable
chmod +x .githooks/commit-msg-validate
chmod +x .githooks/pre-commit-elite
chmod +x .githooks/pre-push-elite

# Configure git to use .githooks directory
git config core.hooksPath .githooks

# Create symbolic links in .git/hooks (for backup compatibility)
mkdir -p .git/hooks
ln -sf ../../.githooks/commit-msg-validate .git/hooks/commit-msg 2>/dev/null || true
ln -sf ../../.githooks/pre-commit-elite .git/hooks/pre-commit 2>/dev/null || true
ln -sf ../../.githooks/pre-push-elite .git/hooks/pre-push 2>/dev/null || true

echo "✅ Git hooks configured:"
echo "   - commit-msg validation: checks conventional commit format"
echo "   - pre-commit: runs type checking, linting, formatting, security audit"
echo "   - pre-push: validates branch naming, runs full test suite"
echo ""
echo "📝 To make commits with verification:"
echo "   git commit -S -m 'type(scope): description'  # -S for GPG signing"
echo ""
echo "🚫 To skip hooks (not recommended):"
echo "   git commit --no-verify"
echo ""
echo ""

# Configure commit message template
if [ -f .gitmessage ]; then
    echo "📝 Setting commit message template..."
    git config commit.template .gitmessage
    echo "✅ Commit template configured"
    echo ""
fi

# Show current configuration
echo "📋 Current Git Configuration:"
echo "  core.hooksPath: $(git config core.hooksPath)"
echo "  commit.gpgsign: $(git config commit.gpgsign)"
echo "  user.email: $(git config user.email)"
echo "  user.name: $(git config user.name)"
echo ""

echo "✅ Git setup complete!"
echo ""
echo "📚 Elite Standards Summary:"
echo "  - Commit Format: type(scope): description"
echo "  - Valid Types: feat, fix, refactor, perf, test, docs, infra, security"
echo "  - All commits must be GPG signed: git commit -S"
echo "  - Push frequency: Every 4 hours max"
echo "  - Commit frequency: Every 30 minutes min"
echo "  - All tests must pass before push"
echo ""
echo "🔗 For more information, see .github/copilot-instructions.md"
echo ""
