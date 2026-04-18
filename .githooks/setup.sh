#!/bin/bash
# Setup Git hooks for Ollama Elite AI Platform
# Run this once to initialize Git hooks

set -e

echo "🔧 Setting up Git hooks for Ollama Elite AI Platform..."
echo ""

# Verify we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ ERROR: Not in a git repository"
    exit 1
fi

# Get git hooks directory
GIT_DIR=$(git rev-parse --git-dir)
HOOKS_DIR="$GIT_DIR/hooks"

# Create hooks directory if it doesn't exist
mkdir -p "$HOOKS_DIR"

# Get the project's .githooks directory
PROJECT_HOOKS_DIR=".githooks"

if [ ! -d "$PROJECT_HOOKS_DIR" ]; then
    echo "❌ ERROR: $PROJECT_HOOKS_DIR directory not found"
    exit 1
fi

# List of hooks to install
HOOKS=("pre-commit" "commit-msg-validate" "post-commit")

# Install hooks
for hook in "${HOOKS[@]}"; do
    SOURCE="$PROJECT_HOOKS_DIR/$hook"
    TARGET="$HOOKS_DIR/$hook"

    if [ ! -f "$SOURCE" ]; then
        echo "⚠️  WARNING: $SOURCE not found, skipping..."
        continue
    fi

    # Copy the hook
    cp "$SOURCE" "$TARGET"
    chmod +x "$TARGET"
    echo "✅ Installed: $hook"
done

echo ""
echo "🔐 Configuring GPG signing for commits..."

# Set git config for automatic signing (optional)
if git config --local user.signingkey > /dev/null 2>&1; then
    echo "✅ GPG signing key already configured"
else
    echo "ℹ️  No GPG signing key configured"
    echo "To configure: git config --local user.signingkey <key-id>"
fi

# Enable commit signing
git config --local commit.gpgSign false  # Set to true if GPG is configured

echo ""
echo "════════════════════════════════════════════════════════════"
echo "✅ Git hooks installed successfully!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Installed hooks:"
echo "  • pre-commit         - Runs tests, linting, type checking"
echo "  • commit-msg-validate - Validates commit message format"
echo "  • post-commit        - Verifies GPG signing (warning only)"
echo ""
echo "Commit message format: type(scope): description"
echo ""
echo "Valid types:"
echo "  feat, fix, refactor, perf, test, docs, infra, security, chore"
echo ""
echo "Example commit:"
echo "  git commit -S -m 'feat(api): add streaming response support'"
echo ""
echo "To bypass checks (not recommended):"
echo "  git commit --no-verify"
echo ""
echo "════════════════════════════════════════════════════════════"
