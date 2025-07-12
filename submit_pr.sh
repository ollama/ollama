#!/bin/bash

# PR Submission Helper Script
# This script helps prepare and validate the reranking fix for submission

set -e

echo "🚀 Ollama Reranking Fix - PR Submission Helper"
echo "=============================================="

# Configuration
BRANCH_NAME="reranking-implementation"
REMOTE_NAME="origin"

echo "📋 Pre-submission Checklist"
echo ""

# Check current branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "$BRANCH_NAME" ]; then
    echo "⚠️  Warning: You're on branch '$CURRENT_BRANCH', expected '$BRANCH_NAME'"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if there are uncommitted changes
if ! git diff --quiet HEAD; then
    echo "⚠️  Warning: You have uncommitted changes"
    git status --short
    read -p "Commit them now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add .
        read -p "Enter commit message: " commit_msg
        git commit -m "$commit_msg"
    fi
fi

# Run tests
echo "🧪 Running tests..."
export PATH="/opt/homebrew/bin:$PATH"

echo "  - Building project..."
if go build -o ollama-test >/dev/null 2>&1; then
    echo "    ✅ Build successful"
else
    echo "    ❌ Build failed"
    exit 1
fi

echo "  - Running unit tests..."
if cd runner/ollamarunner && go test -v > test_results.log 2>&1; then
    echo "    ✅ All tests pass"
    PASSING_TESTS=$(grep "PASS:" test_results.log | wc -l | tr -d ' ')
    echo "    📊 $PASSING_TESTS tests passed"
else
    echo "    ❌ Some tests failed"
    echo "    📄 Check runner/ollamarunner/test_results.log for details"
    exit 1
fi

cd ../..

# Check commit history
echo ""
echo "📚 Recent commits:"
git log --oneline -5

echo ""
echo "🔍 Summary of changes:"
git diff --stat HEAD~2

echo ""
echo "📊 Files modified:"
git diff --name-only HEAD~2

echo ""
echo "✅ Pre-submission checks complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Push your branch: git push $REMOTE_NAME $BRANCH_NAME"
echo "2. Go to GitHub and create a pull request"
echo "3. Use the content from PR_TEMPLATE.md as your PR description"
echo "4. Reference the original PR #11328 in your description"
echo ""
echo "📋 PR Checklist:"
echo "- ✅ Critical bug fix implemented"
echo "- ✅ Comprehensive tests added"
echo "- ✅ Documentation provided" 
echo "- ✅ Build passes"
echo "- ✅ All tests pass"
echo "- ⏳ Integration testing (requires real model)"
echo ""

read -p "Push branch to remote now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Pushing to remote..."
    git push $REMOTE_NAME $BRANCH_NAME
    echo ""
    echo "✅ Branch pushed successfully!"
    echo "🌐 Go to GitHub to create your pull request"
    echo "📄 Use PR_TEMPLATE.md for the description"
else
    echo "👍 Ready to push when you are!"
    echo "    Run: git push $REMOTE_NAME $BRANCH_NAME"
fi

echo ""
echo "🎉 Your reranking fix is ready for submission!"
