#!/bin/bash
# Enable branch protection for kushin77/ollama main branch
# 
# Prerequisites:
#   - GitHub CLI installed: https://cli.github.com
#   - Authenticated: gh auth login
#   - Repository: kushin77/ollama
#
# This script uses the GitHub API via gh CLI to:
#   1. Require pull request reviews (1 approval)
#   2. Require signed commits (GPG)
#   3. Require status checks to pass
#   4. Dismiss stale reviews
#   5. Block force pushes and deletions
#   6. Enforce rules for admins

set -e

OWNER="kushin77"
REPO="ollama"
BRANCH="main"

echo "🔒 Enabling branch protection for $OWNER/$REPO:$BRANCH"
echo ""

# Test GitHub CLI and authentication
if ! gh repo view "$OWNER/$REPO" > /dev/null 2>&1; then
    echo "❌ Error: Cannot access $OWNER/$REPO"
    echo "   Make sure you're authenticated: gh auth login"
    exit 1
fi

echo "✅ Repository access verified"
echo ""

# Enable branch protection via GitHub API
# Note: gh CLI doesn't have direct branch protection commands yet,
# so we use the API endpoint directly
echo "📋 Applying branch protection rules..."

gh api \
  -X PUT \
  repos/$OWNER/$REPO/branches/$BRANCH/protection \
  -f required_status_checks='{
    "strict": true,
    "contexts": ["validate-landing-zone"]
  }' \
  -f enforce_admins=true \
  -f required_pull_request_reviews='{
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": false
  }' \
  -f restrictions=null \
  -f allow_force_pushes=false \
  -f allow_deletions=false \
  -f require_signed_commits=true \
  -f required_conversation_resolution=true

echo ""
echo "✅ Branch protection enabled!"
echo ""
echo "📌 Current rules for $OWNER/$REPO:$BRANCH:"
echo "   • Require 1 pull request review"
echo "   • Require signed commits (GPG)"
echo "   • Require status check: validate-landing-zone"
echo "   • Dismiss stale reviews on new commits"
echo "   • Block force pushes"
echo "   • Block deletions"
echo "   • Enforce rules for admins"
echo ""
echo "To verify: gh api repos/$OWNER/$REPO/branches/$BRANCH/protection"
