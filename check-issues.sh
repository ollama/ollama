#!/bin/bash
# GitHub Issues Report Generator for Ollama
# This script fetches open issues from the ollama/ollama repository
# and generates a formatted report
#
# Usage:
#   ./check-issues.sh [token]
#   OLLAMA_GITHUB_TOKEN=ghp_xxx ./check-issues.sh
#   Or with GSM: OLLAMA_GSM_ENABLED=true ./check-issues.sh

set -e

OWNER="ollama"
REPO="ollama"
GITHUB_API="https://api.github.com"

# Get token from argument or environment
TOKEN="${1:-$OLLAMA_GITHUB_TOKEN}"

if [ -z "$TOKEN" ]; then
    echo "❌ GitHub token not provided"
    echo ""
    echo "Usage:"
    echo "  $0 ghp_xxxxxxxxxxxxx"
    echo "  or"
    echo "  export OLLAMA_GITHUB_TOKEN=ghp_xxxxxxxxxxxxx"
    echo "  $0"
    echo ""
    echo "Get a token at: https://github.com/settings/tokens"
    exit 1
fi

echo "📊 Ollama Repository Issues Report"
echo "==================================="
echo ""
echo "⏳ Fetching open issues from $OWNER/$REPO..."
echo ""

# Fetch open issues (sorted by updated, most recent first)
RESPONSE=$(curl -s -H "Authorization: Bearer $TOKEN" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "$GITHUB_API/repos/$OWNER/$REPO/issues?state=open&sort=updated&order=desc&per_page=100")

# Check if we got valid JSON
if ! echo "$RESPONSE" | grep -q '"number"'; then
    echo "❌ Failed to fetch issues"
    echo "Response: $RESPONSE"
    exit 1
fi

# Count total issues
TOTAL=$(echo "$RESPONSE" | grep -o '"number"' | wc -l)

echo "📈 Summary Statistics"
echo "─────────────────────"
echo "Total Open Issues: $TOTAL"
echo ""

# Extract and display recent issues
echo "⏰ Most Recently Updated Issues (Top 15)"
echo "─────────────────────────────────────────"
echo ""
echo "$RESPONSE" | jq -r '.[] | "\(.number)\t\(.title)\t\(.user.login)\t\(.updated_at)"' | head -10 | \
    awk -F'\t' '{
        printf "#%-6d %-50s @%-15s %s\n", $1, substr($2, 1, 50), $3, substr($4, 1, 10)
    }'

echo ""
echo "🏷️  Issues by Label"
echo "─────────────────"

# Count issues by label
echo "$RESPONSE" | jq -r '.[] | select(.labels | length > 0) | .labels[].name' | \
    sort | uniq -c | sort -rn | head -10 | \
    awk '{printf "  %-15s %d issues\n", $2, $1}'

echo ""
echo "⚠️  Issues by Priority"
echo "────────────────────"

# Count specific labels
BUG_COUNT=$(echo "$RESPONSE" | jq '[.[] | select(.labels[].name == "bug")] | length')
FEATURE_COUNT=$(echo "$RESPONSE" | jq '[.[] | select(.labels[].name == "enhancement")] | length')
DOCS_COUNT=$(echo "$RESPONSE" | jq '[.[] | select(.labels[].name == "documentation")] | length')
HELP_WANTED=$(echo "$RESPONSE" | jq '[.[] | select(.labels[].name == "help wanted")] | length')

echo "  🐛 Bug reports: $BUG_COUNT"
echo "  ✨ Feature requests: $FEATURE_COUNT"
echo "  📚 Documentation needed: $DOCS_COUNT"
echo "  🤝 Help wanted: $HELP_WANTED"

echo ""
echo "✅ Links"
echo "─────"
echo "  View all issues: https://github.com/$OWNER/$REPO/issues"
echo "  Repository: https://github.com/$OWNER/$REPO"

echo ""
echo "✓ Report generated successfully"
