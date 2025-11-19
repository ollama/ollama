#!/bin/bash
# K'uhul Quick Publish - One-liner NPM publish
# Usage: ./scripts/quick-publish.sh [patch|minor|major]

set -e

VERSION_TYPE=${1:-patch}

echo "🚀 Quick Publishing K'uhul (${VERSION_TYPE})..."

# Check login
if ! npm whoami > /dev/null 2>&1; then
    echo "❌ Not logged in. Run: npm login"
    exit 1
fi

# Bump version
npm version $VERSION_TYPE --no-git-tag-version

# Get new version
NEW_VERSION=$(npm pkg get version | tr -d '"')

# Publish
npm publish --access public

echo "✅ Published @kuhul/multi-hive-os@${NEW_VERSION}"
echo "📦 https://www.npmjs.com/package/@kuhul/multi-hive-os"
echo ""
echo "Don't forget to:"
echo "  git add package.json"
echo "  git commit -m 'chore: bump version to ${NEW_VERSION}'"
echo "  git tag -a v${NEW_VERSION} -m 'Release v${NEW_VERSION}'"
echo "  git push && git push --tags"
