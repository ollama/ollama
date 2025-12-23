#!/bin/bash
# K'uhul Multi Hive OS - NPM Publishing Script
# Usage: ./scripts/publish-npm.sh [version]
# Example: ./scripts/publish-npm.sh 1.0.0

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════╗"
echo "║  K'UHUL MULTI HIVE OS - NPM PUBLISHER   ║"
echo "╚══════════════════════════════════════════╝"
echo -e "${NC}"

# Check if logged in to NPM
echo -e "${BLUE}[1/7]${NC} Checking NPM authentication..."
if ! npm whoami > /dev/null 2>&1; then
    echo -e "${RED}❌ Not logged in to NPM${NC}"
    echo -e "${YELLOW}Please run:${NC} npm login"
    exit 1
fi
NPM_USER=$(npm whoami)
echo -e "${GREEN}✅ Logged in as: ${NPM_USER}${NC}"
echo ""

# Get current version
CURRENT_VERSION=$(npm pkg get version | tr -d '"')
echo -e "${BLUE}[2/7]${NC} Current version: ${YELLOW}${CURRENT_VERSION}${NC}"

# Determine new version
if [ -z "$1" ]; then
    echo ""
    echo "Version bump options:"
    echo "  patch  - ${CURRENT_VERSION} → $(npm version patch --no-git-tag-version && npm pkg get version | tr -d '"' && npm version ${CURRENT_VERSION} --no-git-tag-version > /dev/null 2>&1)"
    echo "  minor  - ${CURRENT_VERSION} → $(npm version minor --no-git-tag-version && npm pkg get version | tr -d '"' && npm version ${CURRENT_VERSION} --no-git-tag-version > /dev/null 2>&1)"
    echo "  major  - ${CURRENT_VERSION} → $(npm version major --no-git-tag-version && npm pkg get version | tr -d '"' && npm version ${CURRENT_VERSION} --no-git-tag-version > /dev/null 2>&1)"
    echo ""
    read -p "Enter version (patch/minor/major/X.Y.Z) or 'skip' to keep current: " VERSION_INPUT

    if [ "$VERSION_INPUT" = "skip" ]; then
        NEW_VERSION=$CURRENT_VERSION
    elif [ "$VERSION_INPUT" = "patch" ] || [ "$VERSION_INPUT" = "minor" ] || [ "$VERSION_INPUT" = "major" ]; then
        npm version $VERSION_INPUT --no-git-tag-version > /dev/null
        NEW_VERSION=$(npm pkg get version | tr -d '"')
    else
        npm version $VERSION_INPUT --no-git-tag-version > /dev/null
        NEW_VERSION=$(npm pkg get version | tr -d '"')
    fi
else
    if [ "$1" = "patch" ] || [ "$1" = "minor" ] || [ "$1" = "major" ]; then
        npm version $1 --no-git-tag-version > /dev/null
        NEW_VERSION=$(npm pkg get version | tr -d '"')
    else
        npm version $1 --no-git-tag-version > /dev/null
        NEW_VERSION=$(npm pkg get version | tr -d '"')
    fi
fi

echo -e "${GREEN}✅ New version: ${NEW_VERSION}${NC}"
echo ""

# Validate package.json
echo -e "${BLUE}[3/7]${NC} Validating package.json..."
if npm pkg get name version description repository > /dev/null 2>&1; then
    PACKAGE_NAME=$(npm pkg get name | tr -d '"')
    echo -e "${GREEN}✅ Package: ${PACKAGE_NAME}${NC}"
else
    echo -e "${RED}❌ Invalid package.json${NC}"
    exit 1
fi
echo ""

# Dry run
echo -e "${BLUE}[4/7]${NC} Running dry-run..."
echo ""
npm pack --dry-run 2>&1 | grep -E "^npm notice (package|tarball|unpacked)" | sed 's/npm notice /  /'
echo ""

# Show what will be published
echo -e "${BLUE}[5/7]${NC} Files to be published:"
npm pack --dry-run 2>&1 | grep -E "\.py$|\.json$|\.md$|\.sh$|\.bat$|\.html$|\.svg$" | sed 's/npm notice /  /' | head -20
echo "  ... and more"
echo ""

# Confirm
echo -e "${YELLOW}═══════════════════════════════════════${NC}"
echo -e "  Package: ${GREEN}${PACKAGE_NAME}${NC}"
echo -e "  Version: ${GREEN}${NEW_VERSION}${NC}"
echo -e "  Registry: ${BLUE}https://registry.npmjs.org${NC}"
echo -e "${YELLOW}═══════════════════════════════════════${NC}"
echo ""
read -p "Publish to NPM? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo -e "${RED}❌ Publish cancelled${NC}"
    # Revert version change
    npm version $CURRENT_VERSION --no-git-tag-version > /dev/null 2>&1
    exit 0
fi
echo ""

# Publish
echo -e "${BLUE}[6/7]${NC} Publishing to NPM..."
echo ""

if npm publish --access public; then
    echo ""
    echo -e "${GREEN}✅ Successfully published ${PACKAGE_NAME}@${NEW_VERSION}${NC}"
else
    echo ""
    echo -e "${RED}❌ Publish failed${NC}"
    # Revert version change
    npm version $CURRENT_VERSION --no-git-tag-version > /dev/null 2>&1
    exit 1
fi
echo ""

# Post-publish tasks
echo -e "${BLUE}[7/7]${NC} Post-publish tasks..."
echo ""
echo -e "${GREEN}✅ Package published successfully!${NC}"
echo ""
echo "📦 View package:"
echo "   https://www.npmjs.com/package/${PACKAGE_NAME}"
echo ""
echo "📥 Install command:"
echo "   ${YELLOW}npm install -g ${PACKAGE_NAME}${NC}"
echo ""
echo "🏷️  Next steps:"
echo "   1. Create git tag: ${YELLOW}git tag -a v${NEW_VERSION} -m 'Release v${NEW_VERSION}'${NC}"
echo "   2. Push tag: ${YELLOW}git push origin v${NEW_VERSION}${NC}"
echo "   3. Create GitHub release"
echo "   4. Update CHANGELOG.md"
echo "   5. Announce release"
echo ""
echo -e "${BLUE}═══════════════════════════════════════${NC}"
echo -e "   🎉 ${GREEN}K'uhul v${NEW_VERSION} is live!${NC} 🎉"
echo -e "${BLUE}═══════════════════════════════════════${NC}"
