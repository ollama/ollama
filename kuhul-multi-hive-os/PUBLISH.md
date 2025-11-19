# 📦 Publishing K'uhul to NPM

Complete guide for publishing K'uhul Multi Hive OS to NPM as `@kuhul/multi-hive-os`.

---

## 🔑 Prerequisites

1. **NPM Account**: Create at [npmjs.com](https://www.npmjs.com/signup)
2. **Organization**: Create `@kuhul` organization on NPM
3. **Access Token**: Generate from npmjs.com/settings
4. **Verified Email**: NPM requires email verification

---

## 📋 Pre-Publish Checklist

- [ ] Version bumped in `package.json`
- [ ] CHANGELOG.md updated
- [ ] All tests passing
- [ ] Documentation up to date
- [ ] README.md has NPM install instructions
- [ ] LICENSE file present
- [ ] .npmignore configured

---

## 🚀 Publishing Commands

### Step 1: Login to NPM

```bash
# Login to NPM (interactive)
npm login

# Or use auth token
npm config set //registry.npmjs.org/:_authToken YOUR_TOKEN_HERE
```

### Step 2: Verify Package Configuration

```bash
# Check package.json syntax
npm pkg get name version description

# Preview what will be published
npm pack --dry-run

# Check for issues
npm doctor
```

### Step 3: Create Organization (First Time Only)

```bash
# Via NPM website:
# 1. Go to https://www.npmjs.com/org/create
# 2. Create "@kuhul" organization
# 3. Set to public (free)

# Or via CLI (requires paid account for private orgs):
npm org create @kuhul
```

### Step 4: Publish Package

```bash
# Navigate to package directory
cd kuhul-multi-hive-os

# First time publish (creates package)
npm publish --access public

# For scoped packages (@kuhul/...), access must be public for free tier
```

### Step 5: Verify Publication

```bash
# Check package page
npm view @kuhul/multi-hive-os

# Get package info
npm info @kuhul/multi-hive-os

# View on web
open https://www.npmjs.com/package/@kuhul/multi-hive-os
```

---

## 🔄 Updating the Package

### Bump Version

```bash
# Patch release (1.0.0 → 1.0.1)
npm version patch

# Minor release (1.0.0 → 1.1.0)
npm version minor

# Major release (1.0.0 → 2.0.0)
npm version major

# Custom version
npm version 1.2.3
```

### Publish Update

```bash
# After version bump
git add package.json
git commit -m "chore: bump version to $(npm pkg get version)"
git push

# Publish to NPM
npm publish
```

---

## 📝 Complete Publishing Script

Save as `scripts/publish.sh`:

```bash
#!/bin/bash
# K'uhul NPM Publishing Script

set -e

echo "🚀 K'uhul NPM Publishing"
echo "========================"

# Check if logged in
if ! npm whoami > /dev/null 2>&1; then
    echo "❌ Not logged in to NPM. Run: npm login"
    exit 1
fi

# Get current version
CURRENT_VERSION=$(npm pkg get version | tr -d '"')
echo "Current version: $CURRENT_VERSION"

# Ask for new version
read -p "Enter new version (or 'skip' to keep current): " NEW_VERSION

if [ "$NEW_VERSION" != "skip" ]; then
    npm version $NEW_VERSION --no-git-tag-version
    NEW_VERSION=$(npm pkg get version | tr -d '"')
    echo "✅ Version updated to: $NEW_VERSION"
fi

# Run checks
echo ""
echo "🔍 Running pre-publish checks..."

# Check package.json
echo "  ✓ Validating package.json..."
npm pkg get name version description > /dev/null

# Dry run
echo "  ✓ Running dry run..."
npm pack --dry-run > /dev/null

# Show what will be published
echo ""
echo "📦 Package contents:"
npm pack --dry-run 2>&1 | grep -v npm

# Confirm
echo ""
read -p "Publish @kuhul/multi-hive-os@$NEW_VERSION to NPM? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "❌ Publish cancelled"
    exit 0
fi

# Publish
echo ""
echo "📤 Publishing to NPM..."
npm publish --access public

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Successfully published @kuhul/multi-hive-os@$NEW_VERSION"
    echo ""
    echo "View at: https://www.npmjs.com/package/@kuhul/multi-hive-os"
    echo ""
    echo "Install with: npm install @kuhul/multi-hive-os"
else
    echo ""
    echo "❌ Publish failed"
    exit 1
fi
```

Make it executable:
```bash
chmod +x scripts/publish.sh
```

---

## 📁 .npmignore Configuration

Create `.npmignore`:

```gitignore
# Git
.git
.gitignore
.gitattributes

# Development
*.log
*.tmp
.DS_Store
Thumbs.db

# Testing
test/
tests/
*.test.py
__pycache__/
*.pyc

# CI/CD
.github/
.gitlab-ci.yml
.travis.yml

# Documentation (optional - include if you want)
docs/
*.md
!README.md
!LICENSE

# Python virtual env
venv/
env/
.venv/

# Data directories (don't publish user data)
kuhul_data/
*.db
*.sqlite

# IDE
.vscode/
.idea/
*.swp
*.swo

# Large files
*.mp4
*.avi
*.zip
*.tar.gz
```

---

## 🏷️ Package Tags

Add tags after publishing:

```bash
# Add latest tag (default)
npm dist-tag add @kuhul/multi-hive-os@1.0.0 latest

# Add beta tag
npm dist-tag add @kuhul/multi-hive-os@1.1.0-beta.1 beta

# Add specific version tags
npm dist-tag add @kuhul/multi-hive-os@1.0.0 stable

# List all tags
npm dist-tag ls @kuhul/multi-hive-os
```

---

## 📊 Package Statistics

View package stats:

```bash
# Download stats
npm view @kuhul/multi-hive-os downloads

# All versions
npm view @kuhul/multi-hive-os versions

# Dependencies
npm view @kuhul/multi-hive-os dependencies

# Package size
npm view @kuhul/multi-hive-os dist.unpackedSize
```

---

## 🔐 Security

### Enable 2FA (Recommended)

```bash
# Enable 2FA for auth and publishing
npm profile enable-2fa auth-and-writes

# Enable 2FA for auth only
npm profile enable-2fa auth-only
```

### Audit Package

```bash
# Security audit
npm audit

# Fix vulnerabilities
npm audit fix
```

---

## 🐛 Troubleshooting

### Error: Package already exists

```bash
# This means the package name is taken
# Solutions:
# 1. Use scoped package: @kuhul/multi-hive-os ✅
# 2. Choose different name
# 3. Contact NPM support if you own the name
```

### Error: Forbidden - need auth

```bash
# Re-login
npm logout
npm login
```

### Error: No permission to publish

```bash
# Check organization membership
npm org ls @kuhul

# Add yourself as owner
npm owner add YOUR_USERNAME @kuhul/multi-hive-os
```

### Error: Package.json not found

```bash
# Make sure you're in the correct directory
cd kuhul-multi-hive-os

# Verify package.json exists
ls -la package.json
```

---

## 📈 Post-Publish Tasks

### 1. Create Git Tag

```bash
VERSION=$(npm pkg get version | tr -d '"')
git tag -a "v$VERSION" -m "Release v$VERSION"
git push origin "v$VERSION"
```

### 2. Create GitHub Release

```bash
# Using GitHub CLI
gh release create "v$VERSION" \
  --title "K'uhul Multi Hive OS v$VERSION" \
  --notes "See CHANGELOG.md for details"
```

### 3. Update Documentation

```bash
# Update badges in README.md
# Update installation instructions
# Update CHANGELOG.md
```

### 4. Announce Release

- Post on GitHub Discussions
- Update ASX Framework repo
- Tweet/social media
- Update examples/tutorials

---

## 🔄 Automated Publishing (GitHub Actions)

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to NPM

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          registry-url: 'https://registry.npmjs.org'

      - name: Install dependencies
        run: npm ci

      - name: Publish to NPM
        run: npm publish --access public
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}
```

Add `NPM_TOKEN` to GitHub Secrets:
1. Generate token at npmjs.com/settings/tokens
2. Add to GitHub: Settings → Secrets → Actions → New repository secret
3. Name: `NPM_TOKEN`
4. Value: Your token

---

## ✅ Publishing Checklist

Before each release:

- [ ] Update version in package.json
- [ ] Update CHANGELOG.md
- [ ] Run tests: `pytest`
- [ ] Update README.md if needed
- [ ] Check .npmignore excludes dev files
- [ ] Run `npm pack --dry-run` to preview
- [ ] Commit all changes
- [ ] Create git tag
- [ ] Run `npm publish --access public`
- [ ] Verify on npmjs.com
- [ ] Create GitHub release
- [ ] Announce release

---

## 📚 Resources

- **NPM Docs**: https://docs.npmjs.com/
- **Publishing Scoped Packages**: https://docs.npmjs.com/creating-and-publishing-scoped-public-packages
- **Package.json Fields**: https://docs.npmjs.com/cli/v8/configuring-npm/package-json
- **Semantic Versioning**: https://semver.org/

---

**🚀 Ready to publish K'uhul to the world!**
