# 🛠️ K'uhul Scripts

Developer scripts for K'uhul Multi Hive OS maintenance and publishing.

---

## 📜 Available Scripts

### `publish-npm.sh`

**Full-featured NPM publishing script with interactive prompts and validation.**

```bash
# Interactive mode (prompts for version)
./scripts/publish-npm.sh

# Specify version type
./scripts/publish-npm.sh patch   # 1.0.0 → 1.0.1
./scripts/publish-npm.sh minor   # 1.0.0 → 1.1.0
./scripts/publish-npm.sh major   # 1.0.0 → 2.0.0

# Specify exact version
./scripts/publish-npm.sh 1.2.3
```

**Features:**
- ✅ NPM login verification
- ✅ Version bump preview
- ✅ Package validation
- ✅ Dry-run preview
- ✅ File listing
- ✅ Confirmation prompt
- ✅ Post-publish instructions

---

### `quick-publish.sh`

**One-liner NPM publish for quick releases.**

```bash
# Patch release (default)
./scripts/quick-publish.sh

# Minor release
./scripts/quick-publish.sh minor

# Major release
./scripts/quick-publish.sh major
```

**Use when:**
- You're already logged in
- You know what version you want
- You want speed over safety

---

## 🚀 Publishing Workflow

### First Time Setup

```bash
# 1. Login to NPM
npm login

# 2. Verify organization access
npm org ls @kuhul

# 3. Test with dry-run
npm pack --dry-run
```

### Regular Publishing

```bash
# 1. Update CHANGELOG.md
vim CHANGELOG.md

# 2. Commit changes
git add .
git commit -m "chore: prepare release"

# 3. Run publish script
./scripts/publish-npm.sh patch

# 4. Create git tag (script will remind you)
git tag -a v1.0.1 -m "Release v1.0.1"
git push origin v1.0.1

# 5. Create GitHub release
gh release create v1.0.1 --notes "See CHANGELOG.md"
```

---

## 📦 Manual Publishing

If you prefer to do it manually:

```bash
# 1. Bump version
npm version patch  # or minor, or major

# 2. Publish
npm publish --access public

# 3. Tag and push
VERSION=$(npm pkg get version | tr -d '"')
git tag -a "v$VERSION" -m "Release v$VERSION"
git push origin "v$VERSION"
```

---

## 🔧 Development Scripts

### Install Dependencies

```bash
# Python dependencies
pip install -r backend/requirements.txt

# If using NPM package
npm install
```

### Start Development Server

```bash
# Full startup script
../start_hive.sh

# Or manual
cd backend
python kuhul_server.py
```

---

## 🐛 Troubleshooting

### Error: Not logged in to NPM

```bash
npm login
# Enter credentials when prompted
```

### Error: Package already exists

```bash
# Make sure you're publishing to @kuhul/multi-hive-os
# NOT multi-hive-os (without scope)
npm pkg get name
# Should output: "@kuhul/multi-hive-os"
```

### Error: 403 Forbidden

```bash
# Check organization membership
npm org ls @kuhul

# Or add yourself
npm owner add YOUR_USERNAME @kuhul/multi-hive-os
```

### Error: Version already exists

```bash
# You can't republish the same version
# Bump to a new version
npm version patch
```

---

## 📊 Post-Publish Checklist

After publishing to NPM:

- [ ] Verify package on npmjs.com
- [ ] Test installation: `npm install -g @kuhul/multi-hive-os`
- [ ] Create git tag
- [ ] Push tags to GitHub
- [ ] Create GitHub release
- [ ] Update CHANGELOG.md
- [ ] Announce on social media/forums
- [ ] Update documentation if needed

---

## 🔗 Useful Commands

```bash
# Check what will be published
npm pack --dry-run

# View package info
npm view @kuhul/multi-hive-os

# Check latest version
npm view @kuhul/multi-hive-os version

# List all versions
npm view @kuhul/multi-hive-os versions

# Check download stats
npm view @kuhul/multi-hive-os downloads

# Deprecate a version
npm deprecate @kuhul/multi-hive-os@1.0.0 "Upgrade to 1.0.1"
```

---

## 📚 References

- [NPM Publishing Guide](https://docs.npmjs.com/packages-and-modules/contributing-packages-to-the-registry)
- [Semantic Versioning](https://semver.org/)
- [K'uhul PUBLISH.md](../PUBLISH.md)
