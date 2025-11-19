# 🚚 Migration Guide: Moving K'uhul to Standalone Repository

This guide helps you migrate from the devmicro subdirectory to the official K'uhul repository.

---

## 🎯 Why Migrate?

K'uhul Multi Hive OS is **much more than an Ollama app** - it's the **execution core of the entire ASX ecosystem**. The standalone repository provides:

- ✅ **Proper positioning** as the K'uhul Engine
- ✅ **Direct integration** with ASX packages
- ✅ **Independent versioning** and releases
- ✅ **Dedicated issue tracking** and discussions
- ✅ **NPM package publishing** (@kuhul/multi-hive-os)
- ✅ **Ecosystem-first** documentation

---

## 📦 For End Users

### Option 1: Fresh Install from New Repository

```bash
# Clone the official K'uhul repository
git clone https://github.com/cannaseedus-bot/KUHUL.git
cd KUHUL

# Install dependencies
pip install -r backend/requirements.txt

# Start the hive
./start_hive.sh
```

### Option 2: Update Existing Installation

If you already have K'uhul from devmicro:

```bash
# Navigate to your current installation
cd devmicro/kuhul-multi-hive-os

# Add the new remote
git remote add kuhul https://github.com/cannaseedus-bot/KUHUL.git

# Fetch from new remote
git fetch kuhul

# Switch to the official repository
git checkout -b main kuhul/main

# Update dependencies if needed
pip install -U -r backend/requirements.txt
```

### Option 3: Move Your Data

If you have custom configurations or ingested files:

```bash
# Backup your data
cp -r devmicro/kuhul-multi-hive-os/kuhul_data ~/kuhul_backup

# Clone new repository
git clone https://github.com/cannaseedus-bot/KUHUL.git
cd KUHUL

# Restore your data
cp -r ~/kuhul_backup ./kuhul_data
```

---

## 🔧 For Developers

### Migrating Custom Integrations

**Old import (devmicro):**
```python
from kuhul-multi-hive-os.integration.asx_bridge import KLHOrchestrator
```

**New import (K'uhul repo):**
```python
from integration.asx_bridge import KLHOrchestrator
# Or if installed as package:
from kuhul.integration.asx_bridge import KLHOrchestrator
```

### Migrating API Endpoints

No changes needed! The API remains the same:

```bash
# Old and new both use:
http://localhost:8000/api/status
http://localhost:8000/api/chat
http://localhost:8000/api/agents
# etc.
```

### Migrating Configuration

Your `config/hive_config.json` is compatible:

```bash
# Copy your custom config
cp devmicro/kuhul-multi-hive-os/config/hive_config.json KUHUL/config/
```

---

## 📋 Migration Checklist

### Before Migration

- [ ] Export your current data (`kuhul_data/`)
- [ ] Backup custom configurations
- [ ] Document your custom agent configurations
- [ ] Note any custom XJSON tapes you've created
- [ ] List any custom integrations

### During Migration

- [ ] Clone the new repository
- [ ] Install dependencies
- [ ] Restore data and configurations
- [ ] Test basic functionality
- [ ] Verify agents are working
- [ ] Test API endpoints

### After Migration

- [ ] Update any scripts pointing to old paths
- [ ] Update documentation references
- [ ] Notify collaborators of new repository
- [ ] Star the new repository! ⭐
- [ ] Update bookmarks and links

---

## 🆕 What's New in Standalone Repository

### Enhanced Ecosystem Integration

```python
# New: Direct ASX package integration
from kuhul.integration import ASXEcosystem

ecosystem = ASXEcosystem()
ecosystem.connect_klh_orchestrator()
ecosystem.connect_xjson_server()
ecosystem.start()
```

### NPM Package Support

```bash
# Install K'uhul as a Node.js package
npm install @kuhul/multi-hive-os

# Use in your Node.js app
const KUhul = require('@kuhul/multi-hive-os');
const hive = new Kuhul.HiveClient('http://localhost:8000');
```

### Improved Documentation

- **ECOSYSTEM.md**: Complete ASX integration guide
- **INSTALL.md**: Platform-specific installation
- **QUICKSTART.md**: 5-minute getting started
- **API.md**: Full API reference (coming soon)

---

## 🔗 Repository Comparison

| Feature | devmicro/kuhul-multi-hive-os | KUHUL (standalone) |
|---------|------------------------------|-------------------|
| **Purpose** | Ollama fork subdirectory | ASX execution core |
| **Positioning** | Add-on to Ollama | Standalone ecosystem hub |
| **NPM Package** | ❌ | ✅ @kuhul/multi-hive-os |
| **Ecosystem Docs** | Limited | Comprehensive |
| **Issue Tracking** | Shared with devmicro | Dedicated |
| **Releases** | N/A | Versioned releases |
| **CI/CD** | None | GitHub Actions |
| **ASX Integration** | Basic | First-class |

---

## 🐛 Troubleshooting Migration

### Issue: Old imports not working

**Solution:**
```bash
# Update your Python path
export PYTHONPATH=/path/to/KUHUL:$PYTHONPATH
```

### Issue: Missing data after migration

**Solution:**
```bash
# Ensure data directory exists
mkdir -p kuhul_data/ingested
mkdir -p kuhul_data/memory

# Copy from backup
cp -r ~/kuhul_backup/* kuhul_data/
```

### Issue: Configuration not loading

**Solution:**
```bash
# Verify config file location
ls -la config/hive_config.json

# Check JSON syntax
python -m json.tool config/hive_config.json
```

---

## 📞 Need Help?

- **Migration Issues**: [GitHub Issues - Migration](https://github.com/cannaseedus-bot/KUHUL/issues/new?labels=migration)
- **Questions**: [GitHub Discussions](https://github.com/cannaseedus-bot/KUHUL/discussions)
- **ASX Ecosystem**: [ASX Framework Repo](https://github.com/cannaseedus-bot/asx-language-framework)

---

## 🎉 Welcome to the Official K'uhul Repository!

The standalone K'uhul repository represents the project's evolution from an Ollama integration to the **execution core of the ASX ecosystem**. We're excited to have you on board!

### Next Steps

1. ⭐ **Star the repository**: https://github.com/cannaseedus-bot/KUHUL
2. 📖 **Read ECOSYSTEM.md**: Learn about ASX integration
3. 🚀 **Try the examples**: Explore XJSON workflows
4. 🤝 **Join discussions**: Share your use cases
5. 🛠️ **Contribute**: Help build the future of multi-agent AI

---

**🛸 K'uhul - The Heart of the ASX Ecosystem 🐝**
