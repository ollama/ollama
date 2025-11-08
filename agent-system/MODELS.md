# Model Management Guide

## Auto-Pull Models on Startup

The agent system now automatically checks and pulls required models when you start the server.

## Quick Start

```bash
npm run start:with-ollama
```

This command will:
1. ✅ Start Ollama server
2. ✅ Check for required models
3. ✅ Auto-pull missing models
4. ✅ Start the Agent Terminal System

## Default Models

The following models are automatically pulled if not present:

| Model | Size | Purpose | Speed |
|-------|------|---------|-------|
| `llama3.2:latest` | 2.0 GB | General purpose, comprehensive | ⚡⚡⚡ |
| `qwen2.5:1.5b` | 986 MB | Fast coding, balanced | ⚡⚡⚡⚡ |
| `deepseek-r1:1.5b` | 1.1 GB | Reasoning, research | ⚡⚡⚡⚡ |
| `gemma2:2b` | 1.6 GB | Analysis, critique | ⚡⚡⚡ |
| `phi3:mini` | 2.2 GB | Technical tasks | ⚡⚡ |
| `qwen2.5:0.5b` | 397 MB | Fastest, simple tasks | ⚡⚡⚡⚡⚡ |

**Total size**: ~8.2 GB

## Customizing Models

### Option 1: Edit Configuration File

Edit `models.config.sh` to customize which models are auto-pulled:

```bash
nano models.config.sh
```

Example customization:

```bash
REQUIRED_MODELS=(
    "llama3.2:latest"
    "qwen2.5:1.5b"
    "mistral:latest"        # Add your favorite model
    "codellama:latest"      # Add specialized model
    # "phi3:mini"           # Comment out models you don't need
)
```

### Option 2: Skip Auto-Pull

If you want to manually manage models, use the regular start command:

```bash
npm start
```

Then pull models manually:

```bash
ollama pull llama3.2:latest
ollama pull qwen2.5:1.5b
```

## First-Time Startup

On first run, expect:
- ⏱️ ~5-15 minutes download time (depends on internet speed)
- 💾 ~8.2 GB disk space used
- 📦 6 models downloaded

**Progress will be shown:**

```
🚀 Starting Ollama server...
✓ Ollama started (PID: 12345)
⏳ Waiting for Ollama to be ready...
✓ Ollama is ready!
✓ Loaded model configuration from models.config.sh

📦 Checking and pulling required models...
════════════════════════════════════════
Checking llama3.2:latest... ✓ Already installed
Checking qwen2.5:1.5b... ⬇️  Pulling...
pulling manifest... done
pulling layers... 45% ███████░░░░░░░
```

## Subsequent Startups

After first run:
- ⚡ Models already present - instant startup
- ✓ Only missing models are downloaded
- 🚀 Server starts in ~3-5 seconds

## Troubleshooting

### Models not pulling?

Check your internet connection and Ollama:

```bash
ollama list          # See installed models
ollama pull <model>  # Manually pull a model
```

### Out of disk space?

Remove unused models:

```bash
ollama list
ollama rm <model-name>
```

Or edit `models.config.sh` to use fewer/smaller models:

```bash
REQUIRED_MODELS=(
    "qwen2.5:0.5b"    # Only use the smallest model
)
```

### Want faster startup?

Use fewer models in `models.config.sh`:

```bash
REQUIRED_MODELS=(
    "llama3.2:latest"   # Just one model
)
```

All agents will use this single model.

## Model Recommendations

### Minimal Setup (Fast, Low Memory)
```bash
REQUIRED_MODELS=(
    "qwen2.5:0.5b"      # 397 MB - fastest
)
```

### Balanced Setup (Recommended)
```bash
REQUIRED_MODELS=(
    "llama3.2:latest"   # 2.0 GB - quality
    "qwen2.5:1.5b"      # 986 MB - speed
    "qwen2.5:0.5b"      # 397 MB - backup
)
```

### Full Setup (Best Quality)
```bash
REQUIRED_MODELS=(
    "llama3.2:latest"
    "qwen2.5:1.5b"
    "deepseek-r1:1.5b"
    "gemma2:2b"
    "phi3:mini"
    "qwen2.5:0.5b"
)
```

## Advanced: Adding Custom Models

Want to add a specialized model?

1. Edit `models.config.sh`:
   ```bash
   REQUIRED_MODELS=(
       "llama3.2:latest"
       "codellama:latest"     # Add for coding
       "mistral:latest"       # Add for general
   )
   ```

2. Restart server:
   ```bash
   npm run start:with-ollama
   ```

3. Select in UI:
   - Click agent in sidebar
   - Choose new model from dropdown
   - Start chatting!

## Questions?

- **Q: Do I need all 6 models?**
  - A: No! Edit `models.config.sh` to use only what you need.

- **Q: Can I add more models?**
  - A: Yes! Add any Ollama model to `models.config.sh`.

- **Q: What if a model fails to pull?**
  - A: Server continues with available models. Check logs for details.

- **Q: How do I update models?**
  - A: Run `ollama pull <model>:latest` to update to newest version.

---

**Enjoy your multi-model agent system!** 🚀
