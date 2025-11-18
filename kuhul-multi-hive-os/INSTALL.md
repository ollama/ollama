# 📦 K'uhul Multi Hive OS - Installation Guide

Complete installation instructions for **Linux**, **macOS**, and **Windows**.

---

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Linux Installation](#linux-installation)
3. [macOS Installation](#macos-installation)
4. [Windows Installation](#windows-installation)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)

---

## 🖥️ System Requirements

### Minimum Requirements
- **CPU**: 4 cores (8+ recommended)
- **RAM**: 8GB (16GB+ recommended for multiple agents)
- **Storage**: 10GB free space (for Ollama models)
- **Python**: 3.9 or higher
- **Internet**: Required for initial model downloads

### Recommended Models
K'uhul works with various Ollama models. Here are the defaults:

| Agent | Model | Size | Purpose |
|-------|-------|------|---------|
| Queen | qwen2.5:latest | ~1.5GB | Orchestration |
| Coder | qwen2.5-coder:latest | ~1.5GB | Code generation |
| Analyst | llama3.2:latest | ~2GB | Analysis |
| Creative | mistral:latest | ~4GB | Creative tasks |
| Memory | llama3.2:latest | ~2GB | Knowledge storage |

**Total**: ~11GB of models

**Note**: You can start with just one model (e.g., `qwen2.5:latest`) and add others later.

---

## 🐧 Linux Installation

### Step 1: Install Ollama

```bash
# Download and install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Verify installation
ollama --version
```

### Step 2: Install Python and Dependencies

**Ubuntu/Debian:**
```bash
# Install Python 3.9+
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Verify
python3 --version
```

**Fedora/RHEL:**
```bash
# Install Python 3.9+
sudo dnf install python3 python3-pip

# Verify
python3 --version
```

**Arch Linux:**
```bash
# Install Python 3.9+
sudo pacman -S python python-pip

# Verify
python --version
```

### Step 3: Pull Ollama Models

```bash
# Start with the essential model (Queen agent)
ollama pull qwen2.5:latest

# Optional: Pull all models (recommended)
ollama pull qwen2.5-coder:latest
ollama pull llama3.2:latest
ollama pull mistral:latest

# Verify models are installed
ollama list
```

### Step 4: Clone and Setup K'uhul

```bash
# Clone the repository
git clone https://github.com/cannaseedus-bot/devmicro.git
cd devmicro/kuhul-multi-hive-os

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
cd backend
pip install -r requirements.txt
```

### Step 5: Start K'uhul Hive

**Option A: Using the startup script (easiest)**
```bash
cd ..
chmod +x start_hive.sh
./start_hive.sh
```

**Option B: Manual start**
```bash
# Terminal 1: Start the backend
cd backend
python kuhul_server.py

# Terminal 2: Serve the frontend
cd ../frontend
python -m http.server 8080
```

### Step 6: Access the Interface

Open your browser and navigate to:
- **Frontend**: http://localhost:8080
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

---

## 🍎 macOS Installation

### Step 1: Install Ollama

```bash
# Download and install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Or download from https://ollama.ai and install the .dmg

# Verify installation
ollama --version
```

### Step 2: Install Python (if not already installed)

**Using Homebrew (recommended):**
```bash
# Install Homebrew if needed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11

# Verify
python3 --version
```

**Or use the system Python (macOS 12+):**
```bash
python3 --version  # Should be 3.9+
```

### Step 3: Pull Ollama Models

```bash
# Start with the essential model
ollama pull qwen2.5:latest

# Optional: Pull all models
ollama pull qwen2.5-coder:latest
ollama pull llama3.2:latest
ollama pull mistral:latest

# Verify
ollama list
```

### Step 4: Clone and Setup K'uhul

```bash
# Clone the repository
git clone https://github.com/cannaseedus-bot/devmicro.git
cd devmicro/kuhul-multi-hive-os

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
cd backend
pip install -r requirements.txt
```

### Step 5: Start K'uhul Hive

```bash
cd ..
chmod +x start_hive.sh
./start_hive.sh
```

### Step 6: Access the Interface

- **Frontend**: http://localhost:8080
- **API**: http://localhost:8000

---

## 🪟 Windows Installation

### Step 1: Install Ollama

1. **Download Ollama**
   - Visit [ollama.ai](https://ollama.ai)
   - Click "Download for Windows"
   - Run the `.exe` installer

2. **Verify Installation**
   ```cmd
   ollama --version
   ```

   If not found, restart your terminal or computer.

### Step 2: Install Python

1. **Download Python**
   - Visit [python.org](https://www.python.org/downloads/)
   - Download Python 3.11+ (or 3.9+)
   - **Important**: Check "Add Python to PATH" during installation

2. **Verify Installation**
   ```cmd
   python --version
   ```

### Step 3: Pull Ollama Models

Open **Command Prompt** or **PowerShell**:

```cmd
REM Start with the essential model
ollama pull qwen2.5:latest

REM Optional: Pull all models (recommended)
ollama pull qwen2.5-coder:latest
ollama pull llama3.2:latest
ollama pull mistral:latest

REM Verify
ollama list
```

### Step 4: Clone and Setup K'uhul

**Using Git Bash (recommended):**
```bash
# Clone the repository
git clone https://github.com/cannaseedus-bot/devmicro.git
cd devmicro/kuhul-multi-hive-os

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
cd backend
pip install -r requirements.txt
```

**Or download ZIP:**
1. Go to https://github.com/cannaseedus-bot/devmicro
2. Click "Code" → "Download ZIP"
3. Extract to your desired location
4. Open terminal in `kuhul-multi-hive-os` folder

### Step 5: Start K'uhul Hive

**Option A: Using the startup script**
```cmd
cd kuhul-multi-hive-os
start_hive.bat
```

**Option B: Manual start**
```cmd
REM Terminal 1: Start backend
cd kuhul-multi-hive-os\backend
python kuhul_server.py

REM Terminal 2: Serve frontend (in new terminal)
cd kuhul-multi-hive-os\frontend
python -m http.server 8080
```

### Step 6: Access the Interface

- **Frontend**: http://localhost:8080
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

---

## ✅ Verification

### Check if Everything Works

1. **Verify Ollama is running:**
   ```bash
   curl http://localhost:11434/api/tags
   ```
   Should return a list of installed models.

2. **Check K'uhul backend:**
   ```bash
   curl http://localhost:8000/api/status
   ```
   Should return hive status JSON.

3. **Test the hive:**
   ```bash
   curl -X POST http://localhost:8000/api/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Hello K'\''uhul!"}'
   ```

4. **Open the web interface:**
   - Navigate to http://localhost:8080
   - You should see the K'uhul logo and dashboard
   - Click "🔄 Refresh Status" to verify connection

---

## 🐛 Troubleshooting

### Ollama Issues

**Problem**: `ollama: command not found`

**Solution (Linux/macOS)**:
```bash
# Check if Ollama is installed
which ollama

# If not found, reinstall
curl -fsSL https://ollama.ai/install.sh | sh

# Add to PATH if needed
export PATH=$PATH:/usr/local/bin
```

**Solution (Windows)**:
- Restart your computer
- Check if Ollama is in: `C:\Users\YourName\AppData\Local\Programs\Ollama`
- Add to PATH manually if needed

---

**Problem**: `Error: failed to get console mode`

**Solution**: This is harmless on Windows. Ollama is working fine.

---

**Problem**: Ollama service not running

**Solution**:
```bash
# Linux/macOS
ollama serve

# Windows: Ollama starts automatically
# If not, search for "Ollama" in Start Menu and run it
```

---

### Python Issues

**Problem**: `python: command not found`

**Solution (Linux/macOS)**:
```bash
# Try python3 instead
python3 --version

# Or install Python
# Ubuntu/Debian:
sudo apt install python3 python3-pip

# macOS:
brew install python@3.11
```

**Solution (Windows)**:
- Reinstall Python from python.org
- **Check** "Add Python to PATH" during installation

---

**Problem**: `pip: command not found`

**Solution**:
```bash
# Linux/macOS
python3 -m ensurepip

# Windows
python -m ensurepip
```

---

### Port Conflicts

**Problem**: `Address already in use` (port 8000 or 8080)

**Solution**:
```bash
# Linux/macOS - Find and kill process
lsof -ti:8000 | xargs kill -9
lsof -ti:8080 | xargs kill -9

# Windows - Find and kill process
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**Or change ports** in `kuhul_server.py`:
```python
# Change line at the end:
uvicorn.run(app, host="0.0.0.0", port=8001)  # Changed from 8000
```

---

### Model Issues

**Problem**: Models take forever to download

**Solution**:
- Start with smaller models: `ollama pull qwen2.5:1.5b`
- Download one model at a time
- Use a stable internet connection
- Models are cached in:
  - Linux/macOS: `~/.ollama/models`
  - Windows: `C:\Users\YourName\.ollama\models`

---

**Problem**: Out of memory when running models

**Solution**:
- Close other applications
- Use smaller models (e.g., `llama3.2:1b`)
- Run fewer agents at once (edit `config/hive_config.json`)

---

### Connection Issues

**Problem**: Frontend can't connect to backend

**Solution**:
1. Verify backend is running: `curl http://localhost:8000/api/status`
2. Check browser console for errors (F12)
3. Make sure you're using the correct URL
4. Disable browser extensions that might block requests
5. Try a different browser

---

### Permission Issues (Linux/macOS)

**Problem**: `Permission denied` when running startup script

**Solution**:
```bash
chmod +x start_hive.sh
./start_hive.sh
```

---

## 🎯 Next Steps

Once installed:
1. Read the [QUICKSTART.md](QUICKSTART.md) for usage examples
2. Check the [README.md](README.md) for full documentation
3. Explore the API at http://localhost:8000/docs
4. Upload documents to build your knowledge base
5. Experiment with multi-agent queries

---

## 📞 Get Help

- **Issues**: [GitHub Issues](https://github.com/cannaseedus-bot/devmicro/issues)
- **Discussions**: [GitHub Discussions](https://github.com/cannaseedus-bot/devmicro/discussions)
- **Ollama Docs**: [ollama.ai/docs](https://ollama.ai/docs)

---

**🛸 Welcome to K'uhul Multi Hive OS! 🐝**
