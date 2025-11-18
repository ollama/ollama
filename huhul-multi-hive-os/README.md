# 🛸 H'UHUL MULTI HIVE OS

![Status](https://img.shields.io/badge/Status-Active-00cc88?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Ollama-Multi--Agent-0099ff?style=for-the-badge)
![ASX](https://img.shields.io/badge/ASX-Integration-9966ff?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9+-ffaa00?style=for-the-badge)

**An advanced multi-agent AI operating system powered by Ollama, integrated with the ASX Language Framework (XJSON, KLH, SCXQ2, Tape Runtime).**

---

## 🌟 Overview

H'uhul Multi Hive OS is a sophisticated multi-agent orchestration system that leverages Ollama's local LLM capabilities to create a "hive mind" of specialized AI agents working in harmony. Inspired by and integrated with the [ASX Language Framework](https://github.com/cannaseedus-bot/asx-language-framework), it brings together:

- **🐝 Multi-Agent Hive Architecture**: Queen + specialized worker agents
- **⚡ Ollama Integration**: Local, private, powerful LLM execution
- **🧬 XJSON Runtime**: Executable JSON for agent workflows
- **🎯 KLH Orchestration**: Multi-hive coordination patterns
- **🔐 SCXQ2 Compression**: Symbolic compression and cipher layer
- **📼 Tape Runtime**: Modular execution containers
- **🌐 Quantum Torrent**: Distributed data sharding with integrity verification

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    H'UHUL MULTI HIVE OS                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │    QUEEN     │  │    CODER     │  │   ANALYST    │      │
│  │ Orchestrator │  │   Specialist │  │  Specialist  │      │
│  │ qwen2.5:3b   │  │ qwen2.5-coder│  │ llama3.2:3b  │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                  │              │
│         └─────────────────┴──────────────────┘              │
│                           │                                 │
│                  ┌────────▼────────┐                        │
│                  │  HIVE CORE API  │                        │
│                  │   FastAPI       │                        │
│                  └────────┬────────┘                        │
│                           │                                 │
│         ┌─────────────────┼─────────────────┐              │
│         │                 │                 │              │
│   ┌─────▼─────┐   ┌──────▼──────┐   ┌──────▼──────┐       │
│   │  MEMORY   │   │  CREATIVE   │   │   OLLAMA    │       │
│   │ Knowledge │   │  Specialist │   │   Backend   │       │
│   │llama3.2:3b│   │ mistral:7b  │   │ :11434      │       │
│   └───────────┘   └─────────────┘   └─────────────┘       │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                    ASX INTEGRATION LAYER                     │
├─────────────────────────────────────────────────────────────┤
│  XJSON Engine  │  KLH Orchestrator  │  Quantum Torrent     │
│  SCXQ2 Cipher  │  Tape Runtime      │  Shard Manager       │
└─────────────────────────────────────────────────────────────┘
```

### Agent Roles

| Agent    | Model              | Role                    | Temperature | Specialty                  |
|----------|--------------------|-------------------------|-------------|----------------------------|
| **Queen**    | qwen2.5:latest     | Orchestrator            | 0.7         | Task coordination & synthesis |
| **Coder**    | qwen2.5-coder:latest | Code specialist       | 0.3         | Code generation & analysis |
| **Analyst**  | llama3.2:latest    | Data analyst            | 0.5         | Pattern recognition & insights |
| **Creative** | mistral:latest     | Creative specialist     | 0.9         | Innovation & ideation      |
| **Memory**   | llama3.2:latest    | Knowledge keeper        | 0.2         | Information storage & retrieval |

---

## 🚀 Quick Start

### Prerequisites

1. **Install Ollama**

   **Linux/macOS:**
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

   **Windows:**
   - Download the installer from [ollama.ai](https://ollama.ai)
   - Run the .exe installer
   - Ollama will start automatically

2. **Pull Required Models**
   ```bash
   ollama pull qwen2.5:latest
   ollama pull qwen2.5-coder:latest
   ollama pull llama3.2:latest
   ollama pull mistral:latest
   ```

3. **Python 3.9+**
   ```bash
   python --version  # Should be 3.9 or higher
   ```

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/cannaseedus-bot/devmicro.git
   cd devmicro/huhul-multi-hive-os
   ```

2. **Install Python Dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Start the H'uhul Hive**
   ```bash
   python huhul_server.py
   ```

   The server will start on `http://localhost:8000`

4. **Open the Frontend**
   ```bash
   # In a new terminal
   cd ../frontend
   # Open index.html in your browser
   # Or use a simple HTTP server:
   python -m http.server 8080
   ```

   Navigate to `http://localhost:8080`

---

## 📖 Usage

### Web Interface

The H'uhul Multi Hive OS provides a beautiful cyberpunk-themed web interface:

1. **Status Dashboard**: Monitor hive health and agent activity
2. **Agent Panel**: View all available agents and their specializations
3. **Communication Terminal**: Chat with the hive (multi-agent orchestration)
4. **Knowledge Management**: Upload files and trigger hive optimization

### API Endpoints

#### Get Hive Status
```bash
curl http://localhost:8000/api/status
```

Response:
```json
{
  "status": "online",
  "ollama_connected": true,
  "agents_available": ["queen", "coder", "analyst", "creative", "memory"],
  "agents_active": ["queen", "memory"],
  "files_ingested": 5,
  "knowledge_base_size": 5
}
```

#### Chat with the Hive
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain quantum computing"}'
```

Response:
```json
{
  "response": "Quantum computing harnesses quantum mechanics principles...",
  "agents_used": ["queen", "memory", "analyst"],
  "orchestration": {
    "analysis": "Query requires technical analysis and knowledge retrieval",
    "specialists": 2
  }
}
```

#### Ingest Files
```bash
curl -X POST http://localhost:8000/api/ingest \
  -F "files=@document.txt" \
  -F "files=@code.py"
```

#### List Agents
```bash
curl http://localhost:8000/api/agents
```

#### Get Knowledge Base
```bash
curl http://localhost:8000/api/knowledge
```

---

## 🧬 ASX Integration

H'uhul Multi Hive OS seamlessly integrates with the ASX Language Framework:

### XJSON Execution

Execute XJSON (Executable JSON) workflows:

```python
from integration.asx_bridge import run_xjson_async

xjson_code = {
    "@hive.query": {
        "agent": "coder",
        "message": "Write a Python function to calculate fibonacci",
        "capture": "ctx.code_result"
    }
}

result = await run_xjson_async(xjson_code, hive_client)
```

### XJSON Tape Example

See `config/example_tape.xjson` for a complete multi-agent analysis tape.

### KLH Orchestration

Distribute tasks across multiple hive nodes:

```python
from integration.asx_bridge import KLHOrchestrator

orchestrator = KLHOrchestrator(hive_nodes=[
    "http://localhost:8000",
    "http://localhost:8001"
])

results = await orchestrator.distribute_task({
    "message": "Analyze this dataset",
    "data": large_dataset
})
```

### Quantum Torrent Sharding

Shard data for distributed processing:

```python
from backend.quantum_torrent import QuantumTorrentManager

manager = QuantumTorrentManager()

# Create shards
shards = manager.create_data_shard(
    training_data,
    shard_size=1000,
    category="training"
)

# Validate integrity
validation = manager.validate_data_integrity()
print(f"Integrity: {validation['integrity_valid']}")
```

---

## 🎯 Features

### ✅ Implemented

- [x] **Multi-Agent Orchestration**: Queen-led coordination with specialist agents
- [x] **Ollama Integration**: Full local LLM support
- [x] **File Ingestion**: Upload and process documents into hive memory
- [x] **Knowledge Base**: Persistent storage with summarization
- [x] **Web Interface**: Beautiful cyberpunk-themed UI
- [x] **RESTful API**: Complete HTTP API for all operations
- [x] **XJSON Engine**: Execute XJSON workflows
- [x] **Quantum Torrent**: Distributed data sharding with SHA3-512 verification
- [x] **Agent Status Tracking**: Real-time agent activity monitoring

### 🚧 Planned

- [ ] **Multi-Hive Clustering**: Connect multiple H'uhul instances
- [ ] **SCXQ2 Compression**: Full symbolic compression implementation
- [ ] **Tape Runtime**: Production-ready tape execution system
- [ ] **RAG Integration**: Vector database for advanced retrieval
- [ ] **Model Fine-tuning**: Custom model training from ingested data
- [ ] **P2P Synchronization**: Torrent-style hive synchronization
- [ ] **WebSocket Streaming**: Real-time streaming responses
- [ ] **Agent Specialization Learning**: Dynamic agent improvement

---

## 📁 Project Structure

```
huhul-multi-hive-os/
├── backend/
│   ├── huhul_server.py        # Main FastAPI server
│   ├── quantum_torrent.py     # Distributed sharding system
│   └── requirements.txt       # Python dependencies
├── frontend/
│   └── index.html             # Web interface
├── integration/
│   └── asx_bridge.py          # ASX Framework integration (XJSON, KLH)
├── config/
│   ├── hive_config.json       # Hive configuration
│   └── example_tape.xjson     # Example XJSON tape
├── agents/                    # Agent-specific configurations
├── memory/                    # Knowledge base storage
├── docs/                      # Documentation
└── README.md                  # This file
```

---

## 🔧 Configuration

Edit `config/hive_config.json` to customize:

- Agent models and parameters
- Ollama host URL
- Storage paths
- Multi-hive settings
- Feature flags

Example:
```json
{
  "hive_id": "huhul-primary",
  "ollama_host": "http://localhost:11434",
  "api_port": 8000,
  "agents": {
    "queen": {
      "model": "qwen2.5:latest",
      "temperature": 0.7
    }
  }
}
```

---

## 🎨 Screenshots

### Web Interface
The H'uhul Multi Hive OS features a stunning cyberpunk-themed interface with:
- Real-time status monitoring
- Interactive agent cards
- Terminal-style chat interface
- Progress visualization
- File upload and management

---

## 🤝 Integration with ASX Framework

H'uhul Multi Hive OS is designed to work seamlessly with:

- **[@asx/xjson-runtime-js](https://github.com/cannaseedus-bot/asx-language-framework)**: XJSON execution
- **[@asx/klh-orchestrator](https://github.com/cannaseedus-bot/asx-language-framework)**: Hive orchestration
- **[@asx/scxq2-engine](https://github.com/cannaseedus-bot/asx-language-framework)**: Symbolic compression
- **[@asx/tape-runtime](https://github.com/cannaseedus-bot/asx-language-framework)**: Tape execution

---

## 📚 Documentation

- [API Documentation](docs/API.md) - Complete API reference
- [Agent Guide](docs/AGENTS.md) - Understanding hive agents
- [XJSON Tutorial](docs/XJSON.md) - Writing XJSON workflows
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment

---

## 🐛 Troubleshooting

### Ollama Connection Failed
```bash
# Verify Ollama is running
ollama list

# Start Ollama service
ollama serve
```

### Agent Query Timeout
- Ensure models are pulled: `ollama pull qwen2.5:latest`
- Check system resources (RAM, GPU)
- Try smaller models: `ollama pull qwen2.5:3b`

### Frontend Can't Connect
- Verify backend is running on port 8000
- Check CORS settings if running on different domain
- Open browser console for detailed error messages

---

## 🔒 Security

- **Local First**: All processing happens on your machine
- **No Cloud Dependencies**: Works completely offline
- **Data Privacy**: Your data never leaves your system
- **Quantum-Resistant Hashing**: SHA3-512 for data integrity

---

## 🌟 Credits

- **ASX Language Framework**: [github.com/cannaseedus-bot/asx-language-framework](https://github.com/cannaseedus-bot/asx-language-framework)
- **Ollama**: [ollama.ai](https://ollama.ai)
- **FastAPI**: [fastapi.tiangolo.com](https://fastapi.tiangolo.com)

---

## 📄 License

See [LICENSE](../LICENSE) file for details.

---

## 🚀 What's Next?

1. **Try the Examples**: Start with simple queries and explore multi-agent responses
2. **Upload Documents**: Build your knowledge base
3. **Create XJSON Tapes**: Automate complex workflows
4. **Join the Hive**: Contribute to the project

---

## 💬 Community

- **Issues**: [GitHub Issues](https://github.com/cannaseedus-bot/devmicro/issues)
- **Discussions**: [GitHub Discussions](https://github.com/cannaseedus-bot/devmicro/discussions)

---

<div align="center">

**🛸 Welcome to the H'uhul Hive 🐝**

*Where Multiple Agents Work as One*

</div>
