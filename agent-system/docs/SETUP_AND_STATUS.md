# Agent System - Setup & Status

## What's Built

### Core Components
| Component | Status | Description |
|-----------|--------|-------------|
| SmartAgent | Complete | Intent analysis, planning, execution with confirmations |
| RAG Knowledge Base | Complete | Vector embeddings, SQLite persistence, semantic search |
| Telegram Bot | Complete | Message handling, commands, typing indicators |
| MCP Integration | Complete | Browser automation, tool adapters |
| Monitoring Dashboard | Complete | Health checks, metrics, activity log |

### API Endpoints
```
# Monitoring (NEW)
GET  /dashboard                    - Visual dashboard (auto-refresh)
GET  /api/monitor/status           - Full system status JSON
GET  /api/monitor/health           - Health check for load balancers
GET  /api/monitor/metrics          - Metrics summary
GET  /api/monitor/activity         - Recent activity log

# Knowledge Base
GET  /api/knowledge/stats          - Knowledge base statistics
GET  /api/knowledge/facts          - All learned facts
POST /api/knowledge/index/file     - Index a single file
POST /api/knowledge/index/directory - Index directory of files
POST /api/knowledge/index/emails   - Index exported emails
POST /api/knowledge/index/text     - Index raw text
POST /api/knowledge/facts          - Add a fact manually
GET  /api/knowledge/search         - Search documents/facts
GET  /api/knowledge/profile        - Get user profile
POST /api/knowledge/profile        - Update user profile

# MCP & Messaging
GET  /api/mcp/status               - MCP status
GET  /api/mcp/tools                - Available tools
GET  /api/messaging/stats          - Messaging statistics
GET  /api/integrations/status      - All integrations status
```

---

## Quick Start

### 1. Install Dependencies
```bash
cd agent-system
npm install
```

### 2. Pull Required Models
```bash
# Embedding model (required for RAG)
ollama pull nomic-embed-text

# Chat model (pick one)
ollama pull qwen2.5:7b      # Best for 16GB RAM
# OR
ollama pull mistral:7b
# OR
ollama pull llama3.1:8b
```

### 3. Configure Environment
```bash
cp .env.example .env
```

Edit `.env`:
```env
# Required
OLLAMA_API=http://localhost:11434

# Optional - Telegram Bot
TELEGRAM_BOT_TOKEN=your_token_from_botfather
ENABLE_TELEGRAM=true
```

### 4. Start the Server
```bash
npm start
```

### 5. Open Monitoring Dashboard
```
http://localhost:3000/dashboard
```

### 6. Index Your Personal Data
```bash
# Index a folder
curl -X POST http://localhost:3000/api/knowledge/index/directory \
  -H "Content-Type: application/json" \
  -d '{"dirPath": "/path/to/your/notes"}'

# Add facts about yourself
curl -X POST http://localhost:3000/api/knowledge/facts \
  -H "Content-Type: application/json" \
  -d '{"fact": "My name is John", "category": "identity"}'

curl -X POST http://localhost:3000/api/knowledge/facts \
  -H "Content-Type: application/json" \
  -d '{"fact": "I prefer morning meetings", "category": "preferences"}'
```

---

## What's Missing (TODO)

### High Priority
| Item | Effort | Impact |
|------|--------|--------|
| **Test end-to-end** | 1 hour | Critical - verify it all works |
| **Pull embedding model** | 5 min | Required for RAG to work |
| **Set up .env** | 5 min | Required for Telegram |

### Medium Priority
| Item | Effort | Impact |
|------|--------|--------|
| Live email via IMAP/OAuth | 4-8 hours | Auto-index new emails |
| Calendar OAuth | 4 hours | Real calendar integration |
| Learn from confirmations | 2 hours | Skip redundant confirmations |
| Conversation persistence | 2 hours | Remember conversations across restarts |

### Low Priority (Nice to Have)
| Item | Effort | Impact |
|------|--------|--------|
| WhatsApp integration | 8+ hours | Need Twilio/WhatsApp Business |
| Voice messages | 4 hours | Whisper transcription |
| Web UI chat interface | 8 hours | Alternative to Telegram |
| Rate limiting | 2 hours | Prevent abuse |
| Audit logging | 2 hours | Track all actions |

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER                                     │
│                    (Telegram / API)                              │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MESSAGE GATEWAY                               │
│              (Routes messages to SmartAgent)                     │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SMART AGENT                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Analyze   │─▶│    Plan     │─▶│  Confirm (if needed)    │  │
│  │   Intent    │  │   Steps     │  │  Then Execute           │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└───────────┬─────────────────────────────────────┬───────────────┘
            │                                     │
            ▼                                     ▼
┌───────────────────────┐             ┌───────────────────────────┐
│   KNOWLEDGE BASE      │             │        TOOLS              │
│  ┌─────────────────┐  │             │  ┌─────────────────────┐  │
│  │ SQLite DB       │  │             │  │ Browser Automation  │  │
│  │ - Documents     │  │             │  │ Web Search          │  │
│  │ - Facts         │  │             │  │ Email (SMTP)        │  │
│  │ - Embeddings    │  │             │  │ MCP Servers         │  │
│  └─────────────────┘  │             │  └─────────────────────┘  │
│  Vector Search        │             │                           │
│  (nomic-embed-text)   │             │                           │
└───────────────────────┘             └───────────────────────────┘
            │                                     │
            └──────────────┬──────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       OLLAMA                                     │
│              (Local LLM - qwen2.5:7b)                           │
│                   Runs on Mac Mini                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Monitoring

The dashboard at `/dashboard` shows:
- **System Status** - healthy/degraded/partial
- **Uptime** - how long running
- **Messages processed** - count and errors
- **Knowledge base stats** - docs, facts, embeddings
- **Memory usage** - heap and system
- **Component health** - Ollama, KB, MCP, Telegram
- **Recent activity** - last 20 events

Auto-refreshes every 10 seconds.

---

## Troubleshooting

### "Embedding model not found"
```bash
ollama pull nomic-embed-text
```

### "Knowledge base not initialized"
Check that the data directory exists:
```bash
mkdir -p agent-system/data/knowledge
```

### Telegram bot not responding
1. Check TELEGRAM_BOT_TOKEN in .env
2. Check ENABLE_TELEGRAM=true
3. Restart the server

### "Step timed out"
Some operations take longer. Check:
- Ollama is running
- Model is loaded
- Network connectivity (for web search)

---

## Data Locations

```
agent-system/
├── data/
│   └── knowledge/
│       ├── knowledge.db      # SQLite database
│       ├── profile.json      # User profile
│       ├── documents.json    # Fallback if no SQLite
│       ├── facts.json        # Fallback if no SQLite
│       └── embeddings.json   # Fallback if no SQLite
└── .env                      # Configuration
```

All your personal data stays local on your Mac Mini.
