# MCP + Telegram Integration Setup Guide

This guide explains how to set up the new MCP (Model Context Protocol) and Telegram messaging integration for your Ollama agent system.

## Architecture Overview

```
                    ┌─────────────────┐
                    │   You (User)    │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
      ┌───────▼───────┐           ┌────────▼────────┐
      │   Telegram    │           │   Web UI        │
      │   (Mobile)    │           │   (Desktop)     │
      └───────┬───────┘           └────────┬────────┘
              │                             │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼────────┐
                    │ Message Gateway │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Agent System   │
                    │   (Ollama)      │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼───────┐   ┌───────▼───────┐   ┌───────▼───────┐
│  MCP Servers  │   │    Browser    │   │    Email      │
│  (Calendar,   │   │  Automation   │   │    (SMTP)     │
│   etc.)       │   │  (Puppeteer)  │   │               │
└───────────────┘   └───────────────┘   └───────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
cd agent-system
npm install
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```bash
# Minimum required for Telegram
TELEGRAM_BOT_TOKEN=your_bot_token_here
OLLAMA_API=http://localhost:11434
```

### 3. Create Telegram Bot

1. Open Telegram and search for `@BotFather`
2. Send `/newbot`
3. Follow prompts to name your bot (e.g., "MyOllamaAgent")
4. Copy the token and add to `.env`

### 4. Start the Server

```bash
# Start with Telegram enabled
npm run start:telegram

# Or standard start (Telegram enabled if token is set)
npm start
```

### 5. Test Your Bot

1. Find your bot on Telegram (search for the username you created)
2. Send `/start`
3. Try: "What can you help me with?"

## Features

### Telegram Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message |
| `/help` | Show available commands |
| `/status` | Check system status |
| `/clear` | Clear conversation history |

### Natural Language Requests

Just message your bot naturally:

- "Search for the best restaurants near Kings Cross"
- "What's the weather like today?"
- "Help me write an email to my boss about taking Friday off"

### Workflow Triggers (Coming Soon)

These phrases trigger automated workflows:

- "Book an appointment at..." → Appointment booking workflow
- "Add to my calendar..." → Calendar event creation
- "Send an email to..." → Email composition and sending

## MCP Servers

### Enabled by Default

- **Puppeteer** - Browser automation for web interactions

### Available (Configure in mcp-config.json)

- **Google Calendar** - Requires Google Cloud credentials
- **Gmail** - Requires Google Cloud credentials
- **Filesystem** - File operations in allowed directories
- **SQLite** - Database operations
- **Brave Search** - Web search (requires API key)

### Enabling MCP Servers

Edit `mcp/mcp-config.json`:

```json
{
  "servers": {
    "google-calendar": {
      "enabled": true,  // Change to true
      ...
    }
  }
}
```

## Security

### User Whitelisting

For private bots, whitelist specific Telegram users:

```bash
# In .env
ALLOWED_TELEGRAM_USERS=123456789,987654321
```

Find your Telegram user ID by messaging `@userinfobot`.

### API Key Security

All API keys are encrypted at rest using AES-256-GCM.

## API Endpoints

New endpoints added for integrations:

| Endpoint | Description |
|----------|-------------|
| `GET /api/integrations/status` | Overall integrations status |
| `GET /api/mcp/status` | MCP servers status |
| `GET /api/mcp/tools` | List available MCP tools |
| `GET /api/messaging/stats` | Messaging gateway statistics |

## Troubleshooting

### "Telegram bot not responding"

1. Check token is correct in `.env`
2. Ensure server is running (`npm start`)
3. Check logs for errors

### "MCP server failed to connect"

1. Ensure npx is available (Node.js installed)
2. Check internet connection for downloading MCP servers
3. Review logs for specific error

### "Browser automation not working"

1. Puppeteer requires Chromium - it will download automatically
2. On Mac, you may need to allow it in Security preferences
3. Check for sufficient disk space (~200MB for Chromium)

## Model Recommendations

For your M4 Mac Mini (16GB RAM):

| Model | Best For | Command |
|-------|----------|---------|
| `qwen2.5:7b` | Tool calling, complex tasks | `ollama pull qwen2.5:7b` |
| `llama3.1:8b` | General purpose | `ollama pull llama3.1:8b` |
| `mistral:7b` | Fast responses | `ollama pull mistral:7b` |

## Next Steps

1. **Set up Google Calendar** - For scheduling features
2. **Configure Email** - For sending confirmations
3. **Create Custom Workflows** - In `workflows/workflow-templates.js`
4. **Add WhatsApp** - Via Twilio (optional)

## File Structure

```
agent-system/
├── mcp/
│   ├── mcp-client.js         # MCP protocol client
│   ├── mcp-config.json       # Server configurations
│   ├── mcp-tool-adapter.js   # Adapts MCP to ToolSystem
│   ├── browser-automation.js # Puppeteer wrapper
│   └── index.js              # Main MCP integration
├── integrations/
│   ├── message-gateway.js    # Unified message handling
│   ├── telegram-bot.js       # Telegram adapter
│   └── index.js              # Integration manager
├── workflows/
│   └── workflow-templates.js # Pre-defined workflows
└── server.js                 # Main server (updated)
```
