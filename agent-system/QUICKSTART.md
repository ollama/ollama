# Quick Start Guide

## What You Have

All code is complete and tested (22/22 tests pass). You just need to add credentials.

## Setup Steps (5 minutes)

### 1. Install Dependencies
```bash
cd agent-system
npm install
```

### 2. Pull Models
```bash
# Embedding model (required for RAG)
ollama pull nomic-embed-text

# Chat model (pick one)
ollama pull qwen2.5:7b      # Best for 16GB RAM
```

### 3. Create .env File
```bash
cd agent-system
cat > .env << 'ENVEOF'
# Ollama
OLLAMA_API=http://localhost:11434

# Telegram Bot (optional - get from @BotFather)
TELEGRAM_BOT_TOKEN=your_bot_token_here
ENABLE_TELEGRAM=true

# Server
PORT=3000
ENVEOF
```

### 4. Start Server
```bash
npm start
```

### 5. Open Dashboard
```
http://localhost:3000/dashboard
```

## Commands You Can Use

```
/help              - Show all commands
/status            - System status
/facts             - What I know about you
/remember <fact>   - Remember something
/search <query>    - Search knowledge base
/profile           - View/edit profile
/newcmd <name> <expansion> - Create custom command
```

## Train Your Agent

```bash
# Via CLI
npm run cli
agent> add-fact My dentist is Dr. Smith at 555-1234
agent> index ~/Documents/notes

# Via Telegram
/remember I prefer morning meetings

# Via API
curl -X POST http://localhost:3000/api/knowledge/facts \
  -H "Content-Type: application/json" \
  -d '{"fact": "I like sushi", "category": "food"}'
```

## Test It

```bash
# Run tests
npm test

# Use CLI
npm run cli

# Message Telegram bot
/help
```

Everything is local and free. Your data stays on your Mac Mini.
