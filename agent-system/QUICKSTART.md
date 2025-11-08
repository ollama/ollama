# Quick Start Guide

## ✅ Your Server is Running!

**Server URL:** http://localhost:3001

## How to Access

### Option 1: Simple HTTP Server for the HTML
Since the frontend needs to connect via Socket.IO, the easiest way is:

1. Open your browser to: **http://localhost:3001**

That's it! The server serves the HTML automatically.

### Option 2: VS Code Simple Browser
1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Type "Simple Browser"
3. Enter URL: `http://localhost:3001`

### Option 3: Port Forward (if on remote server)
If you're on a remote server/codespace:
1. Forward port 3001 to your local machine
2. Access via `http://localhost:3001` in your local browser

## What to Expect

When you open the page, you should see:
- ✅ "SYSTEM: CONNECTION ESTABLISHED" message
- ✅ 4 agents in the sidebar (Researcher, Coder, Critic, Planner)
- ✅ Agent names in the dropdown menu
- ✅ Retro green terminal interface

## First Steps

1. **Select an agent** - Click on one in the sidebar or dropdown
2. **Type a message** - Something like "Tell me about yourself"
3. **Press Enter or click [SEND]**
4. **Wait for response** - You'll see "PROCESSING" while the agent thinks

## Testing the System

Try these commands:

### Test Individual Agent
```
Select: RESEARCHER
Message: "What are your capabilities?"
```

### Test Learning
```
Message 1: "I'm a Python developer who loves machine learning"
Message 2: "What do you know about me?"
```
The agent should remember your first message!

### Test Auto-Discussion
1. Click `[AUTO_DISCUSS]`
2. Enter topic: "future of AI"
3. Watch agents debate

### View Your Profile
1. Click `[USER_PROFILE]`
2. See interaction count and interests

## Troubleshooting

### No agents in dropdown?
- Check browser console (F12) for errors
- Verify you're on http://localhost:3001 (not 3000)
- Check server is running: `lsof -i :3001`

### "Failed to get response from Ollama"
```bash
# Check Ollama is running
ollama list

# Pull the model if needed
ollama pull llama3.2:latest

# Test Ollama directly
ollama run llama3.2:latest "Hello"
```

### Can't connect to server
```bash
# Restart the server
cd /workspaces/ollama/agent-system
killall node
PORT=3001 node server.js
```

## Current Status

✅ Server running on port 3001
✅ 4 agents loaded and ready
✅ Ollama integration configured
✅ Memory system active
✅ Layout fixed (no text cutoff)

## Next Steps

Once it's working:
1. Have conversations with different agents
2. Watch them learn your preferences
3. Try agent-to-agent conversations
4. Customize agent personalities in `server.js`

Enjoy your AI agent system!
