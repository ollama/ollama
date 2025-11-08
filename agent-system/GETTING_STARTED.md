# Getting Started with Agent Terminal System

A multi-agent AI system powered by Ollama with learning capabilities.

## Quick Start

### 1. Start the Server

From the `agent-system` directory:

```bash
cd agent-system
npm start
```

The server will start on [http://localhost:3000](http://localhost:3000)

You should see:
```
╔═══════════════════════════════════════════╗
║   AGENT TERMINAL SYSTEM - SERVER ONLINE   ║
╚═══════════════════════════════════════════╝

Server running on: http://localhost:3000
Ollama API: http://localhost:11434
```

### 2. Access the Web Interface

Open your browser and navigate to:
```
http://localhost:3000
```

You'll see a terminal-style interface with 4 AI agents ready to help.

---

## Available Agents

### 1. Coder Agent
**Best for:** Writing code, debugging, technical solutions

**Example prompts:**
```
- Write a function to reverse a string in Python
- Debug this code: [paste code]
- How do I implement a binary search?
- Create a REST API endpoint in Express
```

### 2. Researcher Agent
**Best for:** Information gathering, explanations, learning new topics

**Example prompts:**
```
- What is TypeScript?
- Explain how async/await works
- Research the latest React features
- What are the differences between SQL and NoSQL?
```

### 3. Planner Agent
**Best for:** Breaking down tasks, creating roadmaps, organizing projects

**Example prompts:**
```
- Create a plan to learn JavaScript
- Break down building a todo app into steps
- Plan a 3-day coding sprint
- Organize my project structure
```

### 4. Critic Agent
**Best for:** Code reviews, feedback, identifying issues

**Example prompts:**
```
- Review this code: [paste code]
- What are potential issues with this approach?
- Critique my API design
- Suggest improvements for this function
```

---

## Using the Web Interface

1. **Select an Agent:** Click on any of the 4 agents in the sidebar
2. **Type Your Message:** Enter your question or request in the input box
3. **Send:** Press Enter or click the send button
4. **View Response:** The AI will respond in 10-15 seconds
5. **Continue Conversation:** Agents remember your conversation history

### Interface Features

- **Model Selection:** Each agent can use different AI models
- **Conversation History:** Past messages are saved and remembered
- **User Profile:** System learns your preferences over time
- **Web Search:** Agents can search the web for current information
- **Collaboration:** Agents can work together on complex tasks

---

## Using the API

You can also interact with agents via REST API:

### Send a Message

```bash
curl -X POST http://localhost:3000/api/message \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "coder",
    "message": "Write a hello world function"
  }'
```

### Response Format

```json
{
  "response": "Here's a simple hello world function...\n\n```python\ndef hello_world():\n    print(\"Hello, World!\")\n```"
}
```

### Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/message` | POST | Send message to an agent |
| `/agents` | GET | List all available agents |
| `/api/models` | GET | List available AI models |
| `/profile` | GET | View your user profile |
| `/api/memory` | GET | View conversation memory |
| `/health` | GET | Check server health |

---

## Agent Configuration

### Switching Models

Each agent can use different models. To specify a model:

**Via Web UI:**
- Click on the agent
- Use the model dropdown to select a different model

**Via API:**
```bash
curl -X POST http://localhost:3000/api/message \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "coder",
    "message": "Write code",
    "model": "qwen2.5:1.5b"
  }'
```

### Available Models

The system automatically uses the fastest available models:

| Model | Size | Speed | Best For |
|-------|------|-------|----------|
| `qwen2.5:0.5b` | 397 MB | ⚡⚡⚡⚡⚡ | Quick responses, simple tasks |
| `qwen2.5:1.5b` | 986 MB | ⚡⚡⚡⚡ | Balanced quality/speed |
| `deepseek-r1:1.5b` | 1.1 GB | ⚡⚡⚡⚡ | Reasoning tasks |
| `gemma2:2b` | 1.6 GB | ⚡⚡⚡ | Analysis, detailed responses |
| `llama3.2:latest` | 2.0 GB | ⚡⚡⚡ | High quality responses |
| `phi3:mini` | 2.2 GB | ⚡⚡ | Technical deep-dives |

**Default:** System tries smallest/fastest models first for best performance in Codespaces.

---

## Advanced Features

### Web Search Integration

Agents can search the web for current information. In your prompt, mention:
```
Search for the latest React features
```

The agent will automatically search and include results.

### Conversation Memory

The system remembers:
- All your conversations with each agent
- Your communication style and preferences
- Topics you're interested in
- Patterns in your questions

### Multi-Agent Collaboration

Start a collaboration session where multiple agents work together:

```bash
curl -X POST http://localhost:3000/api/collaboration/start \
  -H "Content-Type: application/json" \
  -d '{
    "template": "code-review",
    "task": "Review my authentication system"
  }'
```

---

## Stopping the Server

### From Terminal

Press `Ctrl + C` to gracefully stop the server.

### From Command Line

```bash
pkill -f "node server.js"
```

---

## Troubleshooting

### Server Won't Start - Port Already in Use

```bash
# Kill existing server
pkill -f "node server.js"

# Start again
npm start
```

### Slow Responses

The system tries smallest models first. If responses are slow:
- First response may take longer (models loading)
- Subsequent responses will be faster
- Check that Ollama is running: `ollama list`

### Demo Mode Messages

If you see "[DEMO MODE]" responses:
- Check Ollama is running: `curl http://localhost:11434/api/tags`
- Check server logs for errors
- Restart the server: `pkill -f "node server.js" && npm start`

### Clear Conversation History

```bash
rm -f data/history.json data/summaries.json
# Restart server
```

---

## Tips for Best Results

1. **Be Specific:** Clear, detailed prompts get better responses
2. **Use the Right Agent:** Choose the agent that matches your task
3. **Iterate:** Build on previous responses in the conversation
4. **Context:** Provide code snippets, error messages, or examples
5. **Follow-up:** Ask clarifying questions if needed

---

## Examples

### Example 1: Learning a New Topic

```
Agent: Researcher
Prompt: "Explain what GraphQL is and how it differs from REST"
```

### Example 2: Building a Feature

```
Agent: Planner
Prompt: "Create a step-by-step plan to add user authentication to my Express app"

Agent: Coder
Prompt: "Now write the code for step 1: setting up passport.js"

Agent: Critic
Prompt: "Review this authentication code for security issues"
```

### Example 3: Debugging

```
Agent: Coder
Prompt: "I'm getting this error: [paste error]. Here's my code: [paste code]"
```

---

## Next Steps

- Explore the [MODELS.md](MODELS.md) guide to customize which AI models to use
- Check [README.md](README.md) for detailed technical documentation
- Try the collaboration feature for complex multi-step tasks
- Customize agent personalities in `server.js`

---

**Need Help?**

- Check server logs for errors
- Verify Ollama is running: `ollama list`
- Test Ollama directly: `curl http://localhost:11434/api/tags`
- Restart the server if issues persist

**Enjoy your AI-powered agent system!** 🚀
