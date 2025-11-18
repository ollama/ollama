# 🚀 H'uhul Multi Hive OS - Quick Start Guide

Get your H'uhul Hive running in 5 minutes!

---

## ⚡ Fastest Path

### Linux/macOS

```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Pull a model (at least one)
ollama pull qwen2.5:latest

# 3. Start the hive
cd huhul-multi-hive-os
./start_hive.sh
```

### Windows

```batch
REM 1. Install Ollama from https://ollama.ai

REM 2. Pull a model
ollama pull qwen2.5:latest

REM 3. Start the hive
cd huhul-multi-hive-os
start_hive.bat
```

---

## 🎯 First Steps

### 1. Access the Web Interface

Open `frontend/index.html` in your browser, or:

```bash
cd frontend
python -m http.server 8080
# Then open: http://localhost:8080
```

### 2. Check Hive Status

Click "🔄 Refresh Status" to see:
- Ollama connection status
- Available agents
- Knowledge base size

### 3. Chat with the Hive

Type a message in the input box:
- "What is quantum computing?"
- "Write a Python function to sort a list"
- "Analyze the benefits of microservices"

Watch as multiple agents collaborate to answer!

### 4. Upload Documents

1. Click "Choose Files"
2. Select .txt, .md, .py, or .json files
3. Click "📥 Ingest into Hive"
4. The memory agent will summarize and store them

### 5. Optimize the Hive

Click "⚡ Optimize Hive" to simulate training/optimization across your knowledge base.

---

## 🐝 Understanding Agent Responses

When you ask a question:

1. **Queen** analyzes your query
2. **Memory** retrieves relevant knowledge
3. **Specialists** (Coder/Analyst/Creative) provide expertise
4. **Queen** synthesizes the final answer

Look for the `[ACTIVATED]` message to see which agents worked on your request!

---

## 📚 Next Steps

- **Read the full README**: `README.md`
- **Try XJSON**: `config/example_tape.xjson`
- **API Examples**: Test the REST API endpoints
- **Customize Agents**: Edit `config/hive_config.json`

---

## 🐛 Common Issues

### "Ollama connection failed"
```bash
# Check if Ollama is running
ollama list

# Start Ollama
ollama serve
```

### "Model not found"
```bash
# Pull the required model
ollama pull qwen2.5:latest
```

### "Port 8000 already in use"
```bash
# Find and kill the process using port 8000
lsof -ti:8000 | xargs kill -9
# Or change the port in huhul_server.py
```

---

## 🎓 Example Queries

**Technical Questions:**
- "Explain Docker containers"
- "What are the SOLID principles?"
- "How does async/await work in Python?"

**Code Generation:**
- "Write a REST API in FastAPI"
- "Create a React component for a todo list"
- "Generate a binary search algorithm"

**Analysis:**
- "Compare SQL vs NoSQL databases"
- "What are the pros and cons of microservices?"
- "Analyze this data: [paste data]"

**Creative:**
- "Design a logo concept for a tech startup"
- "Brainstorm app ideas for education"
- "Create a name for a space exploration company"

---

## 🌟 Pro Tips

1. **Upload related documents** before asking questions - the hive will use them as context
2. **Use specific questions** for better agent specialization
3. **Check the agent cards** to see which agents are active
4. **Monitor the terminal** for detailed orchestration logs

---

**🛸 Welcome to the Hive! 🐝**
