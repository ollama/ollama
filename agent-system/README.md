# AGENT TERMINAL SYSTEM v1.0

A multi-agent AI system powered by **Ollama** that learns from your interactions and develops understanding of you over time.

![Terminal Interface](docs/terminal-preview.png)

## Features

### 🤖 Four Specialized Agents
- **Researcher** - Deep analytical capabilities, gathers and synthesizes information
- **Coder** - Software development specialist, learns your coding style
- **Critic** - Critical thinking and improvement suggestions
- **Planner** - Organization and strategic planning

### 🧠 Learning System
- **Persistent Memory** - Agents remember all conversations across sessions
- **User Profiling** - Builds understanding of your interests, preferences, and style
- **Context Awareness** - Each agent has access to conversation history and user profile
- **Adaptive Responses** - Agents adapt their communication style based on your preferences

### 💬 Communication Features
- **Direct Messaging** - Chat with any agent individually
- **Agent-to-Agent** - Make agents discuss topics with each other
- **Auto-Discussion** - Watch agents debate and analyze topics autonomously
- **Real-time Updates** - Live thinking status and streaming responses

### 💾 Data Persistence
- Conversation history saved to `data/history.json`
- User profile saved to `data/profile.json`
- Auto-save every 5 minutes
- Data persists across server restarts

## Prerequisites

1. **Ollama** - Must be installed and running
   ```bash
   # Install Ollama (if not already installed)
   curl -fsSL https://ollama.com/install.sh | sh

   # Pull a model (using llama3.2 as default)
   ollama pull llama3.2:latest

   # Verify Ollama is running
   ollama list
   ```

2. **Node.js** - Version 16 or higher
   ```bash
   # Check Node version
   node --version
   ```

## Installation

1. Navigate to the agent-system directory:
   ```bash
   cd agent-system
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. (Optional) Configure Ollama API endpoint:
   ```bash
   # If Ollama is running on a different host/port
   export OLLAMA_API=http://localhost:11434
   ```

## Usage

### Starting the Server

```bash
npm start
```

You should see:
```
╔═══════════════════════════════════════════╗
║   AGENT TERMINAL SYSTEM - SERVER ONLINE   ║
╚═══════════════════════════════════════════╝

Server running on: http://localhost:3000
Ollama API: http://localhost:11434

Agents loaded: 4
User interactions: 0

Ready to learn and assist!
```

### Accessing the Interface

Open your browser and navigate to:
```
http://localhost:3000
```

### Using the Terminal

1. **Select an Agent** - Click on an agent in the sidebar or use the dropdown
2. **Type Your Message** - Enter your question or command
3. **Send** - Press Enter or click [SEND]

### Advanced Features

**Auto-Discussion**
- Click `[AUTO_DISCUSS]`
- Enter a topic
- Watch agents discuss it autonomously

**Agent-to-Agent Communication**
- Click `[AGENT_COM]`
- Select source and target agents
- Enter a message for the conversation

**View Profile**
- Click `[USER_PROFILE]`
- See your interaction statistics and what agents have learned about you

**Clear Terminal**
- Click `[CLEAR_TERM]`
- Clears display and conversation history

## How the Learning System Works

### Conversation History
Each agent maintains its own conversation history (last 20 exchanges). This allows agents to:
- Reference previous discussions
- Maintain context across multiple messages
- Build continuity in conversations

### User Profile
The system builds a profile that includes:
- **Total Interactions** - Count of all messages sent
- **Interests** - Extracted from your messages over time
- **Learnings** - Key insights agents have noted about you
- **Timestamps** - First and last interaction times

### Agent Adaptation
Agents use your profile to:
- Personalize their responses
- Remember your preferences
- Adapt communication style
- Provide increasingly relevant assistance

### Example Learning Path
1. **First Session** - Agents learn basic information
   ```
   User: "I'm a Python developer working on machine learning projects"
   ```

2. **Profile Updates**
   - Interests: python, developer, machine, learning, projects
   - Learnings: "User is a Python developer focused on ML"

3. **Next Session** - Agents reference this knowledge
   ```
   Agent: "Given your focus on machine learning, here's a Python implementation..."
   ```

## Configuration

### Changing the Model
Edit [server.js:17-57](server.js#L17-L57) and modify the `model` field for each agent:
```javascript
researcher: {
    name: 'Researcher',
    model: 'llama3.2:latest',  // Change this to any Ollama model
    // ...
}
```

### Adjusting Agent Personalities
Modify the `systemPrompt` field in [server.js:17-57](server.js#L17-L57) to change how agents behave:
```javascript
systemPrompt: `You are a Research Agent with deep analytical capabilities...`
```

### Temperature Settings
Control agent creativity by adjusting `temperature` (0.0 = deterministic, 1.0 = creative):
```javascript
temperature: 0.7  // 0.0 to 1.0
```

### Port Configuration
Change the server port:
```bash
PORT=8080 npm start
```

## File Structure

```
agent-system/
├── server.js           # Main server and agent logic
├── package.json        # Dependencies
├── public/
│   └── index.html      # Frontend interface
├── data/               # Created automatically
│   ├── history.json    # Conversation history
│   └── profile.json    # User profile
└── README.md           # This file
```

## Troubleshooting

### "Failed to get response from Ollama"
- Ensure Ollama is running: `ollama list`
- Check Ollama API endpoint: `curl http://localhost:11434/api/tags`
- Verify model is pulled: `ollama pull llama3.2:latest`

### "Socket.IO Library Failed to Load"
- Check internet connection (CDN required)
- Or download Socket.IO locally and update the script src in index.html

### Port Already in Use
```bash
# Use a different port
PORT=8080 npm start
```

### Agent Responses are Slow
- Use a smaller/faster model: `ollama pull llama3.2:1b`
- Reduce temperature in agent configuration
- Lower the conversation history limit in `buildContext()`

## Advanced Customization

### Adding New Agents
Edit [server.js:17-57](server.js#L17-L57) and add a new agent:
```javascript
myagent: {
    name: 'My Agent',
    model: 'llama3.2:latest',
    systemPrompt: `You are...`,
    temperature: 0.7
}
```

Update the frontend icons in [public/index.html:326](public/index.html#L326):
```javascript
const agentIcons = {
    myagent: '🎯',
    // ...
};
```

### Enhancing Learning
Modify `updateUserProfile()` in [server.js:162-174](server.js#L162-L174) to:
- Add sentiment analysis
- Track preferred topics
- Monitor interaction patterns
- Implement more sophisticated NLP

### Database Integration
Replace JSON file storage with a database:
- MongoDB for document storage
- PostgreSQL for relational data
- Redis for caching

## Development

```bash
# Install dev dependencies
npm install

# Run with auto-reload
npm run dev
```

## Performance Optimization

### Memory Management
- Conversation history limited to last 20 exchanges per agent
- User interests capped at 50 keywords
- Auto-save runs every 5 minutes (adjust in server.js)

### Recommended Models
- **Fast**: `llama3.2:1b` - Quick responses, lower quality
- **Balanced**: `llama3.2:latest` (3B) - Good speed/quality ratio
- **Quality**: `llama3.1:8b` - Better responses, slower

### Scaling
For production use:
- Use Redis for session storage
- Add rate limiting
- Implement user authentication
- Deploy behind nginx/proxy
- Use PM2 for process management

## Privacy & Security

- **All data stays local** - No external API calls except to your Ollama instance
- Conversation history stored in local JSON files
- No telemetry or tracking
- Full control over your data

## Contributing

Feel free to:
- Add new agent types
- Improve learning algorithms
- Enhance the UI
- Add more features

## License

MIT License - Feel free to use and modify as needed.

## Support

If you encounter issues:
1. Check Ollama is running: `ollama list`
2. Verify Node.js version: `node --version`
3. Review server logs for errors
4. Check browser console for frontend errors

## Future Enhancements

Potential improvements:
- [ ] Voice interaction
- [ ] File upload/analysis
- [ ] Image generation integration
- [ ] Multi-user support
- [ ] Agent memory sharing
- [ ] Custom agent creation UI
- [ ] Export conversation transcripts
- [ ] Mobile responsive design
- [ ] Dark/light theme toggle
- [ ] Agent performance metrics

---

**Built with Ollama** - Local AI that respects your privacy.
