const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const path = require('path');
const fs = require('fs').promises;
const axios = require('axios');
const { search } = require('duck-duck-scrape');

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
    cors: {
        origin: "*",
        methods: ["GET", "POST"]
    },
    transports: ['websocket', 'polling'],
    allowEIO3: true,
    pingTimeout: 60000,
    pingInterval: 25000
});

const PORT = process.env.PORT || 3000;
const OLLAMA_API = process.env.OLLAMA_API || 'http://localhost:11434';

// Serve static files
app.use(express.static('public'));
app.use(express.json());

// Agent definitions with personalities and roles
const agents = {
    researcher: {
        name: 'Researcher',
        model: 'llama3.2:latest',
        systemPrompt: `You are a Research Agent with deep analytical capabilities. Your role is to:
- Gather and synthesize information
- Provide well-researched insights
- Ask probing questions to understand the user better
- Remember user preferences and interests
- Build a knowledge base about the user over time

You have access to conversation history and user profile data. Use this to personalize your responses and show that you understand and remember the user.`,
        temperature: 0.7
    },
    coder: {
        name: 'Coder',
        model: 'llama3.2:latest',
        systemPrompt: `You are a Coding Agent specialized in software development. Your role is to:
- Write clean, efficient code
- Understand user's coding style and preferences over time
- Provide technical solutions
- Learn from user's feedback on code quality
- Adapt to the user's preferred languages and frameworks

You have access to conversation history. Learn the user's coding patterns, preferred tools, and technical level to provide increasingly personalized assistance.`,
        temperature: 0.5
    },
    critic: {
        name: 'Critic',
        model: 'llama3.2:latest',
        systemPrompt: `You are a Critical Thinking Agent focused on analysis and improvement. Your role is to:
- Provide constructive criticism
- Identify potential issues and improvements
- Challenge assumptions thoughtfully
- Learn what standards and values matter to the user
- Remember past critiques and user responses

Use conversation history to understand the user's goals, standards, and how they respond to feedback. Adapt your critique style to be most helpful for this specific user.`,
        temperature: 0.6
    },
    planner: {
        name: 'Planner',
        model: 'llama3.2:latest',
        systemPrompt: `You are a Planning Agent specialized in organization and strategy. Your role is to:
- Create structured plans and roadmaps
- Break down complex goals into actionable steps
- Learn the user's working style and preferences
- Remember ongoing projects and goals
- Track user's progress and adapt plans accordingly

You have access to conversation history. Use it to understand the user's priorities, time management style, and how they prefer to structure their work.`,
        temperature: 0.6
    }
};

// In-memory storage (will be persisted to files)
let conversationHistory = {};
let userProfile = {
    preferences: {},
    interests: [],
    learnings: [],
    interactions: 0,
    firstSeen: new Date().toISOString(),
    lastSeen: new Date().toISOString()
};

// Initialize conversation history for each agent
Object.keys(agents).forEach(agentKey => {
    conversationHistory[agentKey] = [];
});

// Load persistent data
async function loadData() {
    try {
        const historyData = await fs.readFile(path.join(__dirname, 'data', 'history.json'), 'utf8');
        conversationHistory = JSON.parse(historyData);
        console.log('✓ Loaded conversation history');
    } catch (error) {
        console.log('○ Starting with fresh conversation history');
    }

    try {
        const profileData = await fs.readFile(path.join(__dirname, 'data', 'profile.json'), 'utf8');
        userProfile = JSON.parse(profileData);
        console.log('✓ Loaded user profile');
    } catch (error) {
        console.log('○ Starting with fresh user profile');
    }
}

// Save persistent data
async function saveData() {
    try {
        await fs.mkdir(path.join(__dirname, 'data'), { recursive: true });

        await fs.writeFile(
            path.join(__dirname, 'data', 'history.json'),
            JSON.stringify(conversationHistory, null, 2)
        );

        await fs.writeFile(
            path.join(__dirname, 'data', 'profile.json'),
            JSON.stringify(userProfile, null, 2)
        );

        console.log('✓ Data saved');
    } catch (error) {
        console.error('✗ Error saving data:', error);
    }
}

// Auto-save every 5 minutes
setInterval(saveData, 5 * 60 * 1000);

// Web search function with rate limiting
let lastSearchTime = 0;
const SEARCH_COOLDOWN = 2000; // 2 seconds between searches

async function webSearch(query) {
    try {
        // Rate limiting: wait if needed
        const now = Date.now();
        const timeSinceLastSearch = now - lastSearchTime;
        if (timeSinceLastSearch < SEARCH_COOLDOWN) {
            const waitTime = SEARCH_COOLDOWN - timeSinceLastSearch;
            console.log(`⏱️ Rate limiting: waiting ${waitTime}ms...`);
            await new Promise(resolve => setTimeout(resolve, waitTime));
        }

        console.log(`🔍 Searching for: ${query}`);
        lastSearchTime = Date.now();

        const results = await search(query, {
            safeSearch: 0,
            locale: 'en-us'
        });

        const searchResults = results.results.slice(0, 5).map(r => ({
            title: r.title,
            description: r.description,
            url: r.url
        }));

        console.log(`✓ Found ${searchResults.length} results`);
        return searchResults;
    } catch (error) {
        console.error('✗ Search error:', error.message);

        // If rate limited, provide helpful message
        if (error.message.includes('anomaly') || error.message.includes('too quickly')) {
            console.warn('⚠️ Search rate limited - try again in a few seconds');
        }

        return [];
    }
}

// Build context for agent from history and profile
function buildContext(agentKey, currentMessage) {
    const agent = agents[agentKey];
    const history = conversationHistory[agentKey] || [];

    // Get recent conversation (last 10 exchanges)
    const recentHistory = history.slice(-20);

    // Build profile summary
    const profileSummary = `
USER PROFILE:
- Total interactions: ${userProfile.interactions}
- Member since: ${new Date(userProfile.firstSeen).toLocaleDateString()}
- Interests: ${userProfile.interests.join(', ') || 'Learning...'}
- Key learnings: ${userProfile.learnings.slice(-5).join('; ') || 'Building understanding...'}
`;

    // Add web search capability instructions
    const searchInstructions = `

WEB SEARCH CAPABILITY:
You can search the internet by including [SEARCH: your query] in your response.
When you need current information, news, or real-time data, use this format.
Example: "Let me check that for you. [SEARCH: latest AI developments 2025]"
The search results will be automatically retrieved and included in the conversation.`;

    return {
        model: agent.model,
        messages: [
            {
                role: 'system',
                content: agent.systemPrompt + '\n\n' + profileSummary + searchInstructions
            },
            ...recentHistory,
            {
                role: 'user',
                content: currentMessage
            }
        ],
        options: {
            temperature: agent.temperature
        },
        stream: false
    };
}

// Get available Ollama models (cached)
let modelCache = null;
let modelCacheTime = 0;
const MODEL_CACHE_TTL = 60000; // 1 minute

async function getAvailableModels() {
    const now = Date.now();
    if (modelCache && (now - modelCacheTime) < MODEL_CACHE_TTL) {
        return modelCache;
    }

    try {
        const response = await axios.get(`${OLLAMA_API}/api/tags`, { timeout: 3000 });
        const models = response.data.models
            .map(m => ({
                name: m.name,
                size: m.size,
                category: m.size < 1e9 ? 'tiny' : m.size < 3e9 ? 'small' : m.size < 7e9 ? 'medium' : 'large'
            }))
            .sort((a, b) => b.size - a.size); // Sort largest to smallest for cascade

        modelCache = models;
        modelCacheTime = now;
        return models;
    } catch (error) {
        return [];
    }
}

// Process search requests in agent response
async function processSearchRequests(response) {
    const searchPattern = /\[SEARCH:\s*([^\]]+)\]/g;
    let matches = [];
    let match;

    while ((match = searchPattern.exec(response)) !== null) {
        matches.push(match[1].trim());
    }

    if (matches.length === 0) {
        return response;
    }

    // Execute all searches
    let enhancedResponse = response;
    for (const query of matches) {
        const results = await webSearch(query);

        if (results.length > 0) {
            const searchSummary = results.map((r, i) =>
                `${i + 1}. ${r.title}\n   ${r.description}\n   Source: ${r.url}`
            ).join('\n\n');

            // Replace the search tag with results
            enhancedResponse = enhancedResponse.replace(
                `[SEARCH: ${query}]`,
                `\n\n🔍 SEARCH RESULTS for "${query}":\n\n${searchSummary}\n`
            );
        } else {
            enhancedResponse = enhancedResponse.replace(
                `[SEARCH: ${query}]`,
                `\n[Search failed: No results found for "${query}"]`
            );
        }
    }

    return enhancedResponse;
}

// Call Ollama API (with auto-fallback cascade and demo mode)
async function callOllama(agentKey, message, preferredModel = null) {
    const context = buildContext(agentKey, message);

    // If a preferred model is specified, try it first
    if (preferredModel) {
        try {
            context.model = preferredModel;
            const response = await axios.post(`${OLLAMA_API}/api/chat`, context, { timeout: 30000 });
            console.log(`✓ Success with model: ${preferredModel}`);
            const content = response.data.message.content;
            return await processSearchRequests(content);
        } catch (error) {
            console.warn(`✗ Failed with ${preferredModel}: ${error.message}`);
            // Fall through to cascade
        }
    }

    // Auto-fallback cascade: try available models from largest to smallest
    const availableModels = await getAvailableModels();

    for (const model of availableModels) {
        // Skip if we already tried this model
        if (model.name === preferredModel) continue;

        try {
            context.model = model.name;
            console.log(`→ Trying model: ${model.name} (${model.category})...`);
            const response = await axios.post(`${OLLAMA_API}/api/chat`, context, { timeout: 30000 });
            console.log(`✓ Success with model: ${model.name}`);
            const content = response.data.message.content;
            return await processSearchRequests(content);
        } catch (error) {
            console.warn(`✗ Failed with ${model.name}: ${error.message}`);
            // Continue to next model
        }
    }

    // All models failed, use demo mode
    console.warn('All Ollama models failed, using demo mode');

    // DEMO MODE - Simulated responses
    const agent = agents[agentKey];
    const responses = {
        researcher: [
            `As a Research Agent, I've analyzed your message: "${message}". Based on our conversation history, I notice you're interested in ${userProfile.interests.slice(0,3).join(', ') || 'various topics'}. I'm learning your preferences and will provide increasingly personalized insights over time.`,
            `Interesting question! Let me gather my thoughts on "${message}". I'm tracking ${userProfile.interactions} total interactions with you, which helps me understand your research style better.`,
            `I've processed your query about "${message}". As your dedicated researcher, I'm building a comprehensive understanding of your interests and information needs.`
        ],
        coder: [
            `As your Coding Agent, I'd approach "${message}" from a technical perspective. I've noticed through our ${userProfile.interactions} interactions that you prefer ${userProfile.interests[0] || 'efficient'} solutions. Here's my recommendation based on best practices.`,
            `Let me help you with "${message}". I'm learning your coding style and will adapt my suggestions accordingly. Each interaction helps me understand your technical preferences better.`,
            `Technical analysis of "${message}": I'm tracking patterns in your questions to provide more relevant code examples over time.`
        ],
        critic: [
            `From a critical thinking perspective on "${message}": I'm analyzing this through the lens of what I've learned about your standards and goals across ${userProfile.interactions} interactions.`,
            `Let me provide constructive feedback on "${message}". I'm learning what type of critique is most helpful for you personally.`,
            `Critical assessment: "${message}" - I'm adapting my critique style based on your previous responses and preferences.`
        ],
        planner: [
            `As your Planning Agent, I'll help organize "${message}" into actionable steps. Based on ${userProfile.interactions} interactions, I'm learning your preferred working style and time management approach.`,
            `Strategic planning for "${message}": I'm building an understanding of how you prefer to structure tasks and projects.`,
            `Let me create a plan for "${message}". I'm tracking your progress patterns to provide increasingly personalized planning assistance.`
        ]
    };

    const agentResponses = responses[agentKey] || responses.researcher;
    const randomResponse = agentResponses[Math.floor(Math.random() * agentResponses.length)];

    return `[DEMO MODE] ${randomResponse}\n\n💡 Tip: This is a simulated response. To use real AI, try a smaller model or run Ollama with more RAM.`;
}

// Update user profile based on interaction
function updateUserProfile(agentKey, userMessage, agentResponse) {
    userProfile.interactions++;
    userProfile.lastSeen = new Date().toISOString();

    // Simple keyword extraction for interests (can be enhanced)
    const keywords = userMessage.toLowerCase().match(/\b\w{5,}\b/g) || [];
    keywords.forEach(keyword => {
        if (!userProfile.interests.includes(keyword) && userProfile.interests.length < 50) {
            userProfile.interests.push(keyword);
        }
    });
}

// API Routes
app.get('/agents', (req, res) => {
    res.json(agents);
});

app.get('/profile', (req, res) => {
    res.json(userProfile);
});

// Get available Ollama models
app.get('/api/models', async (req, res) => {
    try {
        const response = await axios.get(`${OLLAMA_API}/api/tags`, { timeout: 3000 });
        const models = response.data.models.map(m => ({
            name: m.name,
            size: (m.size / (1024 * 1024 * 1024)).toFixed(2) + ' GB',
            sizeBytes: m.size,
            modified: m.modified_at,
            // Categorize by size for auto-fallback
            category: m.size < 1e9 ? 'tiny' : m.size < 3e9 ? 'small' : m.size < 7e9 ? 'medium' : 'large'
        }));

        // Sort by size (smallest first for fallback order)
        models.sort((a, b) => a.sizeBytes - b.sizeBytes);

        res.json({ models, demo: false });
    } catch (error) {
        console.warn('Could not fetch Ollama models, returning demo mode');
        // Return demo mode indicator
        res.json({
            models: [{ name: 'demo', size: '0 GB', category: 'demo' }],
            demo: true
        });
    }
});

app.post('/add-learning', async (req, res) => {
    const { learning } = req.body;
    if (learning) {
        userProfile.learnings.push(learning);
        await saveData();
        res.json({ success: true });
    } else {
        res.status(400).json({ error: 'Learning text required' });
    }
});

// Web search API endpoint
app.post('/api/search', async (req, res) => {
    const { query } = req.body;

    if (!query) {
        return res.status(400).json({ error: 'Query required' });
    }

    try {
        const results = await webSearch(query);
        res.json({ query, results, count: results.length });
    } catch (error) {
        console.error('Search API error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Simple REST API for messages (no Socket.IO required)
app.post('/api/message', async (req, res) => {
    const { agent: agentKey, message, model } = req.body;

    if (!agents[agentKey]) {
        return res.status(400).json({ error: 'Invalid agent' });
    }

    if (!message) {
        return res.status(400).json({ error: 'Message required' });
    }

    try {
        // Get response from Ollama (with optional model preference)
        const response = await callOllama(agentKey, message, model);

        // Save to conversation history
        conversationHistory[agentKey].push(
            { role: 'user', content: message },
            { role: 'assistant', content: response }
        );

        // Update user profile
        updateUserProfile(agentKey, message, response);

        // Auto-save after each interaction
        await saveData();

        res.json({ response: response });

    } catch (error) {
        console.error('Error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Socket.IO handlers
io.on('connection', (socket) => {
    console.log('Client connected');

    socket.on('send_message', async (data) => {
        const { agent: agentKey, message, model } = data;

        if (!agents[agentKey]) {
            socket.emit('error', { message: 'Invalid agent' });
            return;
        }

        try {
            // Emit thinking status
            socket.emit('agent_thinking', { agent: agentKey });

            // Get response from Ollama (with optional model preference)
            const response = await callOllama(agentKey, message, model);

            // Save to conversation history
            conversationHistory[agentKey].push(
                { role: 'user', content: message },
                { role: 'assistant', content: response }
            );

            // Update user profile
            updateUserProfile(agentKey, message, response);

            // Emit response
            socket.emit('agent_response', {
                agent: agentKey,
                message: response
            });

            // Auto-save after each interaction
            await saveData();

        } catch (error) {
            console.error('Error:', error);
            socket.emit('agent_response', {
                agent: agentKey,
                message: `ERROR: ${error.message}`
            });
        }
    });

    socket.on('start_discussion', async (data) => {
        const { topic } = data;

        try {
            // Start a discussion between agents
            const agentKeys = Object.keys(agents);
            let currentTopic = topic;

            for (let i = 0; i < 4; i++) {
                const agentKey = agentKeys[i % agentKeys.length];

                socket.emit('agent_thinking', { agent: agentKey });

                const response = await callOllama(agentKey,
                    `Regarding "${currentTopic}", share your perspective as a ${agents[agentKey].name}. Keep it concise (2-3 sentences).`
                );

                socket.emit('agent_response', {
                    agent: agentKey,
                    message: response
                });

                currentTopic = response;

                // Small delay between agents
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
        } catch (error) {
            console.error('Error in discussion:', error);
        }
    });

    socket.on('agent_conversation', async (data) => {
        const { from, to, message } = data;

        try {
            socket.emit('agent_thinking', { agent: to });

            const prompt = `Another agent (${agents[from].name}) asks you: "${message}". Respond as your role.`;
            const response = await callOllama(to, prompt);

            socket.emit('agent_response', {
                agent: to,
                message: response
            });
        } catch (error) {
            console.error('Error in agent conversation:', error);
        }
    });

    socket.on('clear_history', async (data) => {
        if (data.agent === 'all') {
            Object.keys(agents).forEach(key => {
                conversationHistory[key] = [];
            });
        } else if (conversationHistory[data.agent]) {
            conversationHistory[data.agent] = [];
        }
        await saveData();
    });

    socket.on('disconnect', () => {
        console.log('Client disconnected');
    });
});

// Start server
async function start() {
    await loadData();

    server.listen(PORT, () => {
        console.log(`
╔═══════════════════════════════════════════╗
║   AGENT TERMINAL SYSTEM - SERVER ONLINE   ║
╚═══════════════════════════════════════════╝

Server running on: http://localhost:${PORT}
Ollama API: ${OLLAMA_API}

Agents loaded: ${Object.keys(agents).length}
User interactions: ${userProfile.interactions}

Ready to learn and assist!
        `);
    });
}

// Graceful shutdown
process.on('SIGINT', async () => {
    console.log('\nSaving data before shutdown...');
    await saveData();
    process.exit(0);
});

start();
