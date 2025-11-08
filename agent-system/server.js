require('dotenv').config();

const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const path = require('path');
const fs = require('fs').promises;
const axios = require('axios');
const { search } = require('duck-duck-scrape');
const { tavily } = require('@tavily/core');

// Security & Reliability imports
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const session = require('express-session');
const tmp = require('tmp-promise');
const pRetry = require('p-retry').default;
const CircuitBreaker = require('opossum');
const Ajv = require('ajv');
const winston = require('winston');
const CollaborationEngine = require('./collaboration-engine');

// Initialize logger
const logger = winston.createLogger({
    level: process.env.LOG_LEVEL || 'info',
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.errors({ stack: true }),
        winston.format.json()
    ),
    transports: [
        new winston.transports.File({ filename: 'logs/error.log', level: 'error' }),
        new winston.transports.File({ filename: 'logs/combined.log' }),
        new winston.transports.Console({
            format: winston.format.combine(
                winston.format.colorize(),
                winston.format.simple()
            )
        })
    ]
});

const app = express();
const server = http.createServer(app);

// Secure CORS configuration
const allowedOrigins = process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'];
const io = socketIo(server, {
    cors: {
        origin: (origin, callback) => {
            if (!origin || allowedOrigins.includes(origin)) {
                callback(null, true);
            } else {
                callback(new Error('Not allowed by CORS'));
            }
        },
        methods: ["GET", "POST"],
        credentials: true
    },
    transports: ['websocket', 'polling'],
    allowEIO3: true,
    pingTimeout: 60000,
    pingInterval: 25000
});

// Environment Configuration
const OLLAMA_API = process.env.OLLAMA_API || 'http://localhost:11434';
const TAVILY_API_KEY = process.env.TAVILY_API_KEY;
const PORT = process.env.PORT || 3000;
const SEARCH_RATE_LIMIT = parseInt(process.env.SEARCH_RATE_LIMIT) || 10; // searches per minute
const SEARCH_MAX_RESULTS = parseInt(process.env.SEARCH_MAX_RESULTS) || 5;
const SEARCH_TIMEOUT = parseInt(process.env.SEARCH_TIMEOUT) || 10000;

// Initialize Tavily client if API key is available
let tavilyClient = null;
if (TAVILY_API_KEY) {
    tavilyClient = tavily({ apiKey: TAVILY_API_KEY });
    logger.info('✓ Tavily AI search enabled');
} else {
    logger.warn('⚠️  No Tavily API key found - using DuckDuckGo fallback (may be rate limited)');
}

// Security middleware
app.use(helmet({
    contentSecurityPolicy: {
        directives: {
            defaultSrc: ["'self'"],
            scriptSrc: ["'self'", "'unsafe-inline'", "'unsafe-eval'"], // Needed for inline scripts
            scriptSrcAttr: ["'unsafe-inline'"], // Allow inline event handlers (onclick, etc)
            styleSrc: ["'self'", "'unsafe-inline'", "fonts.googleapis.com"],
            fontSrc: ["'self'", "fonts.gstatic.com"],
            imgSrc: ["'self'", "data:"],
            connectSrc: ["'self'", "ws:", "wss:"]
        }
    }
}));

// Rate limiting
const apiLimiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // Limit each IP to 100 requests per windowMs
    message: 'Too many requests from this IP, please try again later.',
    standardHeaders: true,
    legacyHeaders: false,
});

const messageLimiter = rateLimit({
    windowMs: 1 * 60 * 1000, // 1 minute
    max: 20, // Limit each IP to 20 messages per minute
    message: 'Too many messages, please slow down.',
});

// Session management
app.use(session({
    secret: process.env.SESSION_SECRET || 'change-this-in-production-' + Math.random(),
    resave: false,
    saveUninitialized: false,
    cookie: {
        secure: process.env.NODE_ENV === 'production',
        httpOnly: true,
        maxAge: 24 * 60 * 60 * 1000 // 24 hours
    }
}));

// Apply rate limiting to API routes
app.use('/api/', apiLimiter);
app.use('/api/message', messageLimiter);

// Serve static files with caching
app.use(express.static('public', {
    maxAge: '1d',
    etag: true
}));

// Body parser with size limits
app.use(express.json({ limit: '100kb' }));

// Request logging
app.use((req, res, next) => {
    logger.info(`${req.method} ${req.path}`, {
        ip: req.ip,
        userAgent: req.get('user-agent')
    });
    next();
});

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
let conversationSummaries = {}; // Hierarchical summaries
let userProfile = {
    preferences: {},
    interests: [],
    learnings: [],
    interactions: 0,
    firstSeen: new Date().toISOString(),
    lastSeen: new Date().toISOString(),
    personality: {
        communicationStyle: 'neutral', // casual, formal, technical
        detailPreference: 'medium', // brief, medium, detailed
        expertiseLevel: {}, // domain -> level (beginner, intermediate, expert)
        learningStyle: 'balanced' // visual, example-driven, theoretical, hands-on
    },
    conversationMetrics: {
        totalMessages: 0,
        averageLength: 0,
        topicsDiscussed: {},
        agentPreferences: {}
    }
};

// Conversation Management System
let conversations = {}; // In-memory conversation store: { conversationId: conversationObject }
let activeConversations = {}; // Track active conversation per user/session
const CONVERSATIONS_DIR = path.join(__dirname, 'data', 'conversations');

// Conversation schema
function createConversation(agent, name = null) {
    const id = `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const conversation = {
        id,
        name: name || `New ${agents[agent]?.name || 'Agent'} Chat`,
        agent,
        created: new Date().toISOString(),
        lastUpdated: new Date().toISOString(),
        messages: [],
        tags: [],
        pinned: false,
        archived: false,
        metadata: {
            messageCount: 0,
            searchCount: 0,
            collaborations: []
        }
    };
    conversations[id] = conversation;
    return conversation;
}

// Initialize conversation history and summaries for each agent
Object.keys(agents).forEach(agentKey => {
    conversationHistory[agentKey] = [];
    conversationSummaries[agentKey] = {
        sessionSummaries: [], // Recent session summaries
        weeklyDigest: '', // Week-level summary
        permanentLearnings: [] // Key insights that persist forever
    };
});

// Save single conversation to disk
async function saveConversation(conversationId) {
    try {
        const conv = conversations[conversationId];
        if (!conv) {
            throw new Error(`Conversation ${conversationId} not found`);
        }

        await fs.mkdir(CONVERSATIONS_DIR, { recursive: true });
        const filepath = path.join(CONVERSATIONS_DIR, `${conversationId}.json`);
        await fs.writeFile(filepath, JSON.stringify(conv, null, 2), 'utf8');
        logger.debug(`Saved conversation ${conversationId}`);
    } catch (error) {
        logger.error(`Failed to save conversation ${conversationId}:`, error);
        throw error;
    }
}

// Delete conversation from disk
async function deleteConversation(conversationId) {
    try {
        const filepath = path.join(CONVERSATIONS_DIR, `${conversationId}.json`);
        await fs.unlink(filepath);
        delete conversations[conversationId];
        logger.info(`Deleted conversation ${conversationId}`);
    } catch (error) {
        logger.error(`Failed to delete conversation ${conversationId}:`, error);
        throw error;
    }
}

// Initialize Collaboration Engine
let collaborationEngine = null;

// JSON Schema Validation
const ajv = new Ajv();

const messageSchema = {
    type: 'object',
    required: ['role', 'content'],
    properties: {
        role: { type: 'string', enum: ['user', 'assistant', 'system'] },
        content: { type: 'string', maxLength: 10000 }
    },
    additionalProperties: false
};

const historySchema = {
    type: 'object',
    patternProperties: {
        '^[a-z]+$': {
            type: 'array',
            items: messageSchema
        }
    }
};

const validateMessage = ajv.compile(messageSchema);
const validateHistory = ajv.compile(historySchema);

// Input Sanitization & Validation
function sanitizeInput(input) {
    if (typeof input !== 'string') {
        throw new Error('Invalid input type');
    }

    // Remove control characters except newlines and tabs
    let sanitized = input.replace(/[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]/g, '');

    // Trim and enforce length limits
    sanitized = sanitized.trim();
    if (sanitized.length > 10000) {
        throw new Error('Input too long (max 10,000 characters)');
    }
    if (sanitized.length === 0) {
        throw new Error('Input cannot be empty');
    }

    return sanitized;
}

function validateAgentKey(agentKey) {
    if (!agentKey || typeof agentKey !== 'string') {
        throw new Error('Invalid agent key');
    }
    if (!agents[agentKey]) {
        throw new Error('Unknown agent');
    }
    return agentKey;
}

function safeFilePath(filepath) {
    const resolved = path.resolve(filepath);
    const dataDir = path.resolve(__dirname, 'data');
    if (!resolved.startsWith(dataDir)) {
        throw new Error('Invalid file path - path traversal detected');
    }
    return resolved;
}

// Atomic write with backup
async function atomicWrite(filepath, data) {
    try {
        // Ensure directory exists
        await fs.mkdir(path.dirname(filepath), { recursive: true });

        // Create backup if file exists
        try {
            await fs.access(filepath);
            const backupPath = filepath.replace(/\.json$/, `.backup.json`);
            await fs.copyFile(filepath, backupPath);
        } catch (error) {
            // File doesn't exist yet, no backup needed
        }

        // Write to temp file first (in same directory for atomic rename)
        const tmpPath = filepath + '.tmp';

        await fs.writeFile(tmpPath, data, 'utf8');

        // Atomic rename (works on most filesystems)
        await fs.rename(tmpPath, filepath);

        logger.info(`Saved data to ${filepath}`);
    } catch (error) {
        logger.error(`Failed to save ${filepath}:`, error);
        throw error;
    }
}

// Load persistent data with validation
async function loadData() {
    // Load conversation history
    try {
        const historyData = await fs.readFile(path.join(__dirname, 'data', 'history.json'), 'utf8');
        const parsed = JSON.parse(historyData);

        // Validate schema
        if (validateHistory(parsed)) {
            conversationHistory = parsed;
            logger.info('✓ Loaded and validated conversation history');
        } else {
            throw new Error('Invalid history schema: ' + ajv.errorsText(validateHistory.errors));
        }
    } catch (error) {
        logger.warn('Could not load history, trying backup...', error.message);

        // Try backup
        try {
            const backupData = await fs.readFile(path.join(__dirname, 'data', 'history.backup.json'), 'utf8');
            const parsed = JSON.parse(backupData);
            if (validateHistory(parsed)) {
                conversationHistory = parsed;
                logger.info('✓ Restored from backup');
            }
        } catch (backupError) {
            logger.info('○ Starting with fresh conversation history');
            // Initialize default structure
            Object.keys(agents).forEach(agentKey => {
                conversationHistory[agentKey] = [];
            });
        }
    }

    // Load summaries
    try {
        const summariesData = await fs.readFile(path.join(__dirname, 'data', 'summaries.json'), 'utf8');
        conversationSummaries = JSON.parse(summariesData);
        logger.info('✓ Loaded conversation summaries');
    } catch (error) {
        logger.info('○ Starting with fresh summaries');
    }

    // Load conversations
    try {
        await fs.mkdir(CONVERSATIONS_DIR, { recursive: true });
        const files = await fs.readdir(CONVERSATIONS_DIR);
        const jsonFiles = files.filter(f => f.endsWith('.json'));

        for (const file of jsonFiles) {
            try {
                const data = await fs.readFile(path.join(CONVERSATIONS_DIR, file), 'utf8');
                const conv = JSON.parse(data);
                conversations[conv.id] = conv;
            } catch (err) {
                logger.warn(`Failed to load conversation ${file}:`, err.message);
            }
        }
        logger.info(`✓ Loaded ${Object.keys(conversations).length} conversations`);
    } catch (error) {
        logger.info('○ Starting with no saved conversations');
    }

    // Load user profile
    try {
        const profileData = await fs.readFile(path.join(__dirname, 'data', 'profile.json'), 'utf8');
        const parsed = JSON.parse(profileData);

        // Merge with defaults to handle schema evolution
        userProfile = {
            ...userProfile,
            ...parsed,
            personality: { ...userProfile.personality, ...parsed.personality },
            conversationMetrics: { ...userProfile.conversationMetrics, ...parsed.conversationMetrics }
        };

        logger.info('✓ Loaded user profile');
    } catch (error) {
        logger.info('○ Starting with fresh user profile');
    }
}

// Save persistent data with mutex to prevent race conditions
let saveMutex = Promise.resolve();
async function saveData() {
    // Queue saves to prevent concurrent writes
    saveMutex = saveMutex.then(async () => {
        try {
            await atomicWrite(
                path.join(__dirname, 'data', 'history.json'),
                JSON.stringify(conversationHistory, null, 2)
            );

            await atomicWrite(
                path.join(__dirname, 'data', 'summaries.json'),
                JSON.stringify(conversationSummaries, null, 2)
            );

            await atomicWrite(
                path.join(__dirname, 'data', 'profile.json'),
                JSON.stringify(userProfile, null, 2)
            );

            logger.info('✓ All data saved successfully');
        } catch (error) {
            logger.error('✗ Error saving data:', error);
            throw error;
        }
    });

    return saveMutex;
}

// Auto-save every 5 minutes
setInterval(saveData, 5 * 60 * 1000);

// Web search function with rate limiting
let searchCount = 0;
let searchWindowStart = Date.now();
const SEARCH_WINDOW = 60000; // 1 minute window for rate limiting

async function webSearch(query) {
    try {
        // Rate limiting: reset counter if window has passed
        const now = Date.now();
        if (now - searchWindowStart > SEARCH_WINDOW) {
            searchCount = 0;
            searchWindowStart = now;
        }

        // Check rate limit
        if (searchCount >= SEARCH_RATE_LIMIT) {
            const waitTime = SEARCH_WINDOW - (now - searchWindowStart);
            logger.warn(`⏱️  Rate limit reached (${SEARCH_RATE_LIMIT}/min). Waiting ${Math.ceil(waitTime/1000)}s...`);
            await new Promise(resolve => setTimeout(resolve, waitTime));
            searchCount = 0;
            searchWindowStart = Date.now();
        }

        logger.info(`🔍 Searching for: "${query}"`);
        searchCount++;

        // Try Tavily first (if API key is available)
        if (tavilyClient) {
            try {
                const response = await Promise.race([
                    tavilyClient.search(query, {
                        maxResults: SEARCH_MAX_RESULTS,
                        searchDepth: 'basic',
                        includeAnswer: false,
                        includeRawContent: false
                    }),
                    new Promise((_, reject) =>
                        setTimeout(() => reject(new Error('Tavily timeout')), SEARCH_TIMEOUT)
                    )
                ]);

                const searchResults = response.results.map(r => ({
                    title: r.title,
                    description: r.content,
                    url: r.url,
                    score: r.score
                }));

                logger.info(`✓ Tavily found ${searchResults.length} results`);
                return searchResults;
            } catch (tavilyError) {
                logger.warn(`⚠️  Tavily failed: ${tavilyError.message}, falling back to DuckDuckGo`);
            }
        }

        // Fallback to DuckDuckGo
        const results = await Promise.race([
            search(query, { safeSearch: 0, locale: 'en-us' }),
            new Promise((_, reject) =>
                setTimeout(() => reject(new Error('DuckDuckGo timeout')), SEARCH_TIMEOUT)
            )
        ]);

        const searchResults = results.results.slice(0, SEARCH_MAX_RESULTS).map(r => ({
            title: r.title,
            description: r.description,
            url: r.url
        }));

        logger.info(`✓ DuckDuckGo found ${searchResults.length} results`);
        return searchResults;

    } catch (error) {
        logger.error(`✗ Search error: ${error.message}`);

        // If rate limited, provide helpful message
        if (error.message.includes('anomaly') || error.message.includes('too quickly')) {
            logger.warn('⚠️  DuckDuckGo rate limited - consider adding Tavily API key');
        }

        return [];
    }
}

// Conversation summarization using smallest model
async function summarizeConversation(agentKey) {
    const history = conversationHistory[agentKey] || [];

    // Only summarize if we have enough history (> 30 messages)
    if (history.length < 30) {
        return null;
    }

    // Get messages to summarize (everything except last 10)
    const toSummarize = history.slice(0, -10);
    if (toSummarize.length === 0) {
        return null;
    }

    try {
        console.log(`📝 Summarizing ${toSummarize.length} messages for ${agentKey}...`);

        // Build conversation text
        const conversationText = toSummarize.map(msg =>
            `${msg.role}: ${msg.content}`
        ).join('\n');

        // Use smallest model for summarization
        const summaryPrompt = {
            model: 'qwen2.5:0.5b',
            messages: [
                {
                    role: 'system',
                    content: 'You are a conversation summarizer. Create a concise summary of the key points, topics, and user preferences mentioned. Focus on facts, preferences, and important context. Keep it under 200 words.'
                },
                {
                    role: 'user',
                    content: `Summarize this conversation:\n\n${conversationText}`
                }
            ],
            options: { temperature: 0.3 },
            stream: false
        };

        const response = await axios.post(`${OLLAMA_API}/api/chat`, summaryPrompt, { timeout: 30000 });
        const summary = response.data.message.content;

        console.log(`✓ Created summary (${summary.length} chars)`);

        // Store summary
        conversationSummaries[agentKey].sessionSummaries.push({
            timestamp: new Date().toISOString(),
            messageCount: toSummarize.length,
            summary: summary
        });

        // Keep only last 5 session summaries
        if (conversationSummaries[agentKey].sessionSummaries.length > 5) {
            conversationSummaries[agentKey].sessionSummaries.shift();
        }

        // Clear old messages from history (keep only last 10)
        conversationHistory[agentKey] = history.slice(-10);

        await saveData();
        return summary;
    } catch (error) {
        console.error('✗ Summarization error:', error.message);
        return null;
    }
}

// Enhanced user profiling with semantic extraction
function updateUserProfile(message) {
    // Update metrics
    userProfile.conversationMetrics.totalMessages++;
    userProfile.conversationMetrics.averageLength =
        (userProfile.conversationMetrics.averageLength * (userProfile.conversationMetrics.totalMessages - 1) +
            message.length) / userProfile.conversationMetrics.totalMessages;

    // Extract keywords (simple but effective)
    const words = message.toLowerCase().split(/\W+/).filter(w => w.length > 4);
    const stopWords = ['about', 'would', 'could', 'should', 'which', 'where', 'there', 'their', 'these', 'those'];
    const keywords = words.filter(w => !stopWords.includes(w));

    // Update interests
    keywords.forEach(keyword => {
        if (!userProfile.interests.includes(keyword) && userProfile.interests.length < 50) {
            userProfile.interests.push(keyword);
        }
    });

    // Detect communication style
    const hasExclamation = message.includes('!');
    const hasQuestion = message.includes('?');
    const avgWordLength = words.reduce((sum, w) => sum + w.length, 0) / (words.length || 1);

    if (avgWordLength > 6) {
        userProfile.personality.communicationStyle = 'technical';
    } else if (hasExclamation || message.toLowerCase().includes('lol') || message.toLowerCase().includes('cool')) {
        userProfile.personality.communicationStyle = 'casual';
    }

    // Detect detail preference
    if (message.length > 200 || message.split('\n').length > 3) {
        userProfile.personality.detailPreference = 'detailed';
    } else if (message.length < 50) {
        userProfile.personality.detailPreference = 'brief';
    }
}

// Build context for agent from history and profile
function buildContext(agentKey, currentMessage) {
    const agent = agents[agentKey];
    const history = conversationHistory[agentKey] || [];
    const summaries = conversationSummaries[agentKey] || { sessionSummaries: [], permanentLearnings: [] };

    // Get recent conversation (last 10 exchanges)
    const recentHistory = history.slice(-10);

    // Build conversation memory from summaries
    let conversationMemory = '';
    if (summaries.sessionSummaries.length > 0) {
        conversationMemory = '\n\nPAST CONVERSATION SUMMARIES:\n';
        summaries.sessionSummaries.forEach((summary, i) => {
            conversationMemory += `Session ${i + 1} (${new Date(summary.timestamp).toLocaleDateString()}): ${summary.summary}\n`;
        });
    }

    if (summaries.permanentLearnings.length > 0) {
        conversationMemory += '\n\nPERMANENT LEARNINGS:\n- ' + summaries.permanentLearnings.join('\n- ');
    }

    // Build enhanced profile summary
    const profileSummary = `
USER PROFILE:
- Total interactions: ${userProfile.interactions}
- Total messages: ${userProfile.conversationMetrics.totalMessages}
- Member since: ${new Date(userProfile.firstSeen).toLocaleDateString()}
- Communication style: ${userProfile.personality.communicationStyle}
- Detail preference: ${userProfile.personality.detailPreference}
- Learning style: ${userProfile.personality.learningStyle}
- Interests: ${userProfile.interests.slice(0, 10).join(', ') || 'Learning...'}
- Key learnings: ${userProfile.learnings.slice(-5).join('; ') || 'Building understanding...'}
${conversationMemory}`;

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
            .sort((a, b) => a.size - b.size); // Sort smallest to largest for faster response in resource-constrained environments

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

// Circuit Breaker for Ollama API
const ollamaBreaker = new CircuitBreaker(async (url, data, config) => {
    return await axios.post(url, data, config);
}, {
    timeout: 60000, // 60s timeout (larger models may need more time on first load)
    errorThresholdPercentage: 50, // Open circuit at 50% errors
    resetTimeout: 30000, // Try again after 30s
    name: 'ollamaAPI'
});

ollamaBreaker.fallback(() => {
    throw new Error('Ollama API circuit breaker is open');
});

ollamaBreaker.on('open', () => logger.warn('Circuit breaker opened - Ollama API not responding'));
ollamaBreaker.on('halfOpen', () => logger.info('Circuit breaker half-open - testing Ollama API'));
ollamaBreaker.on('close', () => logger.info('Circuit breaker closed - Ollama API recovered'));

// Retry wrapper with exponential backoff
async function retryOllamaCall(fn, options = {}) {
    return await pRetry(fn, {
        retries: 3,
        factor: 2,
        minTimeout: 1000,
        maxTimeout: 10000,
        onFailedAttempt: error => {
            logger.warn(`Ollama API attempt ${error.attemptNumber} failed. ${error.retriesLeft} retries left.`, {
                error: error.message
            });
        },
        ...options
    });
}

// Call Ollama API (with circuit breaker, retry, auto-fallback cascade and demo mode)
async function callOllama(agentKey, message, preferredModel = null) {
    const context = buildContext(agentKey, message);

    // If a preferred model is specified, try it first
    if (preferredModel) {
        try {
            context.model = preferredModel;
            const response = await retryOllamaCall(() =>
                ollamaBreaker.fire(`${OLLAMA_API}/api/chat`, context, { timeout: 30000 })
            );
            logger.info(`✓ Success with model: ${preferredModel}`);
            const content = response.data.message.content;
            return await processSearchRequests(content);
        } catch (error) {
            logger.warn(`✗ Failed with ${preferredModel}: ${error.message}`);
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
            logger.info(`→ Trying model: ${model.name} (${model.category})...`);
            const response = await retryOllamaCall(() =>
                ollamaBreaker.fire(`${OLLAMA_API}/api/chat`, context, { timeout: 30000 })
            );
            logger.info(`✓ Success with model: ${model.name}`);
            const content = response.data.message.content;
            return await processSearchRequests(content);
        } catch (error) {
            logger.warn(`✗ Failed with ${model.name}: ${error.message}`);
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

// Get memory data for dashboard
app.get('/api/memory', (req, res) => {
    const memoryData = {
        profile: userProfile,
        agents: {}
    };

    // Add agent-specific memory
    Object.keys(agents).forEach(agentKey => {
        memoryData.agents[agentKey] = {
            name: agents[agentKey].name,
            messageCount: conversationHistory[agentKey]?.length || 0,
            summaries: conversationSummaries[agentKey]?.sessionSummaries || [],
            permanentLearnings: conversationSummaries[agentKey]?.permanentLearnings || [],
            lastInteraction: conversationHistory[agentKey]?.slice(-1)[0]?.content.substring(0, 100) || 'No interactions yet'
        };
    });

    res.json(memoryData);
});

// Update memory (edit/delete)
app.post('/api/memory/update', async (req, res) => {
    const { action, target, value, agentKey } = req.body;

    try {
        if (action === 'delete_interest') {
            userProfile.interests = userProfile.interests.filter(i => i !== value);
        } else if (action === 'delete_learning') {
            userProfile.learnings = userProfile.learnings.filter(l => l !== value);
        } else if (action === 'add_permanent_learning' && agentKey) {
            if (!conversationSummaries[agentKey].permanentLearnings.includes(value)) {
                conversationSummaries[agentKey].permanentLearnings.push(value);
            }
        } else if (action === 'delete_permanent_learning' && agentKey) {
            conversationSummaries[agentKey].permanentLearnings =
                conversationSummaries[agentKey].permanentLearnings.filter(l => l !== value);
        } else if (action === 'clear_history' && agentKey) {
            conversationHistory[agentKey] = [];
            conversationSummaries[agentKey].sessionSummaries = [];
        }

        await saveData();
        res.json({ success: true });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
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

// ========================================
// CONVERSATION MANAGEMENT API ROUTES
// ========================================

// Get all conversations
app.get('/api/conversations', (req, res) => {
    try {
        const { agent, archived, search } = req.query;
        let results = Object.values(conversations);

        // Filter by agent
        if (agent) {
            results = results.filter(c => c.agent === agent);
        }

        // Filter by archived status
        if (archived !== undefined) {
            const isArchived = archived === 'true';
            results = results.filter(c => c.archived === isArchived);
        }

        // Search by name or messages
        if (search) {
            const searchLower = search.toLowerCase();
            results = results.filter(c =>
                c.name.toLowerCase().includes(searchLower) ||
                c.messages.some(m => m.content.toLowerCase().includes(searchLower))
            );
        }

        // Sort by last updated (newest first)
        results.sort((a, b) => new Date(b.lastUpdated) - new Date(a.lastUpdated));

        res.json({ conversations: results, count: results.length });
    } catch (error) {
        logger.error('Error fetching conversations:', error);
        res.status(500).json({ error: 'Failed to fetch conversations' });
    }
});

// Get specific conversation
app.get('/api/conversations/:id', (req, res) => {
    try {
        const conversation = conversations[req.params.id];
        if (!conversation) {
            return res.status(404).json({ error: 'Conversation not found' });
        }
        res.json(conversation);
    } catch (error) {
        logger.error('Error fetching conversation:', error);
        res.status(500).json({ error: 'Failed to fetch conversation' });
    }
});

// Create new conversation
app.post('/api/conversations', async (req, res) => {
    try {
        const { agent, name } = req.body;

        if (!agent || !agents[agent]) {
            return res.status(400).json({ error: 'Valid agent required' });
        }

        const conversation = createConversation(agent, name);
        await saveConversation(conversation.id);

        logger.info(`Created new conversation: ${conversation.id}`);
        res.json(conversation);
    } catch (error) {
        logger.error('Error creating conversation:', error);
        res.status(500).json({ error: 'Failed to create conversation' });
    }
});

// Update conversation (rename, tag, pin, archive)
app.put('/api/conversations/:id', async (req, res) => {
    try {
        const conversation = conversations[req.params.id];
        if (!conversation) {
            return res.status(404).json({ error: 'Conversation not found' });
        }

        const { name, tags, pinned, archived } = req.body;

        if (name !== undefined) conversation.name = name;
        if (tags !== undefined) conversation.tags = tags;
        if (pinned !== undefined) conversation.pinned = pinned;
        if (archived !== undefined) conversation.archived = archived;

        conversation.lastUpdated = new Date().toISOString();

        await saveConversation(conversation.id);
        logger.info(`Updated conversation: ${conversation.id}`);
        res.json(conversation);
    } catch (error) {
        logger.error('Error updating conversation:', error);
        res.status(500).json({ error: 'Failed to update conversation' });
    }
});

// Delete conversation
app.delete('/api/conversations/:id', async (req, res) => {
    try {
        const conversation = conversations[req.params.id];
        if (!conversation) {
            return res.status(404).json({ error: 'Conversation not found' });
        }

        await deleteConversation(conversation.id);
        res.json({ success: true, message: 'Conversation deleted' });
    } catch (error) {
        logger.error('Error deleting conversation:', error);
        res.status(500).json({ error: 'Failed to delete conversation' });
    }
});

// Export conversation
app.post('/api/conversations/:id/export', (req, res) => {
    try {
        const conversation = conversations[req.params.id];
        if (!conversation) {
            return res.status(404).json({ error: 'Conversation not found' });
        }

        const { format = 'markdown' } = req.body;

        if (format === 'json') {
            res.json(conversation);
        } else if (format === 'markdown') {
            const md = `# ${conversation.name}\n\n` +
                `**Agent:** ${conversation.agent}\n` +
                `**Created:** ${new Date(conversation.created).toLocaleString()}\n` +
                `**Messages:** ${conversation.metadata.messageCount}\n\n` +
                `---\n\n` +
                conversation.messages.map(m =>
                    `**${m.role === 'user' ? 'You' : agents[conversation.agent].name}:**\n${m.content}\n`
                ).join('\n');

            res.setHeader('Content-Type', 'text/markdown');
            res.setHeader('Content-Disposition', `attachment; filename="${conversation.name}.md"`);
            res.send(md);
        } else {
            res.status(400).json({ error: 'Invalid format. Use "json" or "markdown"' });
        }
    } catch (error) {
        logger.error('Error exporting conversation:', error);
        res.status(500).json({ error: 'Failed to export conversation' });
    }
});

// Simple REST API for messages (no Socket.IO required)
app.post('/api/message', async (req, res) => {
    try {
        // Validate and sanitize inputs
        const agentKey = validateAgentKey(req.body.agent);
        const message = sanitizeInput(req.body.message);
        const model = req.body.model || null;

        logger.info(`Message received from ${agentKey}`, {
            messageLength: message.length,
            model: model || 'auto'
        });

        // Get response from Ollama (with optional model preference)
        const response = await callOllama(agentKey, message, model);

        // Update user profiling based on message and response
        updateUserProfile(agentKey, message, response);

        // Validate response
        if (typeof response !== 'string' || response.length === 0) {
            throw new Error('Invalid response from model');
        }

        // Create validated message objects
        const userMsg = { role: 'user', content: message };
        const assistantMsg = { role: 'assistant', content: response };

        // Validate schema before saving
        if (!validateMessage(userMsg) || !validateMessage(assistantMsg)) {
            throw new Error('Message validation failed');
        }

        // Save to conversation history
        conversationHistory[agentKey].push(userMsg, assistantMsg);

        // Update user profile
        userProfile.interactions++;
        userProfile.lastSeen = new Date().toISOString();

        // Track agent preference
        userProfile.conversationMetrics.agentPreferences[agentKey] =
            (userProfile.conversationMetrics.agentPreferences[agentKey] || 0) + 1;

        // Check if we need to summarize (every 30 messages)
        if (conversationHistory[agentKey].length > 30) {
            // Summarize in background (don't wait)
            summarizeConversation(agentKey).catch(err =>
                logger.error('Background summarization error:', err)
            );
        }

        // Auto-save after each interaction
        await saveData();

        res.json({ response: response });

    } catch (error) {
        logger.error('API message error:', {
            error: error.message,
            stack: error.stack
        });

        // Don't leak internal errors
        if (error.message.includes('too long') || error.message.includes('empty') || error.message.includes('Invalid')) {
            res.status(400).json({ error: error.message });
        } else {
            res.status(500).json({ error: 'Internal server error' });
        }
    }
});

// ========================================
// COLLABORATION API ROUTES
// ========================================

// Initialize collaboration engine on first use
function getCollaborationEngine() {
    if (!collaborationEngine) {
        try {
            collaborationEngine = new CollaborationEngine(agents, callOllama);
            logger.info('Collaboration engine initialized');
        } catch (error) {
            logger.error('Failed to initialize collaboration engine:', error);
            throw new Error(`Collaboration engine initialization failed: ${error.message}`);
        }
    }
    return collaborationEngine;
}

// Get available collaboration templates
app.get('/api/collaboration/templates', (req, res) => {
    try {
        const engine = getCollaborationEngine();
        const templates = engine.getTemplates();

        // Always return at least an empty array, never fail completely
        res.json({
            templates: templates || [],
            fallbackMode: !templates || templates.length === 0
        });
    } catch (error) {
        logger.error('Error fetching templates:', error);

        // Return empty templates array instead of error
        // This allows frontend to still open modal in custom mode
        res.json({
            templates: [],
            fallbackMode: true,
            error: 'Engine unavailable - custom mode only'
        });
    }
});

// Start a new collaboration session
app.post('/api/collaboration/start', async (req, res) => {
    try {
        const { task, template, participants, rounds, maxRounds } = req.body;

        // Validate inputs
        if (!task || typeof task !== 'string' || task.trim().length === 0) {
            return res.status(400).json({ error: 'Task is required' });
        }

        if (!participants || !Array.isArray(participants) || participants.length < 2) {
            return res.status(400).json({ error: 'At least 2 participants are required' });
        }

        // Validate all participants exist
        for (const agentKey of participants) {
            if (!agents[agentKey]) {
                return res.status(400).json({ error: `Unknown agent: ${agentKey}` });
            }
        }

        const engine = getCollaborationEngine();
        const session = await engine.startCollaboration({
            task: sanitizeInput(task),
            template: template || 'custom',
            participants,
            rounds: rounds || 3,
            maxRounds: maxRounds || 5
        });

        logger.info(`Collaboration session started: ${session.id}`, {
            template: session.template,
            participants: session.participants
        });

        res.json({
            sessionId: session.id,
            status: session.status,
            template: session.template,
            participants: session.participants,
            rounds: session.rounds
        });

    } catch (error) {
        logger.error('Error starting collaboration:', error);
        res.status(500).json({ error: error.message });
    }
});

// Get collaboration session status
app.get('/api/collaboration/:sessionId', (req, res) => {
    try {
        const { sessionId } = req.params;
        const engine = getCollaborationEngine();
        const session = engine.getSession(sessionId);

        if (!session) {
            return res.status(404).json({ error: 'Session not found' });
        }

        res.json({
            id: session.id,
            status: session.status,
            task: session.task,
            template: session.template,
            participants: session.participants,
            currentRound: session.currentRound,
            totalRounds: session.rounds,
            startTime: session.startTime,
            endTime: session.endTime,
            history: session.history,
            synthesis: session.synthesis,
            error: session.error
        });

    } catch (error) {
        logger.error('Error fetching session:', error);
        res.status(500).json({ error: 'Failed to fetch session' });
    }
});

// Get all collaboration sessions
app.get('/api/collaboration', (req, res) => {
    try {
        const engine = getCollaborationEngine();
        const sessions = engine.getAllSessions();

        // Return summary info only
        const summaries = sessions.map(s => ({
            id: s.id,
            status: s.status,
            task: s.task.substring(0, 100) + (s.task.length > 100 ? '...' : ''),
            template: s.template,
            participants: s.participants,
            currentRound: s.currentRound,
            totalRounds: s.rounds,
            startTime: s.startTime,
            endTime: s.endTime
        }));

        res.json({ sessions: summaries });

    } catch (error) {
        logger.error('Error fetching sessions:', error);
        res.status(500).json({ error: 'Failed to fetch sessions' });
    }
});

// Cancel a running collaboration session
app.post('/api/collaboration/:sessionId/cancel', (req, res) => {
    try {
        const { sessionId } = req.params;
        const engine = getCollaborationEngine();
        const cancelled = engine.cancelSession(sessionId);

        if (!cancelled) {
            return res.status(404).json({ error: 'Session not found or already completed' });
        }

        logger.info(`Collaboration session cancelled: ${sessionId}`);
        res.json({ success: true, message: 'Session cancelled' });

    } catch (error) {
        logger.error('Error cancelling session:', error);
        res.status(500).json({ error: 'Failed to cancel session' });
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

// Health Check Endpoint
app.get('/health', async (req, res) => {
    const healthCheck = {
        uptime: process.uptime(),
        timestamp: Date.now(),
        status: 'ok',
        memory: process.memoryUsage(),
        services: {
            ollama: 'unknown',
            agents: Object.keys(agents).length,
            conversations: Object.keys(conversationHistory).reduce((sum, key) =>
                sum + conversationHistory[key].length, 0)
        }
    };

    // Check Ollama connectivity
    try {
        await axios.get(`${OLLAMA_API}/api/tags`, { timeout: 2000 });
        healthCheck.services.ollama = 'connected';
    } catch (error) {
        healthCheck.services.ollama = 'disconnected';
        healthCheck.status = 'degraded';
    }

    const statusCode = healthCheck.status === 'ok' ? 200 : 503;
    res.status(statusCode).json(healthCheck);
});

// Readiness probe
app.get('/ready', (req, res) => {
    res.json({ ready: true });
});

// Start server
async function start() {
    await loadData();

    server.listen(PORT, () => {
        logger.info(`
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
async function gracefulShutdown(signal) {
    logger.info(`\n${signal} received. Starting graceful shutdown...`);

    // Stop accepting new requests
    server.close(() => {
        logger.info('HTTP server closed');
    });

    try {
        // Save all data
        logger.info('Saving data...');
        await saveData();
        logger.info('✓ Data saved successfully');

        // Close socket connections
        io.close(() => {
            logger.info('Socket.IO connections closed');
        });

        logger.info('✓ Graceful shutdown complete. Goodbye!');
        process.exit(0);
    } catch (error) {
        logger.error('Error during shutdown:', error);
        process.exit(1);
    }
}

process.on('SIGINT', () => gracefulShutdown('SIGINT'));
process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));

// Unhandled rejection handling
process.on('unhandledRejection', (reason, promise) => {
    logger.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

process.on('uncaughtException', (error) => {
    logger.error('Uncaught Exception:', error);
    gracefulShutdown('UNCAUGHT_EXCEPTION');
});

// Create logs directory
const fsSync = require('fs');
if (!fsSync.existsSync('./logs')) {
    fsSync.mkdirSync('./logs');
}

start();
