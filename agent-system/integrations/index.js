/**
 * Integrations Module
 *
 * Central initialization for MCP, messaging, and automation features
 * This module is loaded by server.js to add MCP + Telegram capabilities
 */

const MCPIntegration = require('../mcp');
const MessageGateway = require('./message-gateway');
const TelegramAdapter = require('./telegram-bot');
const SmartAgent = require('./smart-agent');
const PersonalKnowledgeBase = require('../personalization/knowledge-base');
const SystemMonitor = require('../monitoring');
const CommandSystem = require('../commands');

class IntegrationsManager {
    constructor(options = {}) {
        this.logger = options.logger || console;
        this.toolSystem = options.toolSystem;
        this.callOllama = options.callOllama;  // Function to call Ollama
        this.agents = options.agents;
        this.workflowOrchestrator = options.workflowOrchestrator;

        this.mcpIntegration = null;
        this.messageGateway = null;
        this.telegramAdapter = null;
        this.smartAgent = null;
        this.knowledgeBase = null;
        this.commandSystem = null;
        this.monitor = new SystemMonitor({ logger: this.logger });
        this.initialized = false;
    }

    /**
     * Initialize all integrations
     */
    async initialize() {
        if (this.initialized) {
            this.logger.warn('Integrations already initialized');
            return;
        }

        this.logger.info('Initializing integrations...');

        try {
            // Initialize MCP
            await this.initializeMCP();

            // Initialize Personal Knowledge Base
            await this.initializeKnowledgeBase();

            // Initialize Smart Agent with progress callback and knowledge base
            this.smartAgent = new SmartAgent({
                logger: this.logger,
                callOllama: this.callOllama,
                toolSystem: this.toolSystem,
                browserAutomation: this.mcpIntegration?.browserAutomation,
                knowledgeBase: this.knowledgeBase,
                onProgress: async (userId, progress) => {
                    await this.sendProgressToUser(userId, progress);
                }
            });
            this.logger.info('Smart agent initialized with knowledge base');

            // Initialize Command System
            await this.initializeCommandSystem();

            // Initialize Messaging
            await this.initializeMessaging();

            // Register components with monitor
            this.monitor.registerComponents({
                knowledgeBase: this.knowledgeBase,
                mcpIntegration: this.mcpIntegration,
                telegramAdapter: this.telegramAdapter,
                smartAgent: this.smartAgent
            });

            this.initialized = true;
            this.logger.info('All integrations initialized successfully');
            this.monitor.logActivity('system', 'System initialized');

            return this.getStatus();
        } catch (error) {
            this.logger.error('Failed to initialize integrations:', error);
            throw error;
        }
    }

    /**
     * Initialize MCP integration
     */
    async initializeMCP() {
        this.mcpIntegration = new MCPIntegration({
            logger: this.logger,
            toolSystem: this.toolSystem
        });

        await this.mcpIntegration.initialize();
        this.logger.info('MCP integration ready');
    }

    /**
     * Initialize Personal Knowledge Base
     */
    async initializeKnowledgeBase() {
        this.knowledgeBase = new PersonalKnowledgeBase({
            logger: this.logger,
            callOllama: this.callOllama
        });

        await this.knowledgeBase.initialize();
        this.logger.info(`Knowledge base ready: ${this.knowledgeBase.getStats().documents} documents, ${this.knowledgeBase.getStats().facts} facts`);
    }

    /**
     * Initialize Command System
     */
    async initializeCommandSystem() {
        this.commandSystem = new CommandSystem({
            logger: this.logger,
            smartAgent: this.smartAgent,
            knowledgeBase: this.knowledgeBase,
            monitor: this.monitor
        });

        await this.commandSystem.initialize();
        this.logger.info(`Command system ready: ${this.commandSystem.userCommands.size} custom commands`);
    }

    /**
     * Initialize messaging gateway
     */
    async initializeMessaging() {
        this.messageGateway = new MessageGateway({
            logger: this.logger
        });

        // Load existing sessions
        await this.messageGateway.loadSessions();

        // Set up agent handler
        this.messageGateway.setAgentHandler(async (request) => {
            return await this.handleAgentMessage(request);
        });

        // Handle clear history events
        this.messageGateway.on('clearHistory', ({ platform, userId }) => {
            this.messageGateway.clearSessionHistory(platform, userId);
        });

        // Initialize Telegram if configured
        if (process.env.TELEGRAM_BOT_TOKEN || process.env.ENABLE_TELEGRAM === 'true') {
            await this.initializeTelegram();
        }

        // Add allowed users from environment
        if (process.env.ALLOWED_TELEGRAM_USERS) {
            const users = process.env.ALLOWED_TELEGRAM_USERS.split(',');
            users.forEach(userId => this.messageGateway.addAllowedUser(userId.trim()));
            this.logger.info(`Added ${users.length} allowed Telegram users`);
        }

        this.logger.info('Messaging gateway ready');
    }

    /**
     * Initialize Telegram bot
     */
    async initializeTelegram() {
        const token = process.env.TELEGRAM_BOT_TOKEN;

        if (!token) {
            this.logger.warn('TELEGRAM_BOT_TOKEN not set, Telegram integration disabled');
            return;
        }

        this.telegramAdapter = new TelegramAdapter(token, {
            logger: this.logger
        });

        this.messageGateway.registerAdapter('telegram', this.telegramAdapter);

        // Get bot info
        try {
            const botInfo = await this.telegramAdapter.getBotInfo();
            this.logger.info(`Telegram bot connected: @${botInfo.username}`);
        } catch (error) {
            this.logger.error('Failed to get Telegram bot info:', error);
        }
    }

    /**
     * Handle incoming agent message from any platform
     */
    async handleAgentMessage(request) {
        const { message, session, platform, userId } = request;

        this.logger.info(`Processing message from ${platform}:${userId}: ${message.substring(0, 50)}...`);
        this.monitor.increment('messages', 'received');

        try {
            // Check for commands first (messages starting with /)
            if (this.commandSystem && message.trim().startsWith('/')) {
                const cmdResult = await this.commandSystem.processMessage(message, {
                    userId,
                    platform,
                    session
                });

                // If command returns an expansion, process it through SmartAgent
                if (cmdResult && typeof cmdResult === 'object' && cmdResult.type === 'expand') {
                    this.logger.info(`Command expanded to: ${cmdResult.text}`);
                    // Process the expanded command through SmartAgent
                    if (this.smartAgent) {
                        const response = await this.smartAgent.handleRequest(cmdResult.text, session);
                        this.monitor.increment('messages', 'processed');
                        return this.cleanResponse(response);
                    }
                }

                // If command returned a direct response, return it
                if (cmdResult !== null) {
                    this.monitor.increment('messages', 'processed');
                    return this.cleanResponse(cmdResult);
                }
            }

            // Use the smart agent for ALL non-command requests
            // It will analyze intent, plan, and execute automatically
            if (this.smartAgent) {
                const response = await this.smartAgent.handleRequest(message, session);
                this.monitor.increment('messages', 'processed');
                return this.cleanResponse(response);
            }

            // Fallback if smart agent not available
            return await this.fallbackChat(message, session);

        } catch (error) {
            this.logger.error('Error processing message:', error);
            this.monitor.increment('messages', 'errors');
            throw error;
        }
    }

    /**
     * Fallback chat when smart agent unavailable
     */
    async fallbackChat(message, session) {
        const historyContext = session.history
            .slice(-6)
            .map(h => `${h.role}: ${h.content}`)
            .join('\n');

        const prompt = `You are a helpful AI assistant.

Conversation:
${historyContext}

User: ${message}

Respond helpfully:`;

        return await this.callOllama('researcher', prompt);
    }

    /**
     * Send progress update to user
     */
    async sendProgressToUser(userId, progress) {
        // Find which platform this user is on
        for (const [platform, adapter] of this.messageGateway.adapters) {
            const session = this.messageGateway.getSession(platform, userId);
            if (session) {
                try {
                    // Format progress message with emoji indicators
                    const emoji = {
                        'thinking': '🤔',
                        'planning': '📋',
                        'executing': '⚙️',
                        'error': '⚠️',
                        'success': '✅'
                    }[progress.type] || '💭';

                    await adapter.sendTypingIndicator(userId);

                    // For longer operations, send actual progress message
                    if (progress.type === 'executing') {
                        await adapter.sendMessage(userId, `${emoji} ${progress.message}`);
                    }
                } catch (error) {
                    this.logger.warn(`Failed to send progress to ${userId}:`, error.message);
                }
                break;
            }
        }
    }

    /**
     * Clean up response for messaging
     */
    cleanResponse(response) {
        if (!response) return "I'm not sure how to respond to that.";

        // Remove excessive whitespace
        let cleaned = response.replace(/\n{3,}/g, '\n\n').trim();

        // Truncate if too long (for Telegram)
        if (cleaned.length > 4000) {
            cleaned = cleaned.substring(0, 3900) + '\n\n... (response truncated)';
        }

        return cleaned;
    }

    /**
     * Get integrations status
     */
    getStatus() {
        return {
            initialized: this.initialized,
            mcp: this.mcpIntegration ? this.mcpIntegration.getStatus() : null,
            messaging: this.messageGateway ? this.messageGateway.getStats() : null,
            telegram: this.telegramAdapter ? this.telegramAdapter.isRunning() : false,
            knowledgeBase: this.knowledgeBase ? this.knowledgeBase.getStats() : null
        };
    }

    /**
     * Add API routes to Express app
     */
    addRoutes(app) {
        // MCP status
        app.get('/api/mcp/status', (req, res) => {
            if (!this.mcpIntegration) {
                return res.status(503).json({ error: 'MCP not initialized' });
            }
            res.json(this.mcpIntegration.getStatus());
        });

        // List MCP tools
        app.get('/api/mcp/tools', (req, res) => {
            if (!this.mcpIntegration) {
                return res.status(503).json({ error: 'MCP not initialized' });
            }
            res.json({ tools: this.mcpIntegration.getAllTools() });
        });

        // Messaging stats
        app.get('/api/messaging/stats', (req, res) => {
            if (!this.messageGateway) {
                return res.status(503).json({ error: 'Messaging not initialized' });
            }
            res.json(this.messageGateway.getStats());
        });

        // Integrations status
        app.get('/api/integrations/status', (req, res) => {
            res.json(this.getStatus());
        });

        // Send message to user (for testing)
        app.post('/api/messaging/send', async (req, res) => {
            try {
                const { platform, userId, message } = req.body;
                await this.messageGateway.sendToUser(platform, userId, message);
                res.json({ success: true });
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        // ==========================================
        // KNOWLEDGE BASE ENDPOINTS
        // ==========================================

        // Knowledge base stats
        app.get('/api/knowledge/stats', (req, res) => {
            if (!this.knowledgeBase) {
                return res.status(503).json({ error: 'Knowledge base not initialized' });
            }
            res.json(this.knowledgeBase.getStats());
        });

        // Get facts summary
        app.get('/api/knowledge/facts', (req, res) => {
            if (!this.knowledgeBase) {
                return res.status(503).json({ error: 'Knowledge base not initialized' });
            }
            res.json({
                summary: this.knowledgeBase.getFactsSummary(),
                count: this.knowledgeBase.facts.size
            });
        });

        // Index a single file
        app.post('/api/knowledge/index/file', async (req, res) => {
            if (!this.knowledgeBase) {
                return res.status(503).json({ error: 'Knowledge base not initialized' });
            }
            try {
                const { filePath } = req.body;
                if (!filePath) {
                    return res.status(400).json({ error: 'filePath is required' });
                }
                const docId = await this.knowledgeBase.indexFile(filePath);
                res.json({ success: true, docId });
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        // Index a directory
        app.post('/api/knowledge/index/directory', async (req, res) => {
            if (!this.knowledgeBase) {
                return res.status(503).json({ error: 'Knowledge base not initialized' });
            }
            try {
                const { dirPath, extensions } = req.body;
                if (!dirPath) {
                    return res.status(400).json({ error: 'dirPath is required' });
                }
                const docIds = await this.knowledgeBase.indexDirectory(dirPath, extensions);
                res.json({ success: true, indexed: docIds.length, docIds });
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        // Index emails
        app.post('/api/knowledge/index/emails', async (req, res) => {
            if (!this.knowledgeBase) {
                return res.status(503).json({ error: 'Knowledge base not initialized' });
            }
            try {
                const { emailsPath } = req.body;
                if (!emailsPath) {
                    return res.status(400).json({ error: 'emailsPath is required' });
                }
                const count = await this.knowledgeBase.indexEmails(emailsPath);
                res.json({ success: true, indexed: count });
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        // Index text directly (notes, etc.)
        app.post('/api/knowledge/index/text', async (req, res) => {
            if (!this.knowledgeBase) {
                return res.status(503).json({ error: 'Knowledge base not initialized' });
            }
            try {
                const { text, metadata } = req.body;
                if (!text) {
                    return res.status(400).json({ error: 'text is required' });
                }
                const docId = await this.knowledgeBase.indexText(text, metadata);
                res.json({ success: true, docId });
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        // Store a fact manually
        app.post('/api/knowledge/facts', async (req, res) => {
            if (!this.knowledgeBase) {
                return res.status(503).json({ error: 'Knowledge base not initialized' });
            }
            try {
                const { fact, category, confidence } = req.body;
                if (!fact) {
                    return res.status(400).json({ error: 'fact is required' });
                }
                const factId = await this.knowledgeBase.storeFact(fact, { category, confidence });
                res.json({ success: true, factId });
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        // Search knowledge base
        app.get('/api/knowledge/search', (req, res) => {
            if (!this.knowledgeBase) {
                return res.status(503).json({ error: 'Knowledge base not initialized' });
            }
            try {
                const { query, limit } = req.query;
                if (!query) {
                    return res.status(400).json({ error: 'query parameter is required' });
                }
                const documents = this.knowledgeBase.search(query, parseInt(limit) || 5);
                const facts = this.knowledgeBase.searchFacts(query, parseInt(limit) || 10);
                res.json({ documents, facts });
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        // Update user profile
        app.post('/api/knowledge/profile', async (req, res) => {
            if (!this.knowledgeBase) {
                return res.status(503).json({ error: 'Knowledge base not initialized' });
            }
            try {
                const updates = req.body;
                await this.knowledgeBase.updateProfile(updates);
                res.json({ success: true, profile: this.knowledgeBase.userProfile });
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        // Get user profile
        app.get('/api/knowledge/profile', (req, res) => {
            if (!this.knowledgeBase) {
                return res.status(503).json({ error: 'Knowledge base not initialized' });
            }
            res.json(this.knowledgeBase.userProfile);
        });

        // Add monitoring routes
        this.monitor.addRoutes(app);

        this.logger.info('Integration routes added to Express app');
    }

    /**
     * Shutdown all integrations
     */
    async shutdown() {
        this.logger.info('Shutting down integrations...');

        if (this.messageGateway) {
            await this.messageGateway.shutdown();
        }

        if (this.mcpIntegration) {
            await this.mcpIntegration.shutdown();
        }

        this.initialized = false;
        this.logger.info('Integrations shutdown complete');
    }
}

module.exports = IntegrationsManager;
