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

            // Initialize Smart Agent
            this.smartAgent = new SmartAgent({
                logger: this.logger,
                callOllama: this.callOllama,
                toolSystem: this.toolSystem,
                browserAutomation: this.mcpIntegration?.browserAutomation
            });
            this.logger.info('Smart agent initialized');

            // Initialize Messaging
            await this.initializeMessaging();

            this.initialized = true;
            this.logger.info('All integrations initialized successfully');

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

        try {
            // Use the smart agent for ALL requests
            // It will analyze intent, plan, and execute automatically
            if (this.smartAgent) {
                const response = await this.smartAgent.handleRequest(message, session);
                return this.cleanResponse(response);
            }

            // Fallback if smart agent not available
            return await this.fallbackChat(message, session);

        } catch (error) {
            this.logger.error('Error processing message:', error);
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
            telegram: this.telegramAdapter ? this.telegramAdapter.isRunning() : false
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
