/**
 * Integrations Module
 *
 * Central initialization for MCP, messaging, and automation features
 * This module is loaded by server.js to add MCP + Telegram capabilities
 */

const MCPIntegration = require('../mcp');
const MessageGateway = require('./message-gateway');
const TelegramAdapter = require('./telegram-bot');
const { workflowTemplates, findMatchingWorkflow, extractWorkflowParams } = require('../workflows/workflow-templates');

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
            // Check for workflow triggers
            const workflow = findMatchingWorkflow(message);

            if (workflow) {
                return await this.executeWorkflow(message, workflow, session);
            }

            // Otherwise, use standard agent chat
            return await this.standardAgentChat(message, session);

        } catch (error) {
            this.logger.error('Error processing message:', error);
            throw error;
        }
    }

    /**
     * Execute a matched workflow
     */
    async executeWorkflow(message, { key, workflow }, session) {
        this.logger.info(`Executing workflow: ${workflow.name}`);

        // Extract parameters from message
        const params = extractWorkflowParams(message, workflow);

        // Add user context
        params.userEmail = session.metadata?.email || session.preferences?.email;
        params.userName = session.metadata?.firstName || session.metadata?.username;
        params.userId = session.userId;

        // Check for missing required params
        const missingParams = (workflow.requiredParams || []).filter(p => !params[p]);

        if (missingParams.length > 0) {
            // Ask for missing params
            return `I'd like to help you with that! Could you please provide:\n\n` +
                missingParams.map(p => `- ${p}`).join('\n');
        }

        // Execute workflow
        if (this.workflowOrchestrator) {
            const result = await this.workflowOrchestrator.startWorkflow({
                name: workflow.name,
                description: workflow.description,
                steps: workflow.steps,
                initialContext: params
            });

            // Wait for completion or timeout
            const finalResult = await this.waitForWorkflow(result.id);

            if (finalResult.status === 'completed') {
                return this.formatWorkflowResult(finalResult);
            } else {
                return `Sorry, I encountered an issue: ${finalResult.errors.join(', ')}`;
            }
        }

        // Fallback to simple agent response
        return await this.standardAgentChat(message, session);
    }

    /**
     * Wait for workflow completion
     */
    async waitForWorkflow(workflowId, timeout = 120000) {
        const startTime = Date.now();

        while (Date.now() - startTime < timeout) {
            const workflow = this.workflowOrchestrator.getWorkflow(workflowId);

            if (!workflow || workflow.status !== 'running') {
                return workflow;
            }

            await new Promise(resolve => setTimeout(resolve, 1000));
        }

        return { status: 'timeout', errors: ['Workflow timed out'] };
    }

    /**
     * Format workflow result for user
     */
    formatWorkflowResult(workflow) {
        const lastResult = workflow.results[workflow.results.length - 1];

        if (lastResult && lastResult.result) {
            if (typeof lastResult.result === 'string') {
                return lastResult.result;
            }
            if (lastResult.result.response) {
                return lastResult.result.response;
            }
        }

        return `Completed: ${workflow.name}\n\nResults: ${workflow.results.length} steps executed successfully.`;
    }

    /**
     * Standard agent chat (no workflow)
     */
    async standardAgentChat(message, session) {
        // Build context from history
        const historyContext = session.history
            .slice(-10)
            .map(h => `${h.role}: ${h.content}`)
            .join('\n');

        // Get tool documentation
        const toolDocs = this.mcpIntegration ?
            this.mcpIntegration.generateToolDocumentation() : '';

        // Create enhanced prompt
        const systemPrompt = `You are a helpful AI assistant with access to various tools and capabilities.

You can:
- Browse websites and interact with them
- Search the web
- Send emails (if configured)
- Execute code
- Manage files

When a user asks you to do something that requires tools, use them appropriately.

${toolDocs}

Conversation context:
${historyContext}`;

        // Call the agent
        const response = await this.callOllama('researcher', message, null, [
            { role: 'system', content: systemPrompt }
        ]);

        // Check for tool calls in response
        if (this.toolSystem && response.includes('[TOOL:')) {
            const toolResults = await this.toolSystem.executeToolCalls(response);
            const enhancedResponse = this.toolSystem.replaceToolCalls(response, toolResults);
            return this.cleanResponse(enhancedResponse);
        }

        return this.cleanResponse(response);
    }

    /**
     * Clean up response for messaging
     */
    cleanResponse(response) {
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
