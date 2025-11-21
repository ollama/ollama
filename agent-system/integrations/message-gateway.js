/**
 * Message Gateway
 *
 * Unified message handling for multiple platforms (Telegram, WhatsApp, etc.)
 * Routes messages to the agent system and manages user sessions
 */

const EventEmitter = require('events');
const fs = require('fs').promises;
const path = require('path');

class MessageGateway extends EventEmitter {
    constructor(options = {}) {
        super();
        this.logger = options.logger || console;
        this.adapters = new Map();           // platform → adapter
        this.userSessions = new Map();       // sessionKey → session
        this.allowedUsers = new Set();       // Whitelist of allowed user IDs
        this.agentHandler = null;            // Function to process messages
        this.sessionTimeout = options.sessionTimeout || 3600000; // 1 hour
        this.maxHistoryLength = options.maxHistoryLength || 20;
        this.sessionsFile = options.sessionsFile || path.join(__dirname, '../data/user-sessions.json');

        // Start session cleanup interval
        this.cleanupInterval = setInterval(() => {
            this.cleanupExpiredSessions();
        }, 300000); // Every 5 minutes
    }

    /**
     * Set the agent handler function
     * This function will be called to process each message
     */
    setAgentHandler(handler) {
        this.agentHandler = handler;
        this.logger.info('Agent handler configured');
    }

    /**
     * Register a messaging platform adapter
     */
    registerAdapter(platform, adapter) {
        this.adapters.set(platform, adapter);

        // Listen for incoming messages from this adapter
        adapter.on('message', async (message) => {
            await this.handleIncomingMessage(platform, message);
        });

        adapter.on('error', (error) => {
            this.logger.error(`[${platform}] Adapter error:`, error);
            this.emit('adapterError', { platform, error });
        });

        this.logger.info(`Registered messaging adapter: ${platform}`);
    }

    /**
     * Add allowed user (for security - whitelist mode)
     */
    addAllowedUser(userId) {
        this.allowedUsers.add(userId.toString());
    }

    /**
     * Remove allowed user
     */
    removeAllowedUser(userId) {
        this.allowedUsers.delete(userId.toString());
    }

    /**
     * Check if user is allowed
     */
    isUserAllowed(userId) {
        // If no whitelist configured, allow all
        if (this.allowedUsers.size === 0) {
            return true;
        }
        return this.allowedUsers.has(userId.toString());
    }

    /**
     * Handle incoming message from any platform
     */
    async handleIncomingMessage(platform, message) {
        const { userId, text, metadata = {} } = message;
        const sessionKey = `${platform}:${userId}`;

        // Security check
        if (!this.isUserAllowed(userId)) {
            this.logger.warn(`Unauthorized message from ${sessionKey}`);
            const adapter = this.adapters.get(platform);
            if (adapter) {
                await adapter.sendMessage(userId,
                    'Sorry, you are not authorized to use this bot. Contact the administrator.'
                );
            }
            return;
        }

        // Get or create user session
        let session = this.userSessions.get(sessionKey);
        if (!session) {
            session = this.createSession(platform, userId, metadata);
            this.userSessions.set(sessionKey, session);
        }

        session.lastActive = Date.now();
        session.messageCount++;

        // Add user message to history
        session.history.push({
            role: 'user',
            content: text,
            timestamp: Date.now()
        });

        // Trim history if too long
        if (session.history.length > this.maxHistoryLength * 2) {
            session.history = session.history.slice(-this.maxHistoryLength);
        }

        this.logger.info(`[${platform}] Message from ${userId}: ${text.substring(0, 50)}...`);

        // Send typing indicator
        const adapter = this.adapters.get(platform);
        if (adapter) {
            await adapter.sendTypingIndicator(userId);
        }

        try {
            // Process with agent
            if (!this.agentHandler) {
                throw new Error('No agent handler configured');
            }

            const response = await this.agentHandler({
                message: text,
                session,
                platform,
                userId,
                metadata
            });

            // Add assistant response to history
            session.history.push({
                role: 'assistant',
                content: response,
                timestamp: Date.now()
            });

            // Send response back to user
            if (adapter) {
                await adapter.sendMessage(userId, response);
            }

            // Emit event for logging/analytics
            this.emit('messageProcessed', {
                platform,
                userId,
                messageLength: text.length,
                responseLength: response.length
            });

        } catch (error) {
            this.logger.error(`Error processing message from ${sessionKey}:`, error);

            // Send error message to user
            if (adapter) {
                await adapter.sendMessage(userId,
                    `Sorry, I encountered an error processing your request. Please try again.\n\nError: ${error.message}`
                );
            }

            this.emit('messageError', { platform, userId, error });
        }

        // Periodically save sessions
        if (Math.random() < 0.1) { // 10% chance
            await this.saveSessions();
        }
    }

    /**
     * Create a new user session
     */
    createSession(platform, userId, metadata) {
        return {
            platform,
            userId,
            metadata,
            history: [],
            context: {},          // Custom context for workflows
            createdAt: Date.now(),
            lastActive: Date.now(),
            messageCount: 0,
            preferences: {}
        };
    }

    /**
     * Get session for a user
     */
    getSession(platform, userId) {
        return this.userSessions.get(`${platform}:${userId}`);
    }

    /**
     * Update session context
     */
    updateSessionContext(platform, userId, context) {
        const session = this.getSession(platform, userId);
        if (session) {
            session.context = { ...session.context, ...context };
        }
    }

    /**
     * Clear session history
     */
    clearSessionHistory(platform, userId) {
        const session = this.getSession(platform, userId);
        if (session) {
            session.history = [];
            this.logger.info(`Cleared history for ${platform}:${userId}`);
        }
    }

    /**
     * Send message to a user (for proactive messaging)
     */
    async sendToUser(platform, userId, message) {
        const adapter = this.adapters.get(platform);
        if (!adapter) {
            throw new Error(`Adapter not found for platform: ${platform}`);
        }
        await adapter.sendMessage(userId, message);
    }

    /**
     * Broadcast message to all users on a platform
     */
    async broadcast(platform, message) {
        const sessions = Array.from(this.userSessions.values())
            .filter(s => s.platform === platform);

        const adapter = this.adapters.get(platform);
        if (!adapter) {
            throw new Error(`Adapter not found for platform: ${platform}`);
        }

        for (const session of sessions) {
            try {
                await adapter.sendMessage(session.userId, message);
            } catch (error) {
                this.logger.error(`Failed to broadcast to ${session.userId}:`, error);
            }
        }
    }

    /**
     * Cleanup expired sessions
     */
    cleanupExpiredSessions() {
        const now = Date.now();
        let cleaned = 0;

        for (const [key, session] of this.userSessions) {
            if (now - session.lastActive > this.sessionTimeout) {
                this.userSessions.delete(key);
                cleaned++;
            }
        }

        if (cleaned > 0) {
            this.logger.info(`Cleaned up ${cleaned} expired sessions`);
        }
    }

    /**
     * Save sessions to file
     */
    async saveSessions() {
        try {
            const data = {};
            for (const [key, session] of this.userSessions) {
                // Don't save full history to file, just metadata
                data[key] = {
                    platform: session.platform,
                    userId: session.userId,
                    metadata: session.metadata,
                    context: session.context,
                    createdAt: session.createdAt,
                    lastActive: session.lastActive,
                    messageCount: session.messageCount,
                    preferences: session.preferences
                };
            }

            await fs.mkdir(path.dirname(this.sessionsFile), { recursive: true });
            await fs.writeFile(this.sessionsFile, JSON.stringify(data, null, 2));
        } catch (error) {
            this.logger.error('Failed to save sessions:', error);
        }
    }

    /**
     * Load sessions from file
     */
    async loadSessions() {
        try {
            const data = JSON.parse(await fs.readFile(this.sessionsFile, 'utf8'));
            for (const [key, session] of Object.entries(data)) {
                // Restore session with empty history
                this.userSessions.set(key, {
                    ...session,
                    history: []
                });
            }
            this.logger.info(`Loaded ${Object.keys(data).length} sessions`);
        } catch (error) {
            if (error.code !== 'ENOENT') {
                this.logger.error('Failed to load sessions:', error);
            }
        }
    }

    /**
     * Get gateway stats
     */
    getStats() {
        const stats = {
            activeSessions: this.userSessions.size,
            adapters: Array.from(this.adapters.keys()),
            allowedUsers: this.allowedUsers.size,
            byPlatform: {}
        };

        for (const [key, session] of this.userSessions) {
            const platform = session.platform;
            if (!stats.byPlatform[platform]) {
                stats.byPlatform[platform] = { sessions: 0, messages: 0 };
            }
            stats.byPlatform[platform].sessions++;
            stats.byPlatform[platform].messages += session.messageCount;
        }

        return stats;
    }

    /**
     * Shutdown gateway
     */
    async shutdown() {
        if (this.cleanupInterval) {
            clearInterval(this.cleanupInterval);
        }

        await this.saveSessions();

        for (const [platform, adapter] of this.adapters) {
            if (adapter.shutdown) {
                await adapter.shutdown();
            }
        }

        this.logger.info('Message gateway shutdown complete');
    }
}

module.exports = MessageGateway;
