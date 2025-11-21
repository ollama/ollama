/**
 * MCP Integration Module
 *
 * Main entry point for MCP functionality
 * Handles initialization, configuration, and coordination
 */

const MCPClient = require('./mcp-client');
const MCPToolAdapter = require('./mcp-tool-adapter');
const BrowserAutomation = require('./browser-automation');
const path = require('path');
const fs = require('fs').promises;

class MCPIntegration {
    constructor(options = {}) {
        this.logger = options.logger || console;
        this.toolSystem = options.toolSystem;
        this.configPath = options.configPath || path.join(__dirname, 'mcp-config.json');

        this.mcpClient = new MCPClient(this.logger);
        this.mcpAdapter = null;
        this.browserAutomation = new BrowserAutomation(this.logger);
        this.config = null;
        this.initialized = false;
    }

    /**
     * Initialize MCP integration
     */
    async initialize() {
        if (this.initialized) {
            this.logger.warn('MCP integration already initialized');
            return;
        }

        this.logger.info('Initializing MCP integration...');

        try {
            // Load configuration
            await this.loadConfig();

            // Initialize browser automation
            await this.browserAutomation.initialize();
            this.browserAutomation.registerTools(this.toolSystem);
            this.logger.info('Browser automation tools registered');

            // Connect to MCP servers
            await this.connectServers();

            // Create MCP tool adapter
            this.mcpAdapter = new MCPToolAdapter(this.toolSystem, this.mcpClient, this.logger);

            // Register MCP tools with tool system
            this.mcpAdapter.registerAllMCPTools();

            // Register email tool (using nodemailer)
            this.registerEmailTool();

            this.initialized = true;
            this.logger.info('MCP integration initialized successfully');

            return this.getStatus();

        } catch (error) {
            this.logger.error('Failed to initialize MCP integration:', error);
            throw error;
        }
    }

    /**
     * Load MCP configuration
     */
    async loadConfig() {
        try {
            const configData = await fs.readFile(this.configPath, 'utf8');
            this.config = JSON.parse(configData);

            // Replace environment variables in config
            this.config = this.replaceEnvVars(this.config);

            this.logger.info('MCP configuration loaded');
        } catch (error) {
            this.logger.warn('Failed to load MCP config, using defaults:', error.message);
            this.config = {
                servers: {},
                settings: {
                    autoStartEnabled: true,
                    connectionTimeout: 30000
                }
            };
        }
    }

    /**
     * Replace ${VAR} placeholders with environment variables
     */
    replaceEnvVars(obj) {
        if (typeof obj === 'string') {
            return obj.replace(/\$\{(\w+)\}/g, (match, varName) => {
                return process.env[varName] || match;
            });
        } else if (Array.isArray(obj)) {
            return obj.map(item => this.replaceEnvVars(item));
        } else if (typeof obj === 'object' && obj !== null) {
            const result = {};
            for (const [key, value] of Object.entries(obj)) {
                result[key] = this.replaceEnvVars(value);
            }
            return result;
        }
        return obj;
    }

    /**
     * Connect to configured MCP servers
     */
    async connectServers() {
        if (!this.config.servers) return;

        const enabledServers = Object.entries(this.config.servers)
            .filter(([name, config]) => config.enabled);

        this.logger.info(`Connecting to ${enabledServers.length} MCP servers...`);

        for (const [name, serverConfig] of enabledServers) {
            try {
                await this.mcpClient.connect(name, serverConfig);
            } catch (error) {
                this.logger.warn(`Failed to connect to MCP server ${name}:`, error.message);
                // Continue with other servers
            }
        }
    }

    /**
     * Register email tool using nodemailer
     */
    registerEmailTool() {
        const nodemailer = require('nodemailer');
        const self = this;

        // Create transporter (configure via env vars)
        let transporter = null;

        if (process.env.SMTP_HOST) {
            transporter = nodemailer.createTransport({
                host: process.env.SMTP_HOST,
                port: parseInt(process.env.SMTP_PORT) || 587,
                secure: process.env.SMTP_SECURE === 'true',
                auth: {
                    user: process.env.SMTP_USER,
                    pass: process.env.SMTP_PASS
                }
            });
        }

        this.toolSystem.registerTool(
            'send_email',
            'Send an email using SMTP',
            async (params) => {
                if (!transporter) {
                    return {
                        success: false,
                        error: 'Email not configured. Set SMTP_HOST, SMTP_USER, and SMTP_PASS environment variables.'
                    };
                }

                try {
                    const result = await transporter.sendMail({
                        from: process.env.SMTP_FROM || process.env.SMTP_USER,
                        to: params.to,
                        subject: params.subject,
                        text: params.body,
                        html: params.html
                    });

                    self.logger.info(`Email sent to ${params.to}: ${result.messageId}`);

                    return {
                        success: true,
                        messageId: result.messageId,
                        to: params.to,
                        subject: params.subject
                    };
                } catch (error) {
                    return {
                        success: false,
                        error: error.message
                    };
                }
            },
            [
                { name: 'to', type: 'string', required: true, description: 'Recipient email' },
                { name: 'subject', type: 'string', required: true, description: 'Email subject' },
                { name: 'body', type: 'string', required: true, description: 'Email body (plain text)' },
                { name: 'html', type: 'string', required: false, description: 'HTML body (optional)' }
            ]
        );

        this.logger.info('Email tool registered');
    }

    /**
     * Get all available tools (native + MCP + browser)
     */
    getAllTools() {
        return this.toolSystem.getAvailableTools();
    }

    /**
     * Generate comprehensive tool documentation for the agent
     */
    generateToolDocumentation() {
        let docs = this.toolSystem.generateToolDocs();

        if (this.mcpAdapter) {
            docs += this.mcpAdapter.generateMCPToolDocs();
        }

        return docs;
    }

    /**
     * Get integration status
     */
    getStatus() {
        return {
            initialized: this.initialized,
            mcpServers: this.mcpClient.getStatus(),
            browserAutomation: {
                running: this.browserAutomation.browser !== null,
                activeSessions: this.browserAutomation.pages.size
            },
            totalTools: this.toolSystem.tools.size
        };
    }

    /**
     * Reconnect a specific MCP server
     */
    async reconnectServer(serverName) {
        const serverConfig = this.config.servers[serverName];
        if (!serverConfig) {
            throw new Error(`Server ${serverName} not found in configuration`);
        }

        await this.mcpClient.disconnect(serverName);
        await this.mcpClient.connect(serverName, serverConfig);

        // Re-register tools
        if (this.mcpAdapter) {
            this.mcpAdapter.refreshTools();
        }
    }

    /**
     * Enable/disable an MCP server
     */
    async setServerEnabled(serverName, enabled) {
        if (!this.config.servers[serverName]) {
            throw new Error(`Server ${serverName} not found`);
        }

        this.config.servers[serverName].enabled = enabled;

        if (enabled) {
            await this.mcpClient.connect(serverName, this.config.servers[serverName]);
        } else {
            await this.mcpClient.disconnect(serverName);
        }

        // Refresh tools
        if (this.mcpAdapter) {
            this.mcpAdapter.refreshTools();
        }
    }

    /**
     * Shutdown all MCP connections and browser
     */
    async shutdown() {
        this.logger.info('Shutting down MCP integration...');

        await this.mcpClient.disconnectAll();
        await this.browserAutomation.shutdown();

        this.initialized = false;
        this.logger.info('MCP integration shutdown complete');
    }
}

module.exports = MCPIntegration;
