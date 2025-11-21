/**
 * MCP (Model Context Protocol) Client
 *
 * Manages connections to MCP servers and provides unified tool access
 * Optimized for M4 Mac Mini (16GB) running 24/7
 */

const { spawn } = require('child_process');
const { EventEmitter } = require('events');
const readline = require('readline');

class MCPClient extends EventEmitter {
    constructor(logger) {
        super();
        this.servers = new Map();      // name → { process, tools, resources, status }
        this.allTools = new Map();     // fullToolName → { server, schema, handler }
        this.logger = logger || console;
        this.reconnectAttempts = new Map();
        this.maxReconnectAttempts = 3;
    }

    /**
     * Connect to an MCP server via stdio
     */
    async connect(name, config) {
        if (this.servers.has(name)) {
            this.logger.warn(`MCP server ${name} already connected, disconnecting first`);
            await this.disconnect(name);
        }

        this.logger.info(`Connecting to MCP server: ${name}`);

        try {
            // Spawn the MCP server process
            const serverProcess = spawn(config.command, config.args || [], {
                env: { ...process.env, ...config.env },
                stdio: ['pipe', 'pipe', 'pipe']
            });

            const server = {
                name,
                process: serverProcess,
                config,
                tools: [],
                resources: [],
                status: 'connecting',
                requestId: 0,
                pendingRequests: new Map(),
                buffer: ''
            };

            // Set up JSON-RPC communication
            this.setupServerCommunication(server);

            this.servers.set(name, server);

            // Initialize the connection
            await this.initializeServer(server);

            // Discover tools
            const tools = await this.listServerTools(server);
            server.tools = tools;
            server.status = 'connected';

            // Index tools for quick lookup
            tools.forEach(tool => {
                const fullName = `${name}:${tool.name}`;
                this.allTools.set(fullName, {
                    server: name,
                    name: tool.name,
                    description: tool.description,
                    inputSchema: tool.inputSchema
                });
            });

            this.logger.info(`Connected to MCP server: ${name} with ${tools.length} tools`);
            this.emit('connected', { server: name, tools: tools.length });

            return tools;

        } catch (error) {
            this.logger.error(`Failed to connect to MCP server ${name}:`, error);
            this.servers.delete(name);
            throw error;
        }
    }

    /**
     * Set up JSON-RPC communication with server
     */
    setupServerCommunication(server) {
        const rl = readline.createInterface({
            input: server.process.stdout,
            crlfDelay: Infinity
        });

        rl.on('line', (line) => {
            try {
                const message = JSON.parse(line);
                this.handleServerMessage(server, message);
            } catch (e) {
                // Not JSON, might be logging output
                this.logger.debug(`[${server.name}] ${line}`);
            }
        });

        server.process.stderr.on('data', (data) => {
            this.logger.debug(`[${server.name}] stderr: ${data.toString()}`);
        });

        server.process.on('close', (code) => {
            this.logger.warn(`MCP server ${server.name} closed with code ${code}`);
            server.status = 'disconnected';
            this.emit('disconnected', { server: server.name, code });

            // Auto-reconnect if enabled
            if (server.config.autoReconnect) {
                this.attemptReconnect(server.name, server.config);
            }
        });

        server.process.on('error', (error) => {
            this.logger.error(`MCP server ${server.name} error:`, error);
            server.status = 'error';
            this.emit('error', { server: server.name, error });
        });
    }

    /**
     * Handle incoming JSON-RPC message from server
     */
    handleServerMessage(server, message) {
        if (message.id !== undefined && server.pendingRequests.has(message.id)) {
            // This is a response to a request
            const { resolve, reject } = server.pendingRequests.get(message.id);
            server.pendingRequests.delete(message.id);

            if (message.error) {
                reject(new Error(message.error.message || JSON.stringify(message.error)));
            } else {
                resolve(message.result);
            }
        } else if (message.method) {
            // This is a notification or request from server
            this.handleServerNotification(server, message);
        }
    }

    /**
     * Handle notifications from server
     */
    handleServerNotification(server, message) {
        this.logger.debug(`[${server.name}] Notification: ${message.method}`);
        this.emit('notification', { server: server.name, ...message });
    }

    /**
     * Send JSON-RPC request to server
     */
    async sendRequest(server, method, params = {}) {
        return new Promise((resolve, reject) => {
            const id = ++server.requestId;
            const request = {
                jsonrpc: '2.0',
                id,
                method,
                params
            };

            server.pendingRequests.set(id, { resolve, reject });

            // Set timeout
            const timeout = setTimeout(() => {
                server.pendingRequests.delete(id);
                reject(new Error(`Request timeout: ${method}`));
            }, 30000);

            server.pendingRequests.set(id, {
                resolve: (result) => {
                    clearTimeout(timeout);
                    resolve(result);
                },
                reject: (error) => {
                    clearTimeout(timeout);
                    reject(error);
                }
            });

            server.process.stdin.write(JSON.stringify(request) + '\n');
        });
    }

    /**
     * Initialize MCP server connection
     */
    async initializeServer(server) {
        const result = await this.sendRequest(server, 'initialize', {
            protocolVersion: '2024-11-05',
            capabilities: {
                tools: {},
                resources: {},
                prompts: {}
            },
            clientInfo: {
                name: 'ollama-agent',
                version: '1.0.0'
            }
        });

        // Send initialized notification
        server.process.stdin.write(JSON.stringify({
            jsonrpc: '2.0',
            method: 'notifications/initialized'
        }) + '\n');

        return result;
    }

    /**
     * List tools from server
     */
    async listServerTools(server) {
        const result = await this.sendRequest(server, 'tools/list', {});
        return result.tools || [];
    }

    /**
     * Call a tool on an MCP server
     */
    async callTool(serverName, toolName, args = {}) {
        const server = this.servers.get(serverName);
        if (!server) {
            throw new Error(`MCP server not found: ${serverName}`);
        }
        if (server.status !== 'connected') {
            throw new Error(`MCP server ${serverName} is not connected (status: ${server.status})`);
        }

        this.logger.info(`Calling MCP tool: ${serverName}:${toolName}`);

        const result = await this.sendRequest(server, 'tools/call', {
            name: toolName,
            arguments: args
        });

        return result;
    }

    /**
     * Call tool by full name (server:toolName)
     */
    async call(fullToolName, args = {}) {
        const [serverName, toolName] = fullToolName.split(':');
        return this.callTool(serverName, toolName, args);
    }

    /**
     * Get all available tools across all servers
     */
    getAllTools() {
        return Array.from(this.allTools.entries()).map(([fullName, tool]) => ({
            fullName,
            ...tool
        }));
    }

    /**
     * Get tools for a specific server
     */
    getServerTools(serverName) {
        const server = this.servers.get(serverName);
        return server ? server.tools : [];
    }

    /**
     * Check if a server is connected
     */
    isConnected(serverName) {
        const server = this.servers.get(serverName);
        return server && server.status === 'connected';
    }

    /**
     * Get server status
     */
    getStatus() {
        const status = {};
        for (const [name, server] of this.servers) {
            status[name] = {
                status: server.status,
                tools: server.tools.length,
                uptime: server.connectedAt ? Date.now() - server.connectedAt : 0
            };
        }
        return status;
    }

    /**
     * Attempt to reconnect to a server
     */
    async attemptReconnect(name, config) {
        const attempts = this.reconnectAttempts.get(name) || 0;

        if (attempts >= this.maxReconnectAttempts) {
            this.logger.error(`Max reconnect attempts reached for ${name}`);
            this.reconnectAttempts.delete(name);
            return;
        }

        this.reconnectAttempts.set(name, attempts + 1);
        const delay = Math.pow(2, attempts) * 1000; // Exponential backoff

        this.logger.info(`Reconnecting to ${name} in ${delay}ms (attempt ${attempts + 1})`);

        setTimeout(async () => {
            try {
                await this.connect(name, config);
                this.reconnectAttempts.delete(name);
            } catch (error) {
                this.logger.error(`Reconnect failed for ${name}:`, error);
            }
        }, delay);
    }

    /**
     * Disconnect from a specific server
     */
    async disconnect(name) {
        const server = this.servers.get(name);
        if (!server) return;

        try {
            server.process.kill();
        } catch (e) {
            // Process may already be dead
        }

        // Remove tools from index
        for (const [fullName, tool] of this.allTools) {
            if (tool.server === name) {
                this.allTools.delete(fullName);
            }
        }

        this.servers.delete(name);
        this.logger.info(`Disconnected from MCP server: ${name}`);
    }

    /**
     * Disconnect all servers
     */
    async disconnectAll() {
        const names = Array.from(this.servers.keys());
        for (const name of names) {
            await this.disconnect(name);
        }
        this.logger.info('All MCP servers disconnected');
    }

    /**
     * Graceful shutdown
     */
    async shutdown() {
        this.logger.info('Shutting down MCP client...');
        await this.disconnectAll();
    }
}

module.exports = MCPClient;
