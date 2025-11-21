/**
 * MCP Tool Adapter
 *
 * Adapts MCP tools to work seamlessly with the existing ToolSystem
 * Provides unified interface for both native and MCP tools
 */

class MCPToolAdapter {
    constructor(toolSystem, mcpClient, logger) {
        this.toolSystem = toolSystem;
        this.mcpClient = mcpClient;
        this.logger = logger || console;
        this.registeredTools = new Set();
    }

    /**
     * Register all MCP tools with the ToolSystem
     */
    registerAllMCPTools() {
        const mcpTools = this.mcpClient.getAllTools();

        for (const tool of mcpTools) {
            this.registerMCPTool(tool);
        }

        this.logger.info(`Registered ${mcpTools.length} MCP tools with ToolSystem`);
        return mcpTools.length;
    }

    /**
     * Register a single MCP tool with the ToolSystem
     */
    registerMCPTool(tool) {
        // Create a unique name: mcp_servername_toolname
        const toolName = `mcp_${tool.server}_${tool.name}`.replace(/-/g, '_');

        // Skip if already registered
        if (this.registeredTools.has(toolName)) {
            return;
        }

        // Create the tool handler
        const handler = async (params) => {
            try {
                this.logger.info(`Executing MCP tool: ${tool.server}:${tool.name}`, params);

                const result = await this.mcpClient.callTool(
                    tool.server,
                    tool.name,
                    params
                );

                // Parse MCP result format
                const content = this.parseMCPResult(result);

                return {
                    success: true,
                    result: content,
                    server: tool.server,
                    tool: tool.name
                };
            } catch (error) {
                this.logger.error(`MCP tool error: ${tool.server}:${tool.name}`, error);
                return {
                    success: false,
                    error: error.message,
                    server: tool.server,
                    tool: tool.name
                };
            }
        };

        // Convert MCP schema to ToolSystem parameter format
        const parameters = this.convertSchema(tool.inputSchema);

        // Register with ToolSystem
        this.toolSystem.registerTool(
            toolName,
            `[MCP:${tool.server}] ${tool.description || tool.name}`,
            handler,
            parameters
        );

        this.registeredTools.add(toolName);
        this.logger.debug(`Registered MCP tool: ${toolName}`);
    }

    /**
     * Parse MCP result into usable format
     */
    parseMCPResult(result) {
        if (!result) return null;

        // MCP results come as content array
        if (result.content && Array.isArray(result.content)) {
            return result.content.map(item => {
                if (item.type === 'text') {
                    return item.text;
                } else if (item.type === 'image') {
                    return { type: 'image', data: item.data, mimeType: item.mimeType };
                } else if (item.type === 'resource') {
                    return { type: 'resource', uri: item.uri, text: item.text };
                }
                return item;
            });
        }

        return result;
    }

    /**
     * Convert MCP JSON Schema to ToolSystem parameter format
     */
    convertSchema(inputSchema) {
        if (!inputSchema || !inputSchema.properties) {
            return [];
        }

        const required = inputSchema.required || [];

        return Object.entries(inputSchema.properties).map(([name, prop]) => ({
            name,
            type: prop.type || 'string',
            required: required.includes(name),
            description: prop.description || '',
            enum: prop.enum || null,
            default: prop.default
        }));
    }

    /**
     * Unregister all MCP tools
     */
    unregisterAllMCPTools() {
        for (const toolName of this.registeredTools) {
            this.toolSystem.tools.delete(toolName);
        }
        this.registeredTools.clear();
        this.logger.info('Unregistered all MCP tools');
    }

    /**
     * Refresh tools (re-register after server reconnect)
     */
    async refreshTools() {
        this.unregisterAllMCPTools();
        return this.registerAllMCPTools();
    }

    /**
     * Get all registered MCP tool names
     */
    getRegisteredTools() {
        return Array.from(this.registeredTools);
    }

    /**
     * Generate documentation for all MCP tools
     */
    generateMCPToolDocs() {
        const tools = this.mcpClient.getAllTools();
        let docs = '\n=== MCP TOOLS ===\n\n';

        // Group by server
        const byServer = {};
        tools.forEach(tool => {
            if (!byServer[tool.server]) {
                byServer[tool.server] = [];
            }
            byServer[tool.server].push(tool);
        });

        for (const [server, serverTools] of Object.entries(byServer)) {
            docs += `--- ${server} ---\n`;
            serverTools.forEach(tool => {
                const toolName = `mcp_${server}_${tool.name}`.replace(/-/g, '_');
                docs += `\n[TOOL: ${toolName}]\n`;
                docs += `Description: ${tool.description || 'No description'}\n`;

                const params = this.convertSchema(tool.inputSchema);
                if (params.length > 0) {
                    docs += `Parameters:\n`;
                    params.forEach(param => {
                        const req = param.required ? '[required]' : '[optional]';
                        docs += `  - ${param.name} (${param.type}) ${req}`;
                        if (param.description) {
                            docs += `: ${param.description}`;
                        }
                        docs += '\n';
                    });
                }
            });
            docs += '\n';
        }

        return docs;
    }
}

module.exports = MCPToolAdapter;
