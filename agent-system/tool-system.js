/**
 * Tool/Function Calling System
 * 
 * Provides structured tool calling for agents to perform complex operations
 */

const fs = require('fs').promises;
const path = require('path');
const { exec } = require('child_process');
const { promisify } = require('util');
const execAsync = promisify(exec);
const logger = require('winston').default || console;

class ToolSystem {
    constructor() {
        this.tools = new Map();
        this.registerDefaultTools();
    }

    /**
     * Register a tool
     */
    registerTool(name, description, handler, parameters = []) {
        this.tools.set(name, {
            name,
            description,
            handler,
            parameters
        });
    }

    /**
     * Register default tools
     */
    registerDefaultTools() {
        // File operations
        this.registerTool(
            'read_file',
            'Read contents of a file',
            async (params) => {
                const { filepath } = params;
                const safePath = this.sanitizePath(filepath);
                try {
                    const content = await fs.readFile(safePath, 'utf8');
                    return { success: true, content, size: content.length };
                } catch (error) {
                    return { success: false, error: error.message };
                }
            },
            [{ name: 'filepath', type: 'string', required: true }]
        );

        this.registerTool(
            'write_file',
            'Write content to a file',
            async (params) => {
                const { filepath, content } = params;
                const safePath = this.sanitizePath(filepath);
                try {
                    await fs.mkdir(path.dirname(safePath), { recursive: true });
                    await fs.writeFile(safePath, content, 'utf8');
                    return { success: true, message: `File written: ${safePath}` };
                } catch (error) {
                    return { success: false, error: error.message };
                }
            },
            [
                { name: 'filepath', type: 'string', required: true },
                { name: 'content', type: 'string', required: true }
            ]
        );

        this.registerTool(
            'list_directory',
            'List files and directories in a path',
            async (params) => {
                const { dirpath } = params;
                const safePath = this.sanitizePath(dirpath);
                try {
                    const entries = await fs.readdir(safePath, { withFileTypes: true });
                    const result = entries.map(entry => ({
                        name: entry.name,
                        type: entry.isDirectory() ? 'directory' : 'file',
                        path: path.join(safePath, entry.name)
                    }));
                    return { success: true, entries: result };
                } catch (error) {
                    return { success: false, error: error.message };
                }
            },
            [{ name: 'dirpath', type: 'string', required: true }]
        );

        // Code execution
        this.registerTool(
            'execute_code',
            'Execute code in a safe environment (Python, JavaScript, Shell)',
            async (params) => {
                const { code, language = 'python', timeout = 30000 } = params;
                return await this.executeCodeSafely(code, language, timeout);
            },
            [
                { name: 'code', type: 'string', required: true },
                { name: 'language', type: 'string', required: false },
                { name: 'timeout', type: 'number', required: false }
            ]
        );

        // System operations
        this.registerTool(
            'run_command',
            'Run a system command (with restrictions)',
            async (params) => {
                const { command } = params;
                // Security: only allow safe commands
                if (!this.isSafeCommand(command)) {
                    return { success: false, error: 'Command not allowed for security reasons' };
                }
                try {
                    const { stdout, stderr } = await execAsync(command, { timeout: 10000 });
                    return { success: true, stdout, stderr };
                } catch (error) {
                    return { success: false, error: error.message, stderr: error.stderr };
                }
            },
            [{ name: 'command', type: 'string', required: true }]
        );

        // Data processing
        this.registerTool(
            'process_json',
            'Parse and process JSON data',
            async (params) => {
                const { json_string, operation } = params;
                try {
                    const data = JSON.parse(json_string);
                    let result;
                    switch (operation) {
                        case 'keys':
                            result = Object.keys(data);
                            break;
                        case 'values':
                            result = Object.values(data);
                            break;
                        case 'get':
                            result = data[params.key];
                            break;
                        default:
                            result = data;
                    }
                    return { success: true, result };
                } catch (error) {
                    return { success: false, error: error.message };
                }
            },
            [
                { name: 'json_string', type: 'string', required: true },
                { name: 'operation', type: 'string', required: false },
                { name: 'key', type: 'string', required: false }
            ]
        );

        // HTTP requests
        this.registerTool(
            'http_request',
            'Make HTTP request to an API',
            async (params) => {
                const axios = require('axios');
                const { url, method = 'GET', headers = {}, data = null } = params;
                try {
                    const response = await axios({
                        url,
                        method,
                        headers,
                        data,
                        timeout: 10000
                    });
                    return {
                        success: true,
                        status: response.status,
                        data: response.data,
                        headers: response.headers
                    };
                } catch (error) {
                    return {
                        success: false,
                        error: error.message,
                        status: error.response?.status
                    };
                }
            },
            [
                { name: 'url', type: 'string', required: true },
                { name: 'method', type: 'string', required: false },
                { name: 'headers', type: 'object', required: false },
                { name: 'data', type: 'object', required: false }
            ]
        );
    }

    /**
     * Execute code safely
     */
    async executeCodeSafely(code, language, timeout) {
        const tmp = require('tmp-promise');
        
        try {
            let file, command;
            
            switch (language.toLowerCase()) {
                case 'python':
                case 'py':
                    file = await tmp.file({ postfix: '.py' });
                    await fs.writeFile(file.path, code);
                    command = `python3 ${file.path}`;
                    break;
                    
                case 'javascript':
                case 'js':
                case 'node':
                    file = await tmp.file({ postfix: '.js' });
                    await fs.writeFile(file.path, code);
                    command = `node ${file.path}`;
                    break;
                    
                case 'shell':
                case 'bash':
                case 'sh':
                    file = await tmp.file({ postfix: '.sh' });
                    await fs.writeFile(file.path, `#!/bin/bash\n${code}`);
                    await fs.chmod(file.path, 0o755);
                    command = `bash ${file.path}`;
                    break;
                    
                default:
                    return { success: false, error: `Unsupported language: ${language}` };
            }

            const { stdout, stderr } = await execAsync(command, {
                timeout,
                maxBuffer: 1024 * 1024 // 1MB
            });

            // Cleanup
            await file.cleanup();

            return {
                success: true,
                stdout,
                stderr: stderr || '',
                language
            };
        } catch (error) {
            return {
                success: false,
                error: error.message,
                stderr: error.stderr || '',
                timeout: error.killed
            };
        }
    }

    /**
     * Check if command is safe to execute
     */
    isSafeCommand(command) {
        const unsafePatterns = [
            /rm\s+-rf/,
            /format\s+/,
            /del\s+/,
            /mkfs/,
            /dd\s+if=/,
            /shutdown/,
            /reboot/,
            />\s*\/dev/,
            /curl.*-X\s+(DELETE|PUT)/,
            /wget.*--delete/,
        ];

        const lowerCommand = command.toLowerCase();
        return !unsafePatterns.some(pattern => pattern.test(lowerCommand));
    }

    /**
     * Sanitize file path to prevent directory traversal
     */
    sanitizePath(filepath) {
        // Resolve to absolute path
        const resolved = path.resolve(filepath);
        
        // Restrict to current working directory or data directory
        const cwd = process.cwd();
        const dataDir = path.join(cwd, 'data');
        
        if (!resolved.startsWith(cwd) && !resolved.startsWith(dataDir)) {
            throw new Error('Path outside allowed directory');
        }
        
        return resolved;
    }

    /**
     * Parse tool calls from agent response
     */
    parseToolCalls(response) {
        const toolCallPattern = /\[TOOL:\s*(\w+)\s*(?:\{([^\}]+)\})?\s*\]/g;
        const calls = [];
        let match;

        while ((match = toolCallPattern.exec(response)) !== null) {
            const toolName = match[1];
            const paramsStr = match[2] || '{}';
            
            let params = {};
            try {
                params = JSON.parse(paramsStr);
            } catch (e) {
                // Try to parse as key=value pairs
                paramsStr.split(',').forEach(pair => {
                    const [key, value] = pair.split('=').map(s => s.trim());
                    if (key && value) {
                        params[key] = value;
                    }
                });
            }

            calls.push({ toolName, params });
        }

        return calls;
    }

    /**
     * Execute tool calls from agent response
     */
    async executeToolCalls(response) {
        const calls = this.parseToolCalls(response);
        const results = [];

        for (const call of calls) {
            const tool = this.tools.get(call.toolName);
            if (!tool) {
                results.push({
                    toolName: call.toolName,
                    success: false,
                    error: `Tool not found: ${call.toolName}`
                });
                continue;
            }

            try {
                logger.info(`Executing tool: ${call.toolName}`, call.params);
                const result = await tool.handler(call.params);
                results.push({
                    toolName: call.toolName,
                    ...result
                });
            } catch (error) {
                logger.error(`Tool execution failed: ${call.toolName}`, error);
                results.push({
                    toolName: call.toolName,
                    success: false,
                    error: error.message
                });
            }
        }

        return results;
    }

    /**
     * Replace tool calls in response with results
     */
    replaceToolCalls(response, results) {
        let enhancedResponse = response;
        let resultIndex = 0;

        const toolCallPattern = /\[TOOL:\s*(\w+)\s*(?:\{[^\}]+\})?\s*\]/g;
        
        enhancedResponse = enhancedResponse.replace(toolCallPattern, () => {
            if (resultIndex < results.length) {
                const result = results[resultIndex++];
                const resultStr = JSON.stringify(result, null, 2);
                return `\n\n[TOOL RESULT: ${result.toolName}]\n${resultStr}\n\n`;
            }
            return '[TOOL: execution pending]';
        });

        return enhancedResponse;
    }

    /**
     * Get list of available tools
     */
    getAvailableTools() {
        return Array.from(this.tools.values()).map(tool => ({
            name: tool.name,
            description: tool.description,
            parameters: tool.parameters
        }));
    }

    /**
     * Generate tool documentation for agents
     */
    generateToolDocs() {
        const tools = this.getAvailableTools();
        let docs = 'AVAILABLE TOOLS:\n\n';
        
        tools.forEach(tool => {
            docs += `[TOOL: ${tool.name}]\n`;
            docs += `Description: ${tool.description}\n`;
            docs += `Parameters:\n`;
            tool.parameters.forEach(param => {
                docs += `  - ${param.name} (${param.type})${param.required ? ' [required]' : ' [optional]'}\n`;
            });
            docs += `\nUsage: [TOOL: ${tool.name} {${tool.parameters.map(p => `"${p.name}": "value"`).join(', ')}}]\n\n`;
        });

        return docs;
    }
}

module.exports = ToolSystem;

