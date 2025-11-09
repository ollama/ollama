/**
 * Enhanced Tool System - Advanced Capabilities
 * 
 * Extends the base tool system with advanced features for complex tasks
 */

const fs = require('fs').promises;
const path = require('path');
const { exec, spawn } = require('child_process');
const { promisify } = require('util');
const execAsync = promisify(exec);
const logger = require('winston').default || console;
const tmp = require('tmp-promise');

class EnhancedTools {
    constructor(baseToolSystem) {
        this.tools = baseToolSystem.tools; // Access the tools Map directly
        this.baseSystem = baseToolSystem; // Keep reference to base system
        this.activeSessions = new Map(); // For REPL sessions
        this.registerEnhancedTools();
    }

    /**
     * Register enhanced tools
     */
    registerEnhancedTools() {
        // Enhanced code execution with more languages
        this.baseSystem.registerTool(
            'execute_code_advanced',
            'Execute code in multiple languages with advanced features',
            async (params) => {
                const { code, language = 'python', timeout = 30000, packages = [], input = null } = params;
                return await this.executeCodeAdvanced(code, language, timeout, packages, input);
            },
            [
                { name: 'code', type: 'string', required: true },
                { name: 'language', type: 'string', required: false },
                { name: 'timeout', type: 'number', required: false },
                { name: 'packages', type: 'array', required: false },
                { name: 'input', type: 'string', required: false }
            ]
        );

        // REPL session management
        this.baseTools.registerTool(
            'repl_session',
            'Start or interact with a REPL session (interactive code execution)',
            async (params) => {
                const { action, sessionId, code, language = 'python' } = params;
                return await this.handleREPLSession(action, sessionId, code, language);
            },
            [
                { name: 'action', type: 'string', required: true }, // start, execute, stop
                { name: 'sessionId', type: 'string', required: false },
                { name: 'code', type: 'string', required: false },
                { name: 'language', type: 'string', required: false }
            ]
        );

        // Database operations
        this.baseTools.registerTool(
            'database_query',
            'Execute SQL queries on databases',
            async (params) => {
                const { database, query, type = 'sqlite' } = params;
                return await this.executeDatabaseQuery(database, query, type);
            },
            [
                { name: 'database', type: 'string', required: true },
                { name: 'query', type: 'string', required: true },
                { name: 'type', type: 'string', required: false } // sqlite, postgres, mysql
            ]
        );

        // Git operations
        this.baseTools.registerTool(
            'git_operation',
            'Perform Git operations (commit, push, pull, branch, etc.)',
            async (params) => {
                const { operation, args = [] } = params;
                return await this.executeGitOperation(operation, args);
            },
            [
                { name: 'operation', type: 'string', required: true },
                { name: 'args', type: 'array', required: false }
            ]
        );

        // Package management
        this.baseTools.registerTool(
            'package_manager',
            'Install or manage packages using package managers',
            async (params) => {
                const { manager, action, packages = [] } = params; // npm, pip, pip3, cargo, go
                return await this.managePackages(manager, action, packages);
            },
            [
                { name: 'manager', type: 'string', required: true },
                { name: 'action', type: 'string', required: true }, // install, uninstall, list
                { name: 'packages', type: 'array', required: false }
            ]
        );

        // Run tests
        this.baseTools.registerTool(
            'run_tests',
            'Run test suites (Jest, pytest, unittest, etc.)',
            async (params) => {
                const { framework, path, options = [] } = params;
                return await this.runTests(framework, path, options);
            },
            [
                { name: 'framework', type: 'string', required: true },
                { name: 'path', type: 'string', required: false },
                { name: 'options', type: 'array', required: false }
            ]
        );

        // Code analysis
        this.baseTools.registerTool(
            'analyze_code',
            'Analyze code for quality, complexity, and issues',
            async (params) => {
                const { filepath, language, analysis = 'all' } = params;
                return await this.analyzeCode(filepath, language, analysis);
            },
            [
                { name: 'filepath', type: 'string', required: true },
                { name: 'language', type: 'string', required: false },
                { name: 'analysis', type: 'string', required: false } // complexity, quality, security, all
            ]
        );

        // Start development server
        this.baseTools.registerTool(
            'start_server',
            'Start a development server (Node, Python, etc.)',
            async (params) => {
                const { type, port, script, options = [] } = params;
                return await this.startServer(type, port, script, options);
            },
            [
                { name: 'type', type: 'string', required: true }, // node, python, go
                { name: 'port', type: 'number', required: false },
                { name: 'script', type: 'string', required: false },
                { name: 'options', type: 'array', required: false }
            ]
        );

        // Data visualization
        this.baseTools.registerTool(
            'create_visualization',
            'Create data visualizations (charts, graphs)',
            async (params) => {
                const { data, type, options = {} } = params;
                return await this.createVisualization(data, type, options);
            },
            [
                { name: 'data', type: 'string', required: true }, // JSON or CSV
                { name: 'type', type: 'string', required: true }, // line, bar, pie, scatter
                { name: 'options', type: 'object', required: false }
            ]
        );

        // File operations with patterns
        this.baseTools.registerTool(
            'file_operations',
            'Advanced file operations (search, replace, batch operations)',
            async (params) => {
                const { operation, pattern, replacement, directory } = params;
                return await this.fileOperations(operation, pattern, replacement, directory);
            },
            [
                { name: 'operation', type: 'string', required: true }, // search, replace, find, batch
                { name: 'pattern', type: 'string', required: true },
                { name: 'replacement', type: 'string', required: false },
                { name: 'directory', type: 'string', required: false }
            ]
        );

        // Process management
        this.baseTools.registerTool(
            'process_manager',
            'Manage system processes (list, kill, monitor)',
            async (params) => {
                const { action, pid, name } = params;
                return await this.manageProcesses(action, pid, name);
            },
            [
                { name: 'action', type: 'string', required: true }, // list, kill, monitor
                { name: 'pid', type: 'number', required: false },
                { name: 'name', type: 'string', required: false }
            ]
        );
    }

    /**
     * Execute code with advanced features
     */
    async executeCodeAdvanced(code, language, timeout, packages, input) {
        try {
            // Install packages if needed
            if (packages && packages.length > 0) {
                await this.installPackagesForLanguage(language, packages);
            }

            let file, command;
            const workDir = await tmp.dir({ unsafeCleanup: true });

            switch (language.toLowerCase()) {
                case 'python':
                case 'py':
                    file = await tmp.file({ dir: workDir.path, postfix: '.py' });
                    await fs.writeFile(file.path, code);
                    command = `cd ${workDir.path} && python3 ${path.basename(file.path)}`;
                    break;

                case 'javascript':
                case 'js':
                case 'node':
                    file = await tmp.file({ dir: workDir.path, postfix: '.js' });
                    await fs.writeFile(file.path, code);
                    command = `cd ${workDir.path} && node ${path.basename(file.path)}`;
                    break;

                case 'r':
                    file = await tmp.file({ dir: workDir.path, postfix: '.R' });
                    await fs.writeFile(file.path, code);
                    command = `cd ${workDir.path} && Rscript ${path.basename(file.path)}`;
                    break;

                case 'go':
                    file = await tmp.file({ dir: workDir.path, postfix: '.go' });
                    await fs.writeFile(file.path, code);
                    command = `cd ${workDir.path} && go run ${path.basename(file.path)}`;
                    break;

                case 'rust':
                    file = await tmp.file({ dir: workDir.path, postfix: '.rs' });
                    await fs.writeFile(file.path, code);
                    command = `cd ${workDir.path} && rustc ${path.basename(file.path)} && ./${path.basename(file.path, '.rs')}`;
                    break;

                default:
                    return { success: false, error: `Unsupported language: ${language}` };
            }

            const { stdout, stderr } = await execAsync(command, {
                timeout,
                maxBuffer: 10 * 1024 * 1024, // 10MB
                input: input || undefined
            });

            // Cleanup
            await file.cleanup();
            await workDir.cleanup();

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
     * Install packages for a language
     */
    async installPackagesForLanguage(language, packages) {
        try {
            let command;
            switch (language.toLowerCase()) {
                case 'python':
                case 'py':
                    command = `pip3 install ${packages.join(' ')}`;
                    break;
                case 'javascript':
                case 'js':
                case 'node':
                    command = `npm install ${packages.join(' ')}`;
                    break;
                case 'r':
                    command = `Rscript -e "install.packages(c('${packages.join("', '")}'), repos='https://cran.rstudio.com/')"`;
                    break;
                default:
                    return { success: false, error: `Package installation not supported for ${language}` };
            }

            await execAsync(command, { timeout: 60000 });
            return { success: true };
        } catch (error) {
            logger.warn(`Package installation warning: ${error.message}`);
            return { success: false, warning: error.message };
        }
    }

    /**
     * Handle REPL sessions
     */
    async handleREPLSession(action, sessionId, code, language) {
        if (action === 'start') {
            const id = sessionId || `repl_${Date.now()}`;
            // Create a persistent session (simplified - in production use proper REPL)
            this.activeSessions.set(id, {
                id,
                language,
                history: [],
                startTime: new Date()
            });
            return { success: true, sessionId: id, message: `REPL session started for ${language}` };
        } else if (action === 'execute') {
            if (!sessionId || !this.activeSessions.has(sessionId)) {
                return { success: false, error: 'Session not found' };
            }
            const session = this.activeSessions.get(sessionId);
            const result = await this.executeCodeAdvanced(code, session.language, 30000, [], null);
            session.history.push({ code, result, timestamp: new Date() });
            return { success: true, result, sessionId };
        } else if (action === 'stop') {
            if (sessionId && this.activeSessions.has(sessionId)) {
                this.activeSessions.delete(sessionId);
                return { success: true, message: 'Session stopped' };
            }
            return { success: false, error: 'Session not found' };
        }
        return { success: false, error: 'Invalid action' };
    }

    /**
     * Execute database queries
     */
    async executeDatabaseQuery(database, query, type) {
        try {
            let command;
            const safePath = this.sanitizePath(database);

            switch (type.toLowerCase()) {
                case 'sqlite':
                    command = `sqlite3 "${safePath}" "${query.replace(/"/g, '\\"')}"`;
                    break;
                default:
                    return { success: false, error: `Database type ${type} not yet supported` };
            }

            const { stdout, stderr } = await execAsync(command, { timeout: 30000 });
            return {
                success: true,
                result: stdout,
                error: stderr || null
            };
        } catch (error) {
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Execute Git operations
     */
    async executeGitOperation(operation, args) {
        const safeOperations = ['status', 'log', 'branch', 'diff', 'show', 'ls-files'];
        const writeOperations = ['add', 'commit', 'push', 'pull', 'checkout', 'merge'];

        if (!safeOperations.includes(operation) && !writeOperations.includes(operation)) {
            return { success: false, error: `Operation ${operation} not allowed` };
        }

        try {
            const command = `git ${operation} ${args.join(' ')}`;
            const { stdout, stderr } = await execAsync(command, { timeout: 30000 });
            return {
                success: true,
                output: stdout,
                error: stderr || null
            };
        } catch (error) {
            return {
                success: false,
                error: error.message,
                stderr: error.stderr
            };
        }
    }

    /**
     * Manage packages
     */
    async managePackages(manager, action, packages) {
        try {
            let command;
            switch (manager.toLowerCase()) {
                case 'npm':
                    command = `npm ${action} ${packages.join(' ')}`;
                    break;
                case 'pip':
                case 'pip3':
                    command = `${manager} ${action} ${packages.join(' ')}`;
                    break;
                case 'cargo':
                    command = `cargo ${action} ${packages.length > 0 ? packages.join(' ') : ''}`;
                    break;
                case 'go':
                    command = `go ${action} ${packages.join(' ')}`;
                    break;
                default:
                    return { success: false, error: `Manager ${manager} not supported` };
            }

            const { stdout, stderr } = await execAsync(command, { timeout: 120000 });
            return {
                success: true,
                output: stdout,
                error: stderr || null
            };
        } catch (error) {
            return {
                success: false,
                error: error.message,
                stderr: error.stderr
            };
        }
    }

    /**
     * Run tests
     */
    async runTests(framework, testPath, options) {
        try {
            let command;
            const safePath = testPath ? this.sanitizePath(testPath) : '.';

            switch (framework.toLowerCase()) {
                case 'jest':
                    command = `cd ${safePath} && npm test -- ${options.join(' ')}`;
                    break;
                case 'pytest':
                    command = `cd ${safePath} && pytest ${options.join(' ')}`;
                    break;
                case 'unittest':
                    command = `cd ${safePath} && python3 -m unittest ${options.join(' ')}`;
                    break;
                case 'mocha':
                    command = `cd ${safePath} && npx mocha ${options.join(' ')}`;
                    break;
                default:
                    return { success: false, error: `Test framework ${framework} not supported` };
            }

            const { stdout, stderr } = await execAsync(command, { timeout: 60000 });
            return {
                success: true,
                output: stdout,
                error: stderr || null,
                passed: !stderr && stdout.includes('passed') || stdout.includes('PASS')
            };
        } catch (error) {
            return {
                success: false,
                error: error.message,
                stderr: error.stderr,
                passed: false
            };
        }
    }

    /**
     * Analyze code
     */
    async analyzeCode(filepath, language, analysis) {
        try {
            const safePath = this.sanitizePath(filepath);
            const content = await fs.readFile(safePath, 'utf8');

            // Basic analysis
            const lines = content.split('\n').length;
            const complexity = this.calculateComplexity(content, language);
            const issues = this.findIssues(content, language);

            return {
                success: true,
                analysis: {
                    lines,
                    complexity,
                    issues,
                    language: language || this.detectLanguage(filepath)
                }
            };
        } catch (error) {
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Calculate code complexity
     */
    calculateComplexity(code, language) {
        // Simplified complexity calculation
        const cyclomatic = (code.match(/if|else|for|while|switch|case/g) || []).length + 1;
        return {
            cyclomatic,
            level: cyclomatic < 10 ? 'low' : cyclomatic < 20 ? 'medium' : 'high'
        };
    }

    /**
     * Find code issues
     */
    findIssues(code, language) {
        const issues = [];
        
        // Check for common issues
        if (code.includes('eval(') || code.includes('exec(')) {
            issues.push({ type: 'security', message: 'Use of eval/exec detected' });
        }
        if (code.includes('TODO') || code.includes('FIXME')) {
            issues.push({ type: 'todo', message: 'TODO/FIXME comments found' });
        }
        if (code.match(/console\.log|print\(/g)?.length > 10) {
            issues.push({ type: 'quality', message: 'Excessive logging statements' });
        }

        return issues;
    }

    /**
     * Detect language from file extension
     */
    detectLanguage(filepath) {
        const ext = path.extname(filepath).toLowerCase();
        const map = {
            '.js': 'javascript',
            '.py': 'python',
            '.go': 'go',
            '.rs': 'rust',
            '.r': 'r',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c'
        };
        return map[ext] || 'unknown';
    }

    /**
     * Start development server
     */
    async startServer(type, port, script, options) {
        try {
            let command;
            const portArg = port ? `--port ${port}` : '';

            switch (type.toLowerCase()) {
                case 'node':
                    command = `node ${script || 'server.js'} ${portArg} ${options.join(' ')}`;
                    break;
                case 'python':
                    command = `python3 -m http.server ${port || 8000} ${options.join(' ')}`;
                    break;
                case 'go':
                    command = `go run ${script || 'main.go'} ${options.join(' ')}`;
                    break;
                default:
                    return { success: false, error: `Server type ${type} not supported` };
            }

            // Start in background (simplified - in production use proper process management)
            const process = spawn('sh', ['-c', command], { detached: true });
            process.unref();

            return {
                success: true,
                message: `Server started on port ${port || 'default'}`,
                pid: process.pid
            };
        } catch (error) {
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Create data visualization
     */
    async createVisualization(data, type, options) {
        try {
            // Parse data
            let parsedData;
            try {
                parsedData = JSON.parse(data);
            } catch {
                // Try CSV
                parsedData = this.parseCSV(data);
            }

            // Generate visualization code
            const vizCode = this.generateVizCode(parsedData, type, options);
            
            return {
                success: true,
                code: vizCode,
                type: 'html',
                message: `Visualization code generated for ${type} chart`
            };
        } catch (error) {
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Generate visualization code
     */
    generateVizCode(data, type, options) {
        // Generate HTML with Chart.js
        return `
<!DOCTYPE html>
<html>
<head>
    <title>${type} Chart</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <canvas id="chart"></canvas>
    <script>
        const ctx = document.getElementById('chart').getContext('2d');
        const data = ${JSON.stringify(data)};
        new Chart(ctx, {
            type: '${type}',
            data: data,
            options: ${JSON.stringify(options)}
        });
    </script>
</body>
</html>`;
    }

    /**
     * Parse CSV data
     */
    parseCSV(csv) {
        const lines = csv.split('\n');
        const headers = lines[0].split(',');
        const data = lines.slice(1).map(line => {
            const values = line.split(',');
            return headers.reduce((obj, header, i) => {
                obj[header.trim()] = values[i]?.trim();
                return obj;
            }, {});
        });
        return { labels: headers, datasets: [{ data }] };
    }

    /**
     * File operations
     */
    async fileOperations(operation, pattern, replacement, directory) {
        try {
            const safeDir = directory ? this.sanitizePath(directory) : process.cwd();

            switch (operation) {
                case 'search':
                    return await this.searchFiles(safeDir, pattern);
                case 'replace':
                    return await this.replaceInFiles(safeDir, pattern, replacement);
                case 'find':
                    return await this.findFiles(safeDir, pattern);
                default:
                    return { success: false, error: `Operation ${operation} not supported` };
            }
        } catch (error) {
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Search files for pattern
     */
    async searchFiles(directory, pattern) {
        const { stdout } = await execAsync(`grep -r "${pattern}" "${directory}"`, { timeout: 30000 });
        return {
            success: true,
            matches: stdout.split('\n').filter(l => l.trim())
        };
    }

    /**
     * Replace in files
     */
    async replaceInFiles(directory, pattern, replacement) {
        const { stdout } = await execAsync(`find "${directory}" -type f -exec sed -i 's/${pattern}/${replacement}/g' {} +`, { timeout: 60000 });
        return {
            success: true,
            message: 'Replacement completed'
        };
    }

    /**
     * Find files
     */
    async findFiles(directory, pattern) {
        const { stdout } = await execAsync(`find "${directory}" -name "${pattern}"`, { timeout: 30000 });
        return {
            success: true,
            files: stdout.split('\n').filter(f => f.trim())
        };
    }

    /**
     * Manage processes
     */
    async manageProcesses(action, pid, name) {
        try {
            let command;
            switch (action) {
                case 'list':
                    command = name ? `ps aux | grep "${name}"` : 'ps aux';
                    break;
                case 'kill':
                    if (!pid) return { success: false, error: 'PID required for kill' };
                    command = `kill ${pid}`;
                    break;
                default:
                    return { success: false, error: `Action ${action} not supported` };
            }

            const { stdout } = await execAsync(command, { timeout: 10000 });
            return {
                success: true,
                output: stdout
            };
        } catch (error) {
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Sanitize path
     */
    sanitizePath(filepath) {
        const resolved = path.resolve(filepath);
        const cwd = process.cwd();
        const dataDir = path.join(cwd, 'data');
        
        if (!resolved.startsWith(cwd) && !resolved.startsWith(dataDir)) {
            throw new Error('Path outside allowed directory');
        }
        
        return resolved;
    }
}

module.exports = EnhancedTools;

