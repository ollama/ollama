/**
 * Custom Command System
 *
 * Allows users to define shortcuts and custom commands:
 * - Built-in commands (/help, /status, /facts)
 * - User-defined commands (/haircut → "Book haircut at...")
 * - Parameterized commands (/remind @time message)
 * - Scheduled commands (run daily at 9am)
 */

const fs = require('fs').promises;
const path = require('path');

class CommandSystem {
    constructor(options = {}) {
        this.logger = options.logger || console;
        this.dataDir = options.dataDir || path.join(__dirname, '../data');
        this.smartAgent = options.smartAgent;
        this.knowledgeBase = options.knowledgeBase;
        this.monitor = options.monitor;

        // Built-in commands
        this.builtInCommands = new Map();
        this.registerBuiltInCommands();

        // User-defined commands
        this.userCommands = new Map();

        // Scheduled commands
        this.scheduledCommands = [];
        this.schedulerInterval = null;

        // Command history for learning
        this.commandHistory = [];
    }

    async initialize() {
        await this.loadUserCommands();
        this.startScheduler();
        this.logger.info(`Command system initialized: ${this.userCommands.size} user commands`);
    }

    /**
     * Register built-in commands
     */
    registerBuiltInCommands() {
        // Help command
        this.builtInCommands.set('help', {
            description: 'Show available commands',
            usage: '/help [command]',
            handler: async (args, context) => {
                if (args.length > 0) {
                    return this.getCommandHelp(args[0]);
                }
                return this.getAllCommandsHelp();
            }
        });

        // Status command
        this.builtInCommands.set('status', {
            description: 'Show system status',
            usage: '/status',
            handler: async (args, context) => {
                if (this.monitor) {
                    const status = await this.monitor.getStatus();
                    return `**System Status: ${status.status}**\n` +
                        `Uptime: ${status.uptime}\n` +
                        `Messages: ${status.metrics.messages.processed}\n` +
                        `Knowledge: ${status.health.knowledgeBase?.documents || 0} docs, ${status.health.knowledgeBase?.facts || 0} facts`;
                }
                return 'Monitor not available';
            }
        });

        // Facts command
        this.builtInCommands.set('facts', {
            description: 'Show what I know about you',
            usage: '/facts [category]',
            handler: async (args, context) => {
                if (!this.knowledgeBase) return 'Knowledge base not available';

                if (args.length > 0) {
                    const facts = this.knowledgeBase.getFactsByCategory(args[0]);
                    if (facts.length === 0) return `No facts in category: ${args[0]}`;
                    return `**${args[0]} facts:**\n` + facts.map(f => `• ${f.fact}`).join('\n');
                }

                return this.knowledgeBase.getFactsSummary();
            }
        });

        // Remember command - add a fact
        this.builtInCommands.set('remember', {
            description: 'Remember a fact about you',
            usage: '/remember <fact>',
            examples: ['/remember I prefer morning meetings', '/remember My dentist is Dr. Smith'],
            handler: async (args, context) => {
                if (args.length === 0) return 'Usage: /remember <fact about you>';

                const fact = args.join(' ');
                if (this.knowledgeBase) {
                    await this.knowledgeBase.storeFact(fact, { source: 'user_command' });
                    return `Got it! I'll remember: "${fact}"`;
                }
                return 'Knowledge base not available';
            }
        });

        // Forget command - remove a fact (mark as low confidence)
        this.builtInCommands.set('forget', {
            description: 'Forget a fact',
            usage: '/forget <search term>',
            handler: async (args, context) => {
                if (args.length === 0) return 'Usage: /forget <search term>';
                // This would need implementation to actually remove facts
                return 'Forget functionality not yet implemented. Facts can be manually removed from the database.';
            }
        });

        // Search command
        this.builtInCommands.set('search', {
            description: 'Search your knowledge base',
            usage: '/search <query>',
            handler: async (args, context) => {
                if (args.length === 0) return 'Usage: /search <query>';

                const query = args.join(' ');
                if (this.knowledgeBase) {
                    const results = await this.knowledgeBase.semanticSearch(query, { limit: 5 });

                    if (results.length === 0) return `No results for: "${query}"`;

                    let response = `**Results for "${query}":**\n\n`;
                    for (const r of results) {
                        if (r.type === 'fact') {
                            response += `📝 ${r.item.fact}\n`;
                        } else {
                            response += `📄 ${r.item.filename || r.item.title || 'Document'}\n`;
                        }
                    }
                    return response;
                }
                return 'Knowledge base not available';
            }
        });

        // Commands management
        this.builtInCommands.set('commands', {
            description: 'List your custom commands',
            usage: '/commands',
            handler: async (args, context) => {
                if (this.userCommands.size === 0) {
                    return 'No custom commands defined.\nUse /newcmd to create one.';
                }

                let response = '**Your Custom Commands:**\n\n';
                for (const [name, cmd] of this.userCommands) {
                    response += `/${name} - ${cmd.description || 'No description'}\n`;
                }
                return response;
            }
        });

        // Create new command
        this.builtInCommands.set('newcmd', {
            description: 'Create a custom command',
            usage: '/newcmd <name> <expansion>',
            examples: [
                '/newcmd haircut Book a haircut at Great Clips for Saturday morning',
                '/newcmd weather What is the weather today?'
            ],
            handler: async (args, context) => {
                if (args.length < 2) {
                    return 'Usage: /newcmd <name> <what it expands to>\n\n' +
                        'Example: /newcmd haircut Book a haircut at Great Clips for Saturday';
                }

                const name = args[0].replace(/^\//, '');
                const expansion = args.slice(1).join(' ');

                await this.addUserCommand(name, {
                    expansion,
                    description: `Expands to: "${expansion.substring(0, 50)}..."`,
                    createdAt: new Date().toISOString()
                });

                return `Created command /${name}\nIt will expand to: "${expansion}"`;
            }
        });

        // Delete command
        this.builtInCommands.set('delcmd', {
            description: 'Delete a custom command',
            usage: '/delcmd <name>',
            handler: async (args, context) => {
                if (args.length === 0) return 'Usage: /delcmd <command name>';

                const name = args[0].replace(/^\//, '');
                if (this.userCommands.has(name)) {
                    this.userCommands.delete(name);
                    await this.saveUserCommands();
                    return `Deleted command /${name}`;
                }
                return `Command /${name} not found`;
            }
        });

        // Quick actions
        this.builtInCommands.set('quick', {
            description: 'Show quick action suggestions',
            usage: '/quick',
            handler: async (args, context) => {
                return `**Quick Actions:**\n\n` +
                    `📅 /today - What's on my schedule today?\n` +
                    `📧 /emails - Check recent emails\n` +
                    `🔍 /search <term> - Search knowledge base\n` +
                    `📝 /remember <fact> - Remember something about me\n` +
                    `⚙️ /status - System status\n\n` +
                    `Create custom commands with /newcmd`;
            }
        });

        // Profile command
        this.builtInCommands.set('profile', {
            description: 'Show or update your profile',
            usage: '/profile [key value]',
            handler: async (args, context) => {
                if (!this.knowledgeBase) return 'Knowledge base not available';

                if (args.length >= 2) {
                    const key = args[0];
                    const value = args.slice(1).join(' ');
                    await this.knowledgeBase.setProfileField(key, value);
                    return `Updated profile: ${key} = ${value}`;
                }

                const profile = this.knowledgeBase.userProfile;
                if (Object.keys(profile).length === 0) {
                    return 'No profile data yet.\nUse /profile <key> <value> to set fields.';
                }

                let response = '**Your Profile:**\n\n';
                for (const [key, value] of Object.entries(profile)) {
                    if (key !== 'lastUpdated') {
                        response += `${key}: ${value}\n`;
                    }
                }
                return response;
            }
        });
    }

    /**
     * Process a message - check if it's a command
     */
    async processMessage(message, context) {
        const trimmed = message.trim();

        // Check if it's a command (starts with /)
        if (!trimmed.startsWith('/')) {
            return null; // Not a command, let SmartAgent handle it
        }

        // Parse command
        const parts = trimmed.slice(1).split(/\s+/);
        const commandName = parts[0].toLowerCase();
        const args = parts.slice(1);

        // Track command usage
        this.commandHistory.push({
            command: commandName,
            args,
            timestamp: new Date().toISOString(),
            userId: context.userId
        });

        // Check built-in commands first
        if (this.builtInCommands.has(commandName)) {
            const cmd = this.builtInCommands.get(commandName);
            return await cmd.handler(args, context);
        }

        // Check user-defined commands
        if (this.userCommands.has(commandName)) {
            const cmd = this.userCommands.get(commandName);
            // Expand the command and send to SmartAgent
            let expansion = cmd.expansion;

            // Replace any parameters (e.g., $1, $2 or @arg)
            args.forEach((arg, i) => {
                expansion = expansion.replace(new RegExp(`\\$${i + 1}|@${i + 1}`, 'g'), arg);
            });

            // Return the expansion to be processed by SmartAgent
            return { type: 'expand', text: expansion };
        }

        // Unknown command
        return `Unknown command: /${commandName}\nType /help for available commands.`;
    }

    /**
     * Get help for a specific command
     */
    getCommandHelp(commandName) {
        const name = commandName.replace(/^\//, '');

        if (this.builtInCommands.has(name)) {
            const cmd = this.builtInCommands.get(name);
            let help = `**/${name}**\n${cmd.description}\n\nUsage: ${cmd.usage}`;
            if (cmd.examples) {
                help += '\n\nExamples:\n' + cmd.examples.map(e => `  ${e}`).join('\n');
            }
            return help;
        }

        if (this.userCommands.has(name)) {
            const cmd = this.userCommands.get(name);
            return `**/${name}** (custom)\n${cmd.description}\n\nExpands to: "${cmd.expansion}"`;
        }

        return `Command /${name} not found`;
    }

    /**
     * Get help for all commands
     */
    getAllCommandsHelp() {
        let help = '**Available Commands:**\n\n';

        help += '**Built-in:**\n';
        for (const [name, cmd] of this.builtInCommands) {
            help += `  /${name} - ${cmd.description}\n`;
        }

        if (this.userCommands.size > 0) {
            help += '\n**Custom:**\n';
            for (const [name, cmd] of this.userCommands) {
                help += `  /${name} - ${cmd.description}\n`;
            }
        }

        help += '\nType /help <command> for details';
        return help;
    }

    /**
     * Add a user-defined command
     */
    async addUserCommand(name, config) {
        this.userCommands.set(name.toLowerCase(), config);
        await this.saveUserCommands();
    }

    /**
     * Load user commands from disk
     */
    async loadUserCommands() {
        try {
            const filePath = path.join(this.dataDir, 'commands.json');
            const data = await fs.readFile(filePath, 'utf8');
            const commands = JSON.parse(data);
            this.userCommands = new Map(Object.entries(commands));
        } catch (error) {
            if (error.code !== 'ENOENT') {
                this.logger.warn('Failed to load user commands:', error.message);
            }
        }
    }

    /**
     * Save user commands to disk
     */
    async saveUserCommands() {
        try {
            await fs.mkdir(this.dataDir, { recursive: true });
            const filePath = path.join(this.dataDir, 'commands.json');
            const data = Object.fromEntries(this.userCommands);
            await fs.writeFile(filePath, JSON.stringify(data, null, 2));
        } catch (error) {
            this.logger.error('Failed to save user commands:', error.message);
        }
    }

    /**
     * Schedule a command to run at a specific time
     */
    scheduleCommand(config) {
        this.scheduledCommands.push({
            id: Date.now().toString(36),
            ...config,
            createdAt: new Date().toISOString()
        });
        // Save scheduled commands
        this.saveScheduledCommands();
    }

    /**
     * Start the scheduler for recurring commands
     */
    startScheduler() {
        // Check every minute for scheduled commands
        this.schedulerInterval = setInterval(() => {
            this.checkScheduledCommands();
        }, 60000);
    }

    /**
     * Check and execute scheduled commands
     */
    async checkScheduledCommands() {
        const now = new Date();

        for (const scheduled of this.scheduledCommands) {
            if (this.shouldRunScheduled(scheduled, now)) {
                this.logger.info(`Running scheduled command: ${scheduled.command}`);
                // Execute through SmartAgent
                if (this.smartAgent) {
                    try {
                        await this.smartAgent.handleRequest(scheduled.command, {
                            userId: scheduled.userId || 'scheduled',
                            platform: 'scheduler'
                        });
                        scheduled.lastRun = now.toISOString();
                    } catch (error) {
                        this.logger.error('Scheduled command failed:', error.message);
                    }
                }
            }
        }
    }

    /**
     * Check if a scheduled command should run
     */
    shouldRunScheduled(scheduled, now) {
        // Simple implementation - check if it matches the schedule
        // Format: { hour: 9, minute: 0, days: ['mon', 'tue', 'wed', 'thu', 'fri'] }
        if (!scheduled.schedule) return false;

        const { hour, minute, days } = scheduled.schedule;
        const dayNames = ['sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat'];
        const today = dayNames[now.getDay()];

        if (days && !days.includes(today)) return false;
        if (hour !== undefined && now.getHours() !== hour) return false;
        if (minute !== undefined && now.getMinutes() !== minute) return false;

        // Check if already run this minute
        if (scheduled.lastRun) {
            const lastRun = new Date(scheduled.lastRun);
            if (now.getTime() - lastRun.getTime() < 60000) return false;
        }

        return true;
    }

    async saveScheduledCommands() {
        try {
            const filePath = path.join(this.dataDir, 'scheduled.json');
            await fs.writeFile(filePath, JSON.stringify(this.scheduledCommands, null, 2));
        } catch (error) {
            this.logger.error('Failed to save scheduled commands:', error.message);
        }
    }

    /**
     * Get command suggestions based on history
     */
    getSuggestions(partial) {
        const suggestions = [];
        const lower = partial.toLowerCase().replace(/^\//, '');

        // Search built-in commands
        for (const name of this.builtInCommands.keys()) {
            if (name.startsWith(lower)) {
                suggestions.push({ name, type: 'builtin' });
            }
        }

        // Search user commands
        for (const name of this.userCommands.keys()) {
            if (name.startsWith(lower)) {
                suggestions.push({ name, type: 'custom' });
            }
        }

        return suggestions;
    }

    /**
     * Get most used commands
     */
    getMostUsedCommands(limit = 5) {
        const counts = {};
        for (const entry of this.commandHistory) {
            counts[entry.command] = (counts[entry.command] || 0) + 1;
        }

        return Object.entries(counts)
            .sort((a, b) => b[1] - a[1])
            .slice(0, limit)
            .map(([command, count]) => ({ command, count }));
    }

    /**
     * Cleanup
     */
    shutdown() {
        if (this.schedulerInterval) {
            clearInterval(this.schedulerInterval);
        }
    }
}

module.exports = CommandSystem;
