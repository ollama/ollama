#!/usr/bin/env node
/**
 * Agent System CLI
 *
 * Interactive terminal for:
 * - Monitoring the agent
 * - Adding knowledge (training)
 * - Testing queries
 * - Managing facts
 */

const readline = require('readline');
const path = require('path');

// Colors for terminal
const colors = {
    reset: '\x1b[0m',
    bright: '\x1b[1m',
    dim: '\x1b[2m',
    green: '\x1b[32m',
    yellow: '\x1b[33m',
    blue: '\x1b[34m',
    magenta: '\x1b[35m',
    cyan: '\x1b[36m',
    red: '\x1b[31m'
};

const c = (color, text) => `${colors[color]}${text}${colors.reset}`;

class AgentCLI {
    constructor() {
        this.baseUrl = process.env.API_URL || 'http://localhost:3000';
        this.rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });
        this.watching = false;
        this.watchInterval = null;
    }

    async start() {
        console.clear();
        this.printBanner();
        await this.checkConnection();
        this.printHelp();
        this.prompt();
    }

    printBanner() {
        console.log(c('cyan', `
╔═══════════════════════════════════════════════════════════╗
║           ${c('bright', 'AGENT SYSTEM CLI')}                              ║
║           Monitor • Train • Test                          ║
╚═══════════════════════════════════════════════════════════╝
`));
    }

    printHelp() {
        console.log(`
${c('bright', 'Commands:')}

  ${c('green', 'status')}              Show system status
  ${c('green', 'health')}              Run health checks
  ${c('green', 'watch')}               Live monitor (updates every 5s)
  ${c('green', 'stop')}                Stop watching

${c('bright', 'Knowledge/Training:')}

  ${c('yellow', 'facts')}               List all learned facts
  ${c('yellow', 'add-fact')} <fact>     Add a new fact about you
  ${c('yellow', 'search')} <query>      Search knowledge base
  ${c('yellow', 'index')} <path>        Index a file or directory
  ${c('yellow', 'profile')}             Show your profile
  ${c('yellow', 'set')} <key> <value>   Set profile field

${c('bright', 'Testing:')}

  ${c('magenta', 'ask')} <question>      Test a query (simulates message)
  ${c('magenta', 'activity')}            Show recent activity

${c('bright', 'Other:')}

  ${c('dim', 'help')}                Show this help
  ${c('dim', 'clear')}               Clear screen
  ${c('dim', 'exit')}                Exit CLI
`);
    }

    async checkConnection() {
        try {
            const res = await fetch(`${this.baseUrl}/api/monitor/health`);
            if (res.ok) {
                const data = await res.json();
                console.log(c('green', `✓ Connected to agent system (${data.status})`));
            } else {
                console.log(c('red', `✗ Agent system returned ${res.status}`));
            }
        } catch (e) {
            console.log(c('red', `✗ Cannot connect to ${this.baseUrl}`));
            console.log(c('dim', '  Make sure the server is running: npm start'));
        }
    }

    prompt() {
        this.rl.question(c('cyan', '\nagent> '), async (input) => {
            await this.handleCommand(input.trim());
            this.prompt();
        });
    }

    async handleCommand(input) {
        if (!input) return;

        const [cmd, ...args] = input.split(' ');
        const arg = args.join(' ');

        try {
            switch (cmd.toLowerCase()) {
                case 'help':
                    this.printHelp();
                    break;

                case 'clear':
                    console.clear();
                    this.printBanner();
                    break;

                case 'exit':
                case 'quit':
                    this.cleanup();
                    process.exit(0);
                    break;

                case 'status':
                    await this.showStatus();
                    break;

                case 'health':
                    await this.showHealth();
                    break;

                case 'watch':
                    this.startWatching();
                    break;

                case 'stop':
                    this.stopWatching();
                    break;

                case 'facts':
                    await this.showFacts();
                    break;

                case 'add-fact':
                    if (!arg) {
                        console.log(c('red', 'Usage: add-fact <fact about you>'));
                        console.log(c('dim', 'Example: add-fact I prefer morning meetings'));
                    } else {
                        await this.addFact(arg);
                    }
                    break;

                case 'search':
                    if (!arg) {
                        console.log(c('red', 'Usage: search <query>'));
                    } else {
                        await this.search(arg);
                    }
                    break;

                case 'index':
                    if (!arg) {
                        console.log(c('red', 'Usage: index <file or directory path>'));
                    } else {
                        await this.indexPath(arg);
                    }
                    break;

                case 'profile':
                    await this.showProfile();
                    break;

                case 'set':
                    const [key, ...valueParts] = args;
                    const value = valueParts.join(' ');
                    if (!key || !value) {
                        console.log(c('red', 'Usage: set <key> <value>'));
                        console.log(c('dim', 'Example: set name John'));
                    } else {
                        await this.setProfile(key, value);
                    }
                    break;

                case 'ask':
                    if (!arg) {
                        console.log(c('red', 'Usage: ask <question>'));
                    } else {
                        await this.askQuestion(arg);
                    }
                    break;

                case 'activity':
                    await this.showActivity();
                    break;

                default:
                    console.log(c('red', `Unknown command: ${cmd}`));
                    console.log(c('dim', 'Type "help" for available commands'));
            }
        } catch (error) {
            console.log(c('red', `Error: ${error.message}`));
        }
    }

    async showStatus() {
        const res = await fetch(`${this.baseUrl}/api/monitor/status`);
        const data = await res.json();

        console.log(`
${c('bright', 'System Status')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status:    ${this.statusColor(data.status)}
Uptime:    ${data.uptime}

${c('bright', 'Metrics')}
Messages:  ${data.metrics.messages.processed} processed, ${data.metrics.messages.errors} errors
Requests:  ${data.metrics.requests.total} total

${c('bright', 'Knowledge Base')}
Documents: ${data.health.knowledgeBase?.documents || 0}
Facts:     ${data.health.knowledgeBase?.facts || 0}
Embeddings: ${data.health.knowledgeBase?.embeddings || 0}

${c('bright', 'Memory')}
Used:      ${data.health.system?.memory?.used || 0} MB
Free:      ${data.health.system?.memory?.free || 0} MB
`);
    }

    async showHealth() {
        const res = await fetch(`${this.baseUrl}/api/monitor/health`);
        const data = await res.json();

        console.log(`\n${c('bright', 'Component Health')}`);
        console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

        for (const [name, check] of Object.entries(data.checks)) {
            const status = this.statusColor(check.status);
            const detail = check.error || (check.models ? `${check.models} models` : '');
            console.log(`${name.padEnd(15)} ${status} ${c('dim', detail)}`);
        }
    }

    statusColor(status) {
        switch (status) {
            case 'healthy': return c('green', '● healthy');
            case 'partial': return c('yellow', '◐ partial');
            case 'degraded': return c('yellow', '◐ degraded');
            case 'unhealthy': return c('red', '○ unhealthy');
            case 'not_configured': return c('dim', '○ not configured');
            default: return c('dim', `○ ${status}`);
        }
    }

    startWatching() {
        if (this.watching) {
            console.log(c('yellow', 'Already watching. Type "stop" to stop.'));
            return;
        }

        this.watching = true;
        console.log(c('green', 'Watching system status... (type "stop" to stop)'));

        this.watchInterval = setInterval(async () => {
            try {
                const res = await fetch(`${this.baseUrl}/api/monitor/status`);
                const data = await res.json();

                // Move cursor up and clear
                process.stdout.write('\x1b[2K\x1b[1A'.repeat(3));

                console.log(`${c('dim', new Date().toLocaleTimeString())} | ` +
                    `Status: ${this.statusColor(data.status)} | ` +
                    `Msgs: ${data.metrics.messages.processed} | ` +
                    `Docs: ${data.health.knowledgeBase?.documents || 0} | ` +
                    `Facts: ${data.health.knowledgeBase?.facts || 0}`);
                console.log('');
                console.log('');
            } catch (e) {
                // Ignore errors during watch
            }
        }, 5000);
    }

    stopWatching() {
        if (!this.watching) {
            console.log(c('dim', 'Not currently watching.'));
            return;
        }

        clearInterval(this.watchInterval);
        this.watching = false;
        console.log(c('green', 'Stopped watching.'));
    }

    async showFacts() {
        const res = await fetch(`${this.baseUrl}/api/knowledge/facts`);
        const data = await res.json();

        console.log(`\n${c('bright', 'Learned Facts')} (${data.count} total)`);
        console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

        if (data.summary) {
            console.log(data.summary);
        } else {
            console.log(c('dim', 'No facts learned yet.'));
            console.log(c('dim', 'Use "add-fact <fact>" to add one.'));
        }
    }

    async addFact(fact) {
        // Try to auto-detect category
        let category = 'general';
        if (/prefer|like|love|hate|favorite/i.test(fact)) category = 'preferences';
        if (/name is|i am|my name/i.test(fact)) category = 'identity';
        if (/live|location|city|address/i.test(fact)) category = 'location';
        if (/work|job|company|role/i.test(fact)) category = 'work';
        if (/schedule|morning|evening|routine/i.test(fact)) category = 'schedule';

        const res = await fetch(`${this.baseUrl}/api/knowledge/facts`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ fact, category, confidence: 0.95 })
        });

        if (res.ok) {
            console.log(c('green', `✓ Added fact (category: ${category})`));
            console.log(c('dim', `  "${fact}"`));
        } else {
            const err = await res.json();
            console.log(c('red', `✗ Failed: ${err.error}`));
        }
    }

    async search(query) {
        const res = await fetch(`${this.baseUrl}/api/knowledge/search?query=${encodeURIComponent(query)}`);
        const data = await res.json();

        console.log(`\n${c('bright', 'Search Results')} for "${query}"`);
        console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

        if (data.documents?.length > 0) {
            console.log(`\n${c('cyan', 'Documents:')}`);
            for (const doc of data.documents) {
                const title = doc.document?.filename || doc.document?.title || 'Untitled';
                console.log(`  • ${title}`);
                if (doc.document?.content) {
                    console.log(c('dim', `    ${doc.document.content.substring(0, 100)}...`));
                }
            }
        }

        if (data.facts?.length > 0) {
            console.log(`\n${c('yellow', 'Facts:')}`);
            for (const fact of data.facts) {
                console.log(`  • ${fact.fact}`);
            }
        }

        if (!data.documents?.length && !data.facts?.length) {
            console.log(c('dim', 'No results found.'));
        }
    }

    async indexPath(inputPath) {
        const fullPath = path.isAbsolute(inputPath) ? inputPath : path.resolve(process.cwd(), inputPath);

        console.log(c('dim', `Indexing: ${fullPath}`));

        // Check if it's a directory or file
        const fs = require('fs');
        const isDir = fs.existsSync(fullPath) && fs.statSync(fullPath).isDirectory();

        const endpoint = isDir ? '/api/knowledge/index/directory' : '/api/knowledge/index/file';
        const body = isDir ? { dirPath: fullPath } : { filePath: fullPath };

        const res = await fetch(`${this.baseUrl}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });

        if (res.ok) {
            const data = await res.json();
            if (isDir) {
                console.log(c('green', `✓ Indexed ${data.indexed} files`));
            } else {
                console.log(c('green', `✓ Indexed file (ID: ${data.docId})`));
            }
        } else {
            const err = await res.json();
            console.log(c('red', `✗ Failed: ${err.error}`));
        }
    }

    async showProfile() {
        const res = await fetch(`${this.baseUrl}/api/knowledge/profile`);
        const data = await res.json();

        console.log(`\n${c('bright', 'User Profile')}`);
        console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

        if (Object.keys(data).length === 0) {
            console.log(c('dim', 'No profile data yet.'));
            console.log(c('dim', 'Use "set <key> <value>" to add fields.'));
        } else {
            for (const [key, value] of Object.entries(data)) {
                if (key !== 'lastUpdated') {
                    console.log(`  ${c('cyan', key)}: ${value}`);
                }
            }
            if (data.lastUpdated) {
                console.log(c('dim', `\nLast updated: ${new Date(data.lastUpdated).toLocaleString()}`));
            }
        }
    }

    async setProfile(key, value) {
        const res = await fetch(`${this.baseUrl}/api/knowledge/profile`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ [key]: value })
        });

        if (res.ok) {
            console.log(c('green', `✓ Set ${key} = ${value}`));
        } else {
            const err = await res.json();
            console.log(c('red', `✗ Failed: ${err.error}`));
        }
    }

    async askQuestion(question) {
        console.log(c('dim', 'Thinking...'));

        // This would need an endpoint to actually process through SmartAgent
        // For now, just search the knowledge base
        const res = await fetch(`${this.baseUrl}/api/knowledge/search?query=${encodeURIComponent(question)}`);
        const data = await res.json();

        console.log(`\n${c('bright', 'Knowledge Base Context:')}`);

        if (data.facts?.length > 0) {
            console.log(c('yellow', 'Relevant facts:'));
            for (const fact of data.facts.slice(0, 5)) {
                console.log(`  • ${fact.fact}`);
            }
        }

        if (data.documents?.length > 0) {
            console.log(c('cyan', '\nRelevant documents:'));
            for (const doc of data.documents.slice(0, 3)) {
                console.log(`  • ${doc.document?.filename || doc.document?.title || 'Document'}`);
            }
        }

        console.log(c('dim', '\n(Full response would come from Telegram bot or API)'));
    }

    async showActivity() {
        const res = await fetch(`${this.baseUrl}/api/monitor/activity?limit=20`);
        const data = await res.json();

        console.log(`\n${c('bright', 'Recent Activity')}`);
        console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

        if (data.length === 0) {
            console.log(c('dim', 'No activity yet.'));
        } else {
            for (const activity of data) {
                const time = new Date(activity.timestamp).toLocaleTimeString();
                console.log(`${c('dim', time)} ${c('cyan', activity.type.padEnd(12))} ${activity.details}`);
            }
        }
    }

    cleanup() {
        this.stopWatching();
        this.rl.close();
    }
}

// Run CLI
const cli = new AgentCLI();
cli.start();

// Handle Ctrl+C
process.on('SIGINT', () => {
    cli.cleanup();
    process.exit(0);
});
