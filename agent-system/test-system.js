#!/usr/bin/env node
/**
 * System Test / Demo Mode
 *
 * Tests the agent system without requiring:
 * - .env file
 * - Running Ollama
 * - Telegram bot token
 * - External services
 *
 * Usage: node test-system.js
 */

const path = require('path');

// Mock logger
const logger = {
    info: (...args) => console.log('ℹ️ ', ...args),
    warn: (...args) => console.log('⚠️ ', ...args),
    error: (...args) => console.log('❌', ...args),
    debug: (...args) => {} // silent
};

// Mock Ollama responses
const mockOllamaResponses = {
    intent: {
        type: 'task',
        summary: 'User wants to book a haircut',
        entities: { service: 'haircut', time: 'Saturday morning' },
        requiresAction: true,
        requiresWebBrowsing: true,
        confidence: 0.9
    },
    plan: {
        steps: [
            { type: 'web_search', description: 'Search for nearby barbershops', query: 'barbershops near me' },
            { type: 'agent_think', description: 'Analyze search results' }
        ],
        estimatedTime: 'medium'
    },
    chat: "I'd be happy to help you book a haircut! Let me search for barbershops near you."
};

async function mockCallOllama(agent, prompt) {
    // Simulate delay
    await new Promise(r => setTimeout(r, 100));

    if (prompt.includes('Analyze this user message')) {
        return JSON.stringify(mockOllamaResponses.intent);
    }
    if (prompt.includes('Create a step-by-step plan')) {
        return JSON.stringify(mockOllamaResponses.plan);
    }
    return mockOllamaResponses.chat;
}

// Colors
const c = {
    reset: '\x1b[0m',
    green: '\x1b[32m',
    red: '\x1b[31m',
    yellow: '\x1b[33m',
    cyan: '\x1b[36m',
    dim: '\x1b[2m'
};

function pass(msg) { console.log(`${c.green}✓${c.reset} ${msg}`); }
function fail(msg, err) { console.log(`${c.red}✗${c.reset} ${msg}: ${err}`); }
function section(msg) { console.log(`\n${c.cyan}▶ ${msg}${c.reset}`); }

async function runTests() {
    console.log(`
╔═══════════════════════════════════════════════════════════╗
║           AGENT SYSTEM TEST SUITE                         ║
║           Testing without external dependencies           ║
╚═══════════════════════════════════════════════════════════╝
`);

    let passed = 0;
    let failed = 0;

    // ========================================
    section('Module Loading');
    // ========================================

    try {
        require('./personalization/knowledge-base');
        pass('PersonalKnowledgeBase loads');
        passed++;
    } catch (e) {
        fail('PersonalKnowledgeBase', e.message);
        failed++;
    }

    try {
        require('./integrations/smart-agent');
        pass('SmartAgent loads');
        passed++;
    } catch (e) {
        fail('SmartAgent', e.message);
        failed++;
    }

    try {
        require('./commands');
        pass('CommandSystem loads');
        passed++;
    } catch (e) {
        fail('CommandSystem', e.message);
        failed++;
    }

    try {
        require('./monitoring');
        pass('SystemMonitor loads');
        passed++;
    } catch (e) {
        fail('SystemMonitor', e.message);
        failed++;
    }

    try {
        require('./integrations/message-gateway');
        pass('MessageGateway loads');
        passed++;
    } catch (e) {
        fail('MessageGateway', e.message);
        failed++;
    }

    // ========================================
    section('Knowledge Base');
    // ========================================

    const PersonalKnowledgeBase = require('./personalization/knowledge-base');
    const kb = new PersonalKnowledgeBase({
        logger,
        dataDir: path.join(__dirname, 'data/test-knowledge'),
        callOllama: mockCallOllama
    });

    try {
        await kb.initialize();
        pass('Knowledge base initializes');
        passed++;
    } catch (e) {
        fail('KB initialize', e.message);
        failed++;
    }

    try {
        const factId = await kb.storeFact('User prefers morning meetings', { category: 'preferences' });
        if (factId) {
            pass('Can store facts');
            passed++;
        } else {
            fail('Store fact', 'No ID returned');
            failed++;
        }
    } catch (e) {
        fail('Store fact', e.message);
        failed++;
    }

    try {
        const facts = kb.searchFacts('morning');
        if (facts.length > 0) {
            pass('Can search facts');
            passed++;
        } else {
            fail('Search facts', 'No results');
            failed++;
        }
    } catch (e) {
        fail('Search facts', e.message);
        failed++;
    }

    try {
        await kb.updateProfile({ name: 'Test User', location: 'Test City' });
        if (kb.userProfile.name === 'Test User') {
            pass('Can update profile');
            passed++;
        } else {
            fail('Update profile', 'Profile not updated');
            failed++;
        }
    } catch (e) {
        fail('Update profile', e.message);
        failed++;
    }

    try {
        const stats = kb.getStats();
        if (stats.facts > 0) {
            pass(`KB stats work (${stats.facts} facts, ${stats.documents} docs)`);
            passed++;
        } else {
            fail('KB stats', 'No stats');
            failed++;
        }
    } catch (e) {
        fail('KB stats', e.message);
        failed++;
    }

    // ========================================
    section('Command System');
    // ========================================

    const CommandSystem = require('./commands');
    const cmdSystem = new CommandSystem({
        logger,
        knowledgeBase: kb
    });

    try {
        await cmdSystem.initialize();
        pass('Command system initializes');
        passed++;
    } catch (e) {
        fail('Cmd init', e.message);
        failed++;
    }

    try {
        const helpResult = await cmdSystem.processMessage('/help', { userId: 'test' });
        if (helpResult && helpResult.includes('Commands')) {
            pass('/help command works');
            passed++;
        } else {
            fail('/help', 'Unexpected result');
            failed++;
        }
    } catch (e) {
        fail('/help', e.message);
        failed++;
    }

    try {
        const statusResult = await cmdSystem.processMessage('/status', { userId: 'test' });
        if (statusResult) {
            pass('/status command works');
            passed++;
        } else {
            fail('/status', 'No result');
            failed++;
        }
    } catch (e) {
        fail('/status', e.message);
        failed++;
    }

    try {
        const factsResult = await cmdSystem.processMessage('/facts', { userId: 'test' });
        if (factsResult && factsResult.includes('morning')) {
            pass('/facts command works (found our test fact)');
            passed++;
        } else {
            fail('/facts', 'Did not find test fact');
            failed++;
        }
    } catch (e) {
        fail('/facts', e.message);
        failed++;
    }

    try {
        const rememberResult = await cmdSystem.processMessage('/remember I like pizza', { userId: 'test' });
        if (rememberResult && rememberResult.includes('remember')) {
            pass('/remember command works');
            passed++;
        } else {
            fail('/remember', 'Unexpected result');
            failed++;
        }
    } catch (e) {
        fail('/remember', e.message);
        failed++;
    }

    try {
        await cmdSystem.addUserCommand('test', { expansion: 'This is a test command', description: 'Test' });
        const customResult = await cmdSystem.processMessage('/test', { userId: 'test' });
        if (customResult && customResult.type === 'expand') {
            pass('Custom commands work');
            passed++;
        } else {
            fail('Custom cmd', 'Did not expand');
            failed++;
        }
    } catch (e) {
        fail('Custom cmd', e.message);
        failed++;
    }

    // ========================================
    section('Smart Agent');
    // ========================================

    const SmartAgent = require('./integrations/smart-agent');
    const agent = new SmartAgent({
        logger,
        callOllama: mockCallOllama,
        knowledgeBase: kb,
        onProgress: async (userId, progress) => {
            console.log(`  ${c.dim}[Progress] ${progress.message}${c.reset}`);
        }
    });

    try {
        const context = agent.createContext({ userId: 'test', platform: 'test' });
        if (context.userId === 'test') {
            pass('Can create user context');
            passed++;
        } else {
            fail('Create context', 'Invalid context');
            failed++;
        }
    } catch (e) {
        fail('Create context', e.message);
        failed++;
    }

    try {
        const intent = await agent.analyzeIntent('Book a haircut for Saturday', {
            history: [],
            knownInfo: {},
            personalContext: ''
        });
        if (intent && intent.type) {
            pass(`Intent analysis works (type: ${intent.type})`);
            passed++;
        } else {
            fail('Intent', 'No type');
            failed++;
        }
    } catch (e) {
        fail('Intent', e.message);
        failed++;
    }

    // ========================================
    section('Monitor');
    // ========================================

    const SystemMonitor = require('./monitoring');
    const monitor = new SystemMonitor({ logger });

    try {
        monitor.logActivity('test', 'Test activity', 'test-user');
        if (monitor.activityLog.length > 0) {
            pass('Can log activity');
            passed++;
        } else {
            fail('Log activity', 'No log');
            failed++;
        }
    } catch (e) {
        fail('Log activity', e.message);
        failed++;
    }

    try {
        monitor.increment('messages', 'received');
        monitor.increment('messages', 'processed');
        if (monitor.metrics.messages.received === 1) {
            pass('Metrics tracking works');
            passed++;
        } else {
            fail('Metrics', 'Not incremented');
            failed++;
        }
    } catch (e) {
        fail('Metrics', e.message);
        failed++;
    }

    try {
        const summary = monitor.getMetricsSummary();
        if (summary.uptime) {
            pass('Metrics summary works');
            passed++;
        } else {
            fail('Summary', 'No uptime');
            failed++;
        }
    } catch (e) {
        fail('Summary', e.message);
        failed++;
    }

    // ========================================
    section('Integration Test');
    // ========================================

    try {
        // Simulate full message flow
        console.log(`  ${c.dim}Simulating: "Book a haircut for Saturday morning"${c.reset}`);

        const session = {
            userId: 'integration-test',
            platform: 'test',
            history: [],
            preferences: {}
        };

        // First check if it's a command
        const cmdResult = await cmdSystem.processMessage('Book a haircut for Saturday morning', {
            userId: session.userId,
            platform: session.platform,
            session
        });

        // Not a command, so should return null
        if (cmdResult === null) {
            pass('Non-command correctly passed through');
            passed++;
        } else {
            fail('Command check', 'Should have returned null');
            failed++;
        }

        // Now process through agent
        // (We can't fully test this without a real Ollama, but we can test the flow)

    } catch (e) {
        fail('Integration', e.message);
        failed++;
    }

    // ========================================
    // Summary
    // ========================================

    console.log(`
╔═══════════════════════════════════════════════════════════╗
║                    TEST RESULTS                           ║
╠═══════════════════════════════════════════════════════════╣
║  ${c.green}Passed: ${passed}${c.reset}
║  ${failed > 0 ? c.red : c.green}Failed: ${failed}${c.reset}
╚═══════════════════════════════════════════════════════════╝
`);

    if (failed === 0) {
        console.log(`${c.green}All tests passed! The system is ready.${c.reset}`);
        console.log(`
${c.cyan}Next steps:${c.reset}
1. Pull the embedding model:
   ${c.dim}ollama pull nomic-embed-text${c.reset}

2. Pull a chat model:
   ${c.dim}ollama pull qwen2.5:7b${c.reset}

3. Create .env file:
   ${c.dim}echo "TELEGRAM_BOT_TOKEN=your_token" > .env${c.reset}

4. Start the server:
   ${c.dim}npm start${c.reset}

5. Open dashboard:
   ${c.dim}http://localhost:3000/dashboard${c.reset}
`);
    } else {
        console.log(`${c.red}Some tests failed. Check the errors above.${c.reset}`);
        process.exit(1);
    }

    // Cleanup test data
    try {
        const fs = require('fs').promises;
        await fs.rm(path.join(__dirname, 'data/test-knowledge'), { recursive: true, force: true });
    } catch (e) {
        // Ignore cleanup errors
    }
}

// Run tests
runTests().catch(err => {
    console.error('Test suite crashed:', err);
    process.exit(1);
});
