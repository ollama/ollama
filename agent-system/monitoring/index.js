/**
 * System Monitoring
 *
 * Real-time monitoring for the agent system:
 * - Health checks
 * - Metrics collection
 * - Activity logging
 * - Status dashboard API
 */

const os = require('os');

class SystemMonitor {
    constructor(options = {}) {
        this.logger = options.logger || console;

        // Metrics storage
        this.metrics = {
            requests: { total: 0, success: 0, failed: 0 },
            messages: { received: 0, processed: 0, errors: 0 },
            knowledge: { searches: 0, factsAdded: 0, docsIndexed: 0 },
            agent: { plansCreated: 0, stepsExecuted: 0, confirmations: 0 },
            startTime: Date.now(),
            lastActivity: null
        };

        // Activity log (circular buffer)
        this.activityLog = [];
        this.maxLogSize = 100;

        // Health check results
        this.healthChecks = {};

        // Component references (set during initialization)
        this.components = {};
    }

    /**
     * Register components for monitoring
     */
    registerComponents(components) {
        this.components = components;
        this.logger.info('Monitor: Components registered');
    }

    /**
     * Log an activity
     */
    logActivity(type, details, userId = null) {
        const activity = {
            id: Date.now().toString(36) + Math.random().toString(36).substr(2, 5),
            timestamp: new Date().toISOString(),
            type,
            details,
            userId
        };

        this.activityLog.unshift(activity);
        if (this.activityLog.length > this.maxLogSize) {
            this.activityLog.pop();
        }

        this.metrics.lastActivity = activity.timestamp;
        return activity;
    }

    /**
     * Increment a metric
     */
    increment(category, metric, amount = 1) {
        if (this.metrics[category] && typeof this.metrics[category][metric] === 'number') {
            this.metrics[category][metric] += amount;
        }
    }

    /**
     * Run all health checks
     */
    async runHealthChecks() {
        const checks = {};

        // System health
        checks.system = {
            status: 'healthy',
            uptime: Math.floor((Date.now() - this.metrics.startTime) / 1000),
            memory: {
                used: Math.round(process.memoryUsage().heapUsed / 1024 / 1024),
                total: Math.round(os.totalmem() / 1024 / 1024),
                free: Math.round(os.freemem() / 1024 / 1024)
            },
            cpu: os.loadavg()[0]
        };

        // Ollama health
        checks.ollama = await this.checkOllama();

        // Knowledge base health
        checks.knowledgeBase = this.checkKnowledgeBase();

        // MCP health
        checks.mcp = this.checkMCP();

        // Telegram health
        checks.telegram = this.checkTelegram();

        this.healthChecks = checks;
        return checks;
    }

    /**
     * Check Ollama connectivity
     */
    async checkOllama() {
        try {
            const ollamaApi = process.env.OLLAMA_API || 'http://localhost:11434';
            const response = await fetch(`${ollamaApi}/api/tags`, {
                signal: AbortSignal.timeout(5000)
            });

            if (response.ok) {
                const data = await response.json();
                return {
                    status: 'healthy',
                    models: data.models?.length || 0,
                    availableModels: data.models?.map(m => m.name) || []
                };
            }
            return { status: 'unhealthy', error: `HTTP ${response.status}` };
        } catch (error) {
            return { status: 'unhealthy', error: error.message };
        }
    }

    /**
     * Check knowledge base
     */
    checkKnowledgeBase() {
        const kb = this.components.knowledgeBase;
        if (!kb) {
            return { status: 'not_configured' };
        }

        try {
            const stats = kb.getStats();
            return {
                status: 'healthy',
                documents: stats.documents,
                facts: stats.facts,
                embeddings: stats.embeddingsLoaded,
                usingDatabase: stats.usingDatabase
            };
        } catch (error) {
            return { status: 'unhealthy', error: error.message };
        }
    }

    /**
     * Check MCP integration
     */
    checkMCP() {
        const mcp = this.components.mcpIntegration;
        if (!mcp) {
            return { status: 'not_configured' };
        }

        try {
            const status = mcp.getStatus();
            return {
                status: status.browserAutomation ? 'healthy' : 'partial',
                browserReady: status.browserAutomation,
                connectedServers: status.connectedServers || 0
            };
        } catch (error) {
            return { status: 'unhealthy', error: error.message };
        }
    }

    /**
     * Check Telegram bot
     */
    checkTelegram() {
        const telegram = this.components.telegramAdapter;
        if (!telegram) {
            return { status: 'not_configured' };
        }

        try {
            return {
                status: telegram.isRunning() ? 'healthy' : 'stopped',
                running: telegram.isRunning()
            };
        } catch (error) {
            return { status: 'unhealthy', error: error.message };
        }
    }

    /**
     * Get full system status
     */
    async getStatus() {
        const health = await this.runHealthChecks();

        return {
            status: this.getOverallStatus(health),
            uptime: this.formatUptime(Date.now() - this.metrics.startTime),
            health,
            metrics: this.metrics,
            recentActivity: this.activityLog.slice(0, 20)
        };
    }

    /**
     * Determine overall system status
     */
    getOverallStatus(health) {
        const statuses = Object.values(health).map(h => h.status);

        if (statuses.includes('unhealthy')) return 'degraded';
        if (statuses.every(s => s === 'healthy')) return 'healthy';
        return 'partial';
    }

    /**
     * Format uptime as human readable
     */
    formatUptime(ms) {
        const seconds = Math.floor(ms / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);
        const days = Math.floor(hours / 24);

        if (days > 0) return `${days}d ${hours % 24}h`;
        if (hours > 0) return `${hours}h ${minutes % 60}m`;
        if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
        return `${seconds}s`;
    }

    /**
     * Get metrics summary
     */
    getMetricsSummary() {
        return {
            ...this.metrics,
            uptime: this.formatUptime(Date.now() - this.metrics.startTime),
            successRate: this.metrics.requests.total > 0
                ? ((this.metrics.requests.success / this.metrics.requests.total) * 100).toFixed(1) + '%'
                : 'N/A'
        };
    }

    /**
     * Add monitoring routes to Express app
     */
    addRoutes(app) {
        // Full status dashboard
        app.get('/api/monitor/status', async (req, res) => {
            try {
                const status = await this.getStatus();
                res.json(status);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        // Health check (for load balancers, etc.)
        app.get('/api/monitor/health', async (req, res) => {
            try {
                const health = await this.runHealthChecks();
                const status = this.getOverallStatus(health);

                res.status(status === 'healthy' ? 200 : 503).json({
                    status,
                    checks: health
                });
            } catch (error) {
                res.status(500).json({ status: 'error', error: error.message });
            }
        });

        // Metrics only
        app.get('/api/monitor/metrics', (req, res) => {
            res.json(this.getMetricsSummary());
        });

        // Activity log
        app.get('/api/monitor/activity', (req, res) => {
            const limit = parseInt(req.query.limit) || 50;
            res.json(this.activityLog.slice(0, limit));
        });

        // Simple HTML dashboard
        app.get('/dashboard', async (req, res) => {
            const status = await this.getStatus();
            res.send(this.renderDashboard(status));
        });

        this.logger.info('Monitor: Routes added');
    }

    /**
     * Render simple HTML dashboard
     */
    renderDashboard(status) {
        const statusColor = {
            healthy: '#22c55e',
            partial: '#f59e0b',
            degraded: '#ef4444'
        };

        return `
<!DOCTYPE html>
<html>
<head>
    <title>Agent System Monitor</title>
    <meta http-equiv="refresh" content="10">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a; color: #e2e8f0; padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { margin-bottom: 20px; display: flex; align-items: center; gap: 10px; }
        .status-badge {
            padding: 4px 12px; border-radius: 20px; font-size: 14px;
            background: ${statusColor[status.status] || '#666'};
            color: white; text-transform: uppercase;
        }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; }
        .card {
            background: #1e293b; border-radius: 12px; padding: 20px;
            border: 1px solid #334155;
        }
        .card h2 { font-size: 14px; color: #94a3b8; margin-bottom: 12px; text-transform: uppercase; }
        .card-value { font-size: 32px; font-weight: bold; }
        .card-sub { color: #64748b; font-size: 14px; margin-top: 4px; }
        .health-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }
        .health-item {
            background: #0f172a; padding: 12px; border-radius: 8px;
            display: flex; align-items: center; gap: 10px;
        }
        .health-dot {
            width: 10px; height: 10px; border-radius: 50%;
        }
        .health-dot.healthy { background: #22c55e; }
        .health-dot.partial { background: #f59e0b; }
        .health-dot.unhealthy { background: #ef4444; }
        .health-dot.not_configured { background: #64748b; }
        .activity-list { max-height: 300px; overflow-y: auto; }
        .activity-item {
            padding: 10px; border-bottom: 1px solid #334155;
            display: flex; justify-content: space-between;
        }
        .activity-type {
            font-size: 12px; padding: 2px 8px; border-radius: 4px;
            background: #334155; margin-right: 10px;
        }
        .activity-time { color: #64748b; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>
            Agent System Monitor
            <span class="status-badge">${status.status}</span>
        </h1>

        <div class="grid">
            <div class="card">
                <h2>Uptime</h2>
                <div class="card-value">${status.uptime}</div>
                <div class="card-sub">Since ${new Date(status.metrics.startTime).toLocaleString()}</div>
            </div>

            <div class="card">
                <h2>Messages</h2>
                <div class="card-value">${status.metrics.messages.processed}</div>
                <div class="card-sub">${status.metrics.messages.errors} errors</div>
            </div>

            <div class="card">
                <h2>Knowledge Base</h2>
                <div class="card-value">${status.health.knowledgeBase?.documents || 0} docs</div>
                <div class="card-sub">${status.health.knowledgeBase?.facts || 0} facts learned</div>
            </div>

            <div class="card">
                <h2>Memory</h2>
                <div class="card-value">${status.health.system?.memory?.used || 0} MB</div>
                <div class="card-sub">${status.health.system?.memory?.free || 0} MB free</div>
            </div>
        </div>

        <div class="card" style="margin-top: 20px;">
            <h2>Component Health</h2>
            <div class="health-grid">
                ${Object.entries(status.health).map(([name, check]) => `
                    <div class="health-item">
                        <div class="health-dot ${check.status}"></div>
                        <div>
                            <strong>${name}</strong>
                            <div style="font-size: 12px; color: #64748b;">
                                ${check.error || check.status}
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>

        <div class="card" style="margin-top: 20px;">
            <h2>Recent Activity</h2>
            <div class="activity-list">
                ${status.recentActivity.length === 0
                    ? '<div style="padding: 20px; text-align: center; color: #64748b;">No activity yet</div>'
                    : status.recentActivity.map(a => `
                        <div class="activity-item">
                            <div>
                                <span class="activity-type">${a.type}</span>
                                ${a.details}
                            </div>
                            <span class="activity-time">${new Date(a.timestamp).toLocaleTimeString()}</span>
                        </div>
                    `).join('')
                }
            </div>
        </div>

        <div style="margin-top: 20px; color: #64748b; font-size: 12px;">
            Auto-refreshes every 10 seconds |
            <a href="/api/monitor/status" style="color: #60a5fa;">JSON API</a> |
            <a href="/api/monitor/health" style="color: #60a5fa;">Health Check</a>
        </div>
    </div>
</body>
</html>`;
    }
}

module.exports = SystemMonitor;
