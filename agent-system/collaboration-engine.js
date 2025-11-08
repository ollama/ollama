const logger = require('winston');

/**
 * Collaboration Engine
 * Orchestrates multi-agent collaborative workflows
 */

class CollaborationEngine {
    constructor(agents, ollamaClient) {
        this.agents = agents;
        this.ollamaClient = ollamaClient;
        this.activeSessions = new Map();

        // Task templates
        this.templates = {
            code_review: {
                name: 'Code Review',
                description: 'Multiple agents review and critique code',
                roles: {
                    primary: 'coder',
                    reviewers: ['critic', 'researcher'],
                    synthesizer: 'planner'
                },
                rounds: 3,
                systemPrompts: {
                    primary: 'Review this code and provide initial analysis.',
                    reviewers: 'Critique the previous analysis and suggest improvements.',
                    synthesizer: 'Synthesize all feedback into actionable recommendations.'
                }
            },
            research_analysis: {
                name: 'Research & Analysis',
                description: 'Collaborative research with multiple perspectives',
                roles: {
                    primary: 'researcher',
                    reviewers: ['critic', 'planner'],
                    synthesizer: 'coder'
                },
                rounds: 4,
                systemPrompts: {
                    primary: 'Research this topic thoroughly and provide initial findings.',
                    reviewers: 'Analyze the research and identify gaps or biases.',
                    synthesizer: 'Create a comprehensive summary with all perspectives.'
                }
            },
            brainstorm: {
                name: 'Brainstorming Session',
                description: 'Creative ideation with all agents',
                roles: {
                    primary: 'planner',
                    reviewers: ['researcher', 'coder', 'critic'],
                    synthesizer: 'planner'
                },
                rounds: 2,
                systemPrompts: {
                    primary: 'Generate creative ideas for this challenge.',
                    reviewers: 'Build on previous ideas and add your unique perspective.',
                    synthesizer: 'Organize and prioritize all ideas into an action plan.'
                }
            },
            custom: {
                name: 'Custom Workflow',
                description: 'User-defined agent collaboration',
                roles: {
                    primary: null,
                    reviewers: [],
                    synthesizer: null
                },
                rounds: 3,
                systemPrompts: {
                    primary: 'Analyze this task from your perspective.',
                    reviewers: 'Review and expand on the previous response.',
                    synthesizer: 'Synthesize all perspectives into a final answer.'
                }
            }
        };
    }

    /**
     * Start a new collaboration session
     */
    async startCollaboration(config) {
        const sessionId = `collab_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

        const session = {
            id: sessionId,
            task: config.task,
            template: config.template || 'custom',
            participants: config.participants || [],
            rounds: config.rounds || 3,
            maxRounds: config.maxRounds || 5,
            currentRound: 0,
            status: 'running',
            startTime: new Date(),
            endTime: null,
            history: [],
            synthesis: null,
            error: null
        };

        this.activeSessions.set(sessionId, session);

        logger.info(`Started collaboration session ${sessionId}`, {
            template: session.template,
            participants: session.participants,
            rounds: session.rounds
        });

        // Start execution
        this.executeCollaboration(sessionId).catch(error => {
            logger.error(`Collaboration session ${sessionId} failed:`, error);
            session.status = 'error';
            session.error = error.message;
        });

        return session;
    }

    /**
     * Execute collaboration workflow
     */
    async executeCollaboration(sessionId) {
        const session = this.activeSessions.get(sessionId);
        if (!session) {
            throw new Error('Session not found');
        }

        const template = this.templates[session.template];
        const participants = session.participants;

        try {
            // Round 1: Primary agent responds
            session.currentRound = 1;
            const primaryAgent = participants[0];
            const primaryResponse = await this.getAgentResponse(
                primaryAgent,
                session.task,
                template.systemPrompts.primary,
                []
            );

            session.history.push({
                round: 1,
                type: 'primary',
                agent: primaryAgent,
                agentName: this.agents[primaryAgent].name,
                prompt: session.task,
                response: primaryResponse,
                timestamp: new Date()
            });

            logger.info(`Round 1 complete for session ${sessionId}`, {
                agent: primaryAgent,
                responseLength: primaryResponse.length
            });

            // Subsequent rounds: Other agents review and respond
            for (let round = 2; round <= session.rounds; round++) {
                session.currentRound = round;
                const reviewers = participants.slice(1);

                for (const reviewerAgent of reviewers) {
                    const context = this.buildContext(session);
                    const reviewPrompt = `${template.systemPrompts.reviewers}\n\nOriginal Task: ${session.task}\n\nPrevious Responses:\n${context}`;

                    const reviewResponse = await this.getAgentResponse(
                        reviewerAgent,
                        reviewPrompt,
                        template.systemPrompts.reviewers,
                        session.history
                    );

                    session.history.push({
                        round: round,
                        type: 'review',
                        agent: reviewerAgent,
                        agentName: this.agents[reviewerAgent].name,
                        prompt: reviewPrompt,
                        response: reviewResponse,
                        timestamp: new Date()
                    });

                    logger.info(`Round ${round} review complete`, {
                        agent: reviewerAgent,
                        responseLength: reviewResponse.length
                    });
                }
            }

            // Final round: Synthesizer creates final answer
            session.currentRound = session.rounds + 1;
            const synthesizerAgent = participants[participants.length - 1];
            const context = this.buildContext(session);
            const synthesisPrompt = `${template.systemPrompts.synthesizer}\n\nOriginal Task: ${session.task}\n\nAll Agent Responses:\n${context}\n\nProvide a comprehensive final answer that synthesizes all perspectives.`;

            const synthesis = await this.getAgentResponse(
                synthesizerAgent,
                synthesisPrompt,
                template.systemPrompts.synthesizer,
                session.history
            );

            session.synthesis = {
                agent: synthesizerAgent,
                agentName: this.agents[synthesizerAgent].name,
                response: synthesis,
                timestamp: new Date()
            };

            session.status = 'completed';
            session.endTime = new Date();

            logger.info(`Collaboration session ${sessionId} completed`, {
                totalRounds: session.currentRound,
                duration: session.endTime - session.startTime
            });

        } catch (error) {
            session.status = 'error';
            session.error = error.message;
            session.endTime = new Date();
            throw error;
        }
    }

    /**
     * Get response from a specific agent
     */
    async getAgentResponse(agentKey, message, systemPrompt, history) {
        const agent = this.agents[agentKey];
        if (!agent) {
            throw new Error(`Agent ${agentKey} not found`);
        }

        // Build context from history
        const contextMessages = history.slice(-5).map(h => ({
            role: h.type === 'primary' || h.type === 'review' ? 'assistant' : 'user',
            content: `[${h.agentName}]: ${h.response}`
        }));

        const messages = [
            { role: 'system', content: `${agent.systemPrompt}\n\n${systemPrompt}` },
            ...contextMessages,
            { role: 'user', content: message }
        ];

        return await this.ollamaClient(agentKey, message, null, messages);
    }

    /**
     * Build context string from session history
     */
    buildContext(session) {
        return session.history
            .map(h => `[Round ${h.round}] ${h.agentName} (${h.type}):\n${h.response}\n`)
            .join('\n---\n\n');
    }

    /**
     * Get session status
     */
    getSession(sessionId) {
        return this.activeSessions.get(sessionId);
    }

    /**
     * Get all active sessions
     */
    getAllSessions() {
        return Array.from(this.activeSessions.values());
    }

    /**
     * Cancel a running session
     */
    cancelSession(sessionId) {
        const session = this.activeSessions.get(sessionId);
        if (session && session.status === 'running') {
            session.status = 'cancelled';
            session.endTime = new Date();
            logger.info(`Collaboration session ${sessionId} cancelled`);
            return true;
        }
        return false;
    }

    /**
     * Get available templates
     */
    getTemplates() {
        return Object.keys(this.templates).map(key => ({
            id: key,
            name: this.templates[key].name,
            description: this.templates[key].description,
            recommendedAgents: [
                this.templates[key].roles.primary,
                ...this.templates[key].roles.reviewers,
                this.templates[key].roles.synthesizer
            ].filter(Boolean),
            rounds: this.templates[key].rounds
        }));
    }
}

module.exports = CollaborationEngine;
