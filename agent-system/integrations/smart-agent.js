/**
 * Smart Agent
 *
 * A general-purpose AI agent that can handle any request by:
 * 1. Understanding the user's intent
 * 2. Planning the necessary steps
 * 3. Executing using available tools
 * 4. Adapting based on results
 *
 * This replaces rigid workflow templates with intelligent planning.
 */

class SmartAgent {
    constructor(options = {}) {
        this.logger = options.logger || console;
        this.callOllama = options.callOllama;
        this.toolSystem = options.toolSystem;
        this.browserAutomation = options.browserAutomation;
        this.knowledgeBase = options.knowledgeBase || null;  // Personal knowledge base for RAG
        this.userContext = new Map();  // userId → context
        this.pendingConfirmations = new Map();  // userId → pending action
        this.progressCallback = options.onProgress || null;  // For real-time updates
    }

    /**
     * Set the knowledge base (can be set after construction)
     */
    setKnowledgeBase(knowledgeBase) {
        this.knowledgeBase = knowledgeBase;
        this.logger.info('Knowledge base connected to SmartAgent');
    }

    /**
     * Set progress callback for real-time feedback
     */
    setProgressCallback(callback) {
        this.progressCallback = callback;
    }

    /**
     * Send progress update to user
     */
    async sendProgress(userId, message, type = 'thinking') {
        if (this.progressCallback) {
            await this.progressCallback(userId, { type, message });
        }
        this.logger.info(`[${type}] ${message}`);
    }

    /**
     * Main entry point - handle any user request
     */
    async handleRequest(message, session) {
        const userId = session.userId;

        // Get or create user context
        let context = this.userContext.get(userId) || this.createContext(session);
        context.lastMessage = message;
        context.history = session.history || [];

        // Check for pending confirmation
        if (this.pendingConfirmations.has(userId)) {
            return await this.handleConfirmationResponse(message, userId, context);
        }

        try {
            // Step 0: Get relevant personal context from knowledge base
            let personalContext = '';
            if (this.knowledgeBase) {
                try {
                    personalContext = await this.knowledgeBase.getRelevantContext(message);
                    if (personalContext) {
                        context.personalContext = personalContext;
                        this.logger.info(`Retrieved personal context: ${personalContext.length} chars`);
                    }
                } catch (kbError) {
                    this.logger.warn('Failed to get personal context:', kbError.message);
                }
            }

            // Step 1: Understand what the user wants
            await this.sendProgress(userId, 'Understanding your request...', 'thinking');
            const intent = await this.analyzeIntent(message, context);
            this.logger.info(`Intent: ${intent.type} - ${intent.summary}`);

            // Step 2: Plan the approach
            await this.sendProgress(userId, 'Planning how to help...', 'planning');
            const plan = await this.createPlan(intent, context);
            this.logger.info(`Plan: ${plan.steps.length} steps`);

            // Step 3: Check if confirmation needed for risky actions
            if (this.requiresConfirmation(intent, plan)) {
                return await this.requestConfirmation(userId, intent, plan, context);
            }

            // Step 4: Execute the plan with progress updates
            const result = await this.executePlanWithProgress(plan, context, userId);

            // Step 5: Formulate response
            const response = await this.formulateResponse(result, intent, context);

            // Update context
            context.lastIntent = intent;
            context.lastPlan = plan;
            this.userContext.set(userId, context);

            return response;

        } catch (error) {
            this.logger.error('Smart agent error:', error);
            return `I encountered an issue: ${error.message}. Could you rephrase or provide more details?`;
        }
    }

    /**
     * Check if action requires user confirmation
     */
    requiresConfirmation(intent, plan) {
        // Actions that modify external state should be confirmed
        const riskyTypes = ['booking', 'purchase', 'email', 'calendar', 'payment'];
        const riskyActions = ['send_email', 'add_calendar', 'browser_click', 'submit'];

        if (riskyTypes.includes(intent.type)) {
            return true;
        }

        // Check if plan contains risky steps
        for (const step of plan.steps) {
            if (riskyActions.some(action =>
                step.type?.includes(action) || step.action?.includes(action) || step.tool?.includes(action)
            )) {
                return true;
            }
        }

        return false;
    }

    /**
     * Request user confirmation before proceeding
     */
    async requestConfirmation(userId, intent, plan, context) {
        const planSummary = plan.steps
            .filter(s => s.description)
            .map((s, i) => `${i + 1}. ${s.description}`)
            .join('\n');

        const confirmationMessage = `I understand you want to: **${intent.summary}**

Here's my plan:
${planSummary}

${plan.warnings?.length ? `\n⚠️ Note: ${plan.warnings.join(', ')}\n` : ''}
**Should I proceed?** (Reply "yes" to confirm, or tell me what to change)`;

        // Store pending confirmation
        this.pendingConfirmations.set(userId, {
            intent,
            plan,
            context,
            timestamp: Date.now()
        });

        return confirmationMessage;
    }

    /**
     * Handle user's response to confirmation request
     */
    async handleConfirmationResponse(message, userId, context) {
        const pending = this.pendingConfirmations.get(userId);

        // Clear pending after 5 minutes
        if (Date.now() - pending.timestamp > 300000) {
            this.pendingConfirmations.delete(userId);
            return "The previous request has expired. Please tell me again what you'd like to do.";
        }

        const response = message.toLowerCase().trim();

        // User confirmed
        if (['yes', 'y', 'ok', 'sure', 'go ahead', 'proceed', 'confirm', 'do it'].includes(response)) {
            this.pendingConfirmations.delete(userId);

            await this.sendProgress(userId, 'Got it! Starting now...', 'executing');

            // Execute the plan
            const result = await this.executePlanWithProgress(pending.plan, pending.context, userId);
            return await this.formulateResponse(result, pending.intent, pending.context);
        }

        // User cancelled
        if (['no', 'n', 'cancel', 'stop', 'nevermind', 'never mind'].includes(response)) {
            this.pendingConfirmations.delete(userId);
            return "No problem, I've cancelled that. Let me know if you need anything else!";
        }

        // User wants changes - treat as new request with context
        this.pendingConfirmations.delete(userId);
        context.lastIntent = pending.intent;
        return await this.handleRequest(message, { ...context, history: context.history });
    }

    /**
     * Execute plan with progress updates
     */
    async executePlanWithProgress(plan, context, userId) {
        const results = [];
        const totalSteps = plan.steps.length;

        for (let i = 0; i < totalSteps; i++) {
            const step = plan.steps[i];
            const stepNum = i + 1;

            // Send progress update
            const progressMsg = step.description || `Step ${stepNum} of ${totalSteps}`;
            await this.sendProgress(userId, `[${stepNum}/${totalSteps}] ${progressMsg}`, 'executing');

            try {
                const result = await this.executeStepWithTimeout(step, context, results);
                results.push({
                    step: i,
                    type: step.type,
                    success: true,
                    result
                });

                // If step asks for clarification, stop here
                if (step.type === 'ask_clarification' || step.type === 'ask_user') {
                    break;
                }

            } catch (error) {
                this.logger.error(`Step ${stepNum} failed:`, error);

                await this.sendProgress(userId, `Step ${stepNum} encountered an issue: ${error.message}`, 'error');

                results.push({
                    step: i,
                    type: step.type,
                    success: false,
                    error: error.message
                });

                // Continue if step is optional
                if (!step.optional) {
                    break;
                }
            }
        }

        return {
            plan,
            results,
            success: results.every(r => r.success || plan.steps[r.step]?.optional)
        };
    }

    /**
     * Execute step with timeout
     */
    async executeStepWithTimeout(step, context, previousResults, timeout = 30000) {
        return Promise.race([
            this.executeStep(step, context, previousResults),
            new Promise((_, reject) =>
                setTimeout(() => reject(new Error('Step timed out')), timeout)
            )
        ]);
    }

    /**
     * Create initial context for a user
     */
    createContext(session) {
        return {
            userId: session.userId,
            platform: session.platform,
            preferences: session.preferences || {},
            metadata: session.metadata || {},
            knownInfo: {},  // Things we've learned about the user
            activeTask: null,
            history: []
        };
    }

    /**
     * Analyze user intent
     */
    async analyzeIntent(message, context) {
        const historyContext = context.history.slice(-6).map(h =>
            `${h.role}: ${h.content}`
        ).join('\n');

        // Include personal context from knowledge base
        const personalInfo = context.personalContext || '';

        const prompt = `Analyze this user message and determine their intent.

Previous conversation:
${historyContext || 'None'}

${personalInfo ? `Personal context about the user:\n${personalInfo}\n` : ''}
Current message: "${message}"

Known user info: ${JSON.stringify(context.knownInfo)}

Respond with JSON only:
{
    "type": "one of: question, task, booking, search, email, calendar, purchase, reminder, information, conversation, clarification, other",
    "summary": "brief description of what user wants",
    "entities": {
        "date": "extracted date if any",
        "time": "extracted time if any",
        "location": "extracted location if any",
        "person": "extracted person/business name if any",
        "service": "extracted service type if any",
        "item": "extracted item if any",
        "amount": "extracted amount/price if any",
        "email": "extracted email if any",
        "phone": "extracted phone if any",
        "url": "extracted URL if any"
    },
    "requiresAction": true/false,
    "requiresWebBrowsing": true/false,
    "requiresEmail": true/false,
    "requiresCalendar": true/false,
    "confidence": 0.0-1.0,
    "clarificationNeeded": "what info is missing, or null",
    "followUp": true/false
}`;

        const response = await this.callOllama('planner', prompt);

        try {
            const jsonMatch = response.match(/\{[\s\S]*\}/);
            if (jsonMatch) {
                return JSON.parse(jsonMatch[0]);
            }
        } catch (e) {
            this.logger.warn('Failed to parse intent JSON, using defaults');
        }

        // Fallback intent
        return {
            type: 'conversation',
            summary: message,
            entities: {},
            requiresAction: false,
            requiresWebBrowsing: false,
            confidence: 0.5,
            clarificationNeeded: null,
            followUp: false
        };
    }

    /**
     * Create execution plan based on intent
     */
    async createPlan(intent, context) {
        // If clarification needed, return simple ask plan
        if (intent.clarificationNeeded) {
            return {
                steps: [{
                    type: 'ask_clarification',
                    question: intent.clarificationNeeded
                }],
                estimatedTime: 'instant'
            };
        }

        // If no action required (just conversation), return chat plan
        if (!intent.requiresAction && !intent.requiresWebBrowsing) {
            return {
                steps: [{
                    type: 'chat_response',
                    intent: intent
                }],
                estimatedTime: 'instant'
            };
        }

        // For complex tasks, ask the AI to plan
        const availableTools = this.getToolDescriptions();

        const prompt = `Create a step-by-step plan to accomplish this task.

User wants: ${intent.summary}
Intent type: ${intent.type}
Extracted info: ${JSON.stringify(intent.entities)}

Available tools:
${availableTools}

Create a practical plan. Respond with JSON only:
{
    "steps": [
        {
            "type": "tool_call | browser_action | agent_think | ask_user | send_email | add_calendar",
            "description": "what this step does",
            "tool": "tool name if applicable",
            "params": {},
            "dependsOn": [step indices this depends on],
            "optional": true/false
        }
    ],
    "estimatedTime": "quick/medium/long",
    "warnings": ["any potential issues"],
    "alternatives": "backup approach if primary fails"
}

Keep it practical - prefer fewer steps. Don't overcomplicate.`;

        const response = await this.callOllama('planner', prompt);

        try {
            const jsonMatch = response.match(/\{[\s\S]*\}/);
            if (jsonMatch) {
                return JSON.parse(jsonMatch[0]);
            }
        } catch (e) {
            this.logger.warn('Failed to parse plan JSON');
        }

        // Fallback: simple web search plan for tasks
        if (intent.requiresWebBrowsing) {
            return {
                steps: [
                    { type: 'browser_action', description: 'Search the web', action: 'search', query: intent.summary },
                    { type: 'agent_think', description: 'Analyze results and respond' }
                ],
                estimatedTime: 'medium'
            };
        }

        return {
            steps: [{ type: 'chat_response', intent }],
            estimatedTime: 'instant'
        };
    }

    /**
     * Execute a single step
     */
    async executeStep(step, context, previousResults) {
        switch (step.type) {
            case 'ask_clarification':
            case 'ask_user':
                return { needsInput: true, question: step.question || step.description };

            case 'chat_response':
                return { type: 'chat', intent: step.intent };

            case 'tool_call':
                return await this.executeToolCall(step, context);

            case 'browser_action':
                return await this.executeBrowserAction(step, context);

            case 'agent_think':
                return await this.agentThink(step, context, previousResults);

            case 'send_email':
                return await this.executeSendEmail(step, context);

            case 'add_calendar':
                return await this.executeAddCalendar(step, context);

            case 'web_search':
                return await this.executeWebSearch(step, context);

            default:
                return { type: 'unknown', step };
        }
    }

    /**
     * Execute a tool call
     */
    async executeToolCall(step, context) {
        const tool = this.toolSystem.tools.get(step.tool);
        if (!tool) {
            throw new Error(`Tool not found: ${step.tool}`);
        }

        const result = await tool.handler(step.params || {});
        return result;
    }

    /**
     * Execute browser action
     */
    async executeBrowserAction(step, context) {
        if (!this.browserAutomation) {
            throw new Error('Browser automation not available');
        }

        const sessionId = context.userId;

        switch (step.action) {
            case 'navigate':
                return await this.browserAutomation.navigate(step.url, sessionId);

            case 'search':
                // Use web search tool instead
                const searchTool = this.toolSystem.tools.get('web_search');
                if (searchTool) {
                    return await searchTool.handler({ query: step.query });
                }
                // Fallback to Google search via browser
                const searchUrl = `https://www.google.com/search?q=${encodeURIComponent(step.query)}`;
                return await this.browserAutomation.navigate(searchUrl, sessionId);

            case 'click':
                return await this.browserAutomation.click(step.selector, sessionId);

            case 'fill':
                return await this.browserAutomation.fill(step.selector, step.value, sessionId);

            case 'extract':
                return await this.browserAutomation.extract(step.selector, {}, sessionId);

            case 'screenshot':
                return await this.browserAutomation.screenshot(sessionId);

            case 'get_page_info':
                const text = await this.browserAutomation.getPageText(sessionId);
                const forms = await this.browserAutomation.getFormFields(sessionId);
                const clickables = await this.browserAutomation.getClickableElements(sessionId);
                return { text, forms, clickables };

            default:
                throw new Error(`Unknown browser action: ${step.action}`);
        }
    }

    /**
     * Let the agent think/analyze
     */
    async agentThink(step, context, previousResults) {
        const resultsContext = previousResults.map((r, i) =>
            `Step ${i + 1} (${r.type}): ${JSON.stringify(r.result).substring(0, 500)}`
        ).join('\n\n');

        const prompt = `Analyze these results and determine the next action or final response.

Task: ${context.lastIntent?.summary || context.lastMessage}

Results so far:
${resultsContext}

What should we do next? If we have enough information, provide the final answer.
If we need more information, specify what action to take.`;

        return await this.callOllama('researcher', prompt);
    }

    /**
     * Execute web search
     */
    async executeWebSearch(step, context) {
        const searchTool = this.toolSystem.tools.get('web_search');
        if (searchTool) {
            return await searchTool.handler({ query: step.query });
        }
        throw new Error('Web search tool not available');
    }

    /**
     * Send email
     */
    async executeSendEmail(step, context) {
        const emailTool = this.toolSystem.tools.get('send_email');
        if (!emailTool) {
            throw new Error('Email not configured');
        }

        return await emailTool.handler({
            to: step.to,
            subject: step.subject,
            body: step.body
        });
    }

    /**
     * Add calendar event
     */
    async executeAddCalendar(step, context) {
        const calendarTool = this.toolSystem.tools.get('mcp_google_calendar_create_event');
        if (!calendarTool) {
            // Return instruction for manual addition
            return {
                success: false,
                manual: true,
                message: `Calendar not configured. Please add manually: ${step.title} on ${step.date} at ${step.time}`
            };
        }

        return await calendarTool.handler({
            title: step.title,
            start: step.start,
            end: step.end,
            location: step.location,
            description: step.description
        });
    }

    /**
     * Formulate final response to user
     */
    async formulateResponse(executionResult, intent, context) {
        const { results, success } = executionResult;
        const lastResult = results[results.length - 1];

        // If asking for clarification
        if (lastResult?.result?.needsInput) {
            return lastResult.result.question;
        }

        // If simple chat response
        if (lastResult?.type === 'chat_response') {
            return await this.generateChatResponse(intent, context);
        }

        // If agent thinking produced a response
        if (lastResult?.type === 'agent_think' && typeof lastResult.result === 'string') {
            return lastResult.result;
        }

        // For complex results, summarize
        const prompt = `Summarize these results into a helpful response for the user.

User asked: ${context.lastMessage}
Intent: ${intent.summary}

Execution results:
${results.map((r, i) => `${i + 1}. ${r.type}: ${r.success ? 'Success' : 'Failed'} - ${JSON.stringify(r.result).substring(0, 300)}`).join('\n')}

Overall success: ${success}

Provide a clear, concise response. If something failed, explain what happened and suggest alternatives.
Keep it conversational and helpful.`;

        return await this.callOllama('researcher', prompt);
    }

    /**
     * Generate simple chat response
     */
    async generateChatResponse(intent, context) {
        const historyContext = context.history.slice(-6).map(h =>
            `${h.role}: ${h.content}`
        ).join('\n');

        // Include personal context if available
        const personalInfo = context.personalContext || '';

        const systemPrompt = `You are a helpful personal AI assistant. You know your user personally and can:
- Search the web and find information
- Browse websites and interact with them
- Book appointments and make reservations
- Manage calendars and set reminders
- Send emails
- Help with research and analysis
- Answer questions on any topic

Be conversational, helpful, and concise. Use what you know about the user to give personalized responses.`;

        const prompt = `${systemPrompt}

${personalInfo ? `What I know about this user:\n${personalInfo}\n` : ''}
Conversation:
${historyContext}

User: ${context.lastMessage}

Respond helpfully and personally:`;

        return await this.callOllama('researcher', prompt);
    }

    /**
     * Learn from conversation (extract and store facts)
     */
    async learnFromConversation(messages, userId) {
        if (!this.knowledgeBase) return;

        try {
            const facts = await this.knowledgeBase.extractFactsFromConversation(messages);
            if (facts.length > 0) {
                this.logger.info(`Learned ${facts.length} new facts about user ${userId}`);
            }
        } catch (error) {
            this.logger.warn('Failed to learn from conversation:', error.message);
        }
    }

    /**
     * Get descriptions of available tools
     */
    getToolDescriptions() {
        const tools = this.toolSystem?.getAvailableTools() || [];

        const descriptions = tools.map(t => `- ${t.name}: ${t.description}`).join('\n');

        return `
BROWSER TOOLS:
- browser_navigate: Go to a URL
- browser_click: Click an element
- browser_fill: Fill a form field
- browser_extract: Extract data from page
- browser_get_text: Get page text content
- browser_get_forms: Get form fields on page
- browser_get_clickables: Get clickable elements
- browser_screenshot: Take screenshot

REGISTERED TOOLS:
${descriptions}

ACTIONS:
- web_search: Search the web
- send_email: Send an email (if configured)
- add_calendar: Add calendar event (if configured)
`;
    }

    /**
     * Update user's known info
     */
    updateKnownInfo(userId, key, value) {
        const context = this.userContext.get(userId);
        if (context) {
            context.knownInfo[key] = value;
        }
    }

    /**
     * Get user context
     */
    getContext(userId) {
        return this.userContext.get(userId);
    }
}

module.exports = SmartAgent;
