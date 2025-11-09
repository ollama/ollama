/**
 * Workflow Orchestrator
 * 
 * Manages complex multi-step workflows with state management and error handling
 */

const logger = require('winston').default || console;

class WorkflowOrchestrator {
    constructor(ollamaClient, agents, toolSystem, taskPlanner) {
        this.ollamaClient = ollamaClient;
        this.agents = agents;
        this.toolSystem = toolSystem;
        this.taskPlanner = taskPlanner;
        this.activeWorkflows = new Map();
    }

    /**
     * Start a new workflow
     */
    async startWorkflow(config) {
        const workflowId = `workflow_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

        const workflow = {
            id: workflowId,
            name: config.name || 'Unnamed Workflow',
            description: config.description || '',
            steps: config.steps || [],
            currentStep: 0,
            status: 'running',
            results: [],
            errors: [],
            context: config.initialContext || {},
            startTime: new Date(),
            endTime: null
        };

        this.activeWorkflows.set(workflowId, workflow);

        // Execute workflow asynchronously
        this.executeWorkflow(workflowId).catch(error => {
            logger.error(`Workflow ${workflowId} failed:`, error);
            workflow.status = 'error';
            workflow.errors.push(error.message);
        });

        return workflow;
    }

    /**
     * Execute workflow steps
     */
    async executeWorkflow(workflowId) {
        const workflow = this.activeWorkflows.get(workflowId);
        if (!workflow) {
            throw new Error('Workflow not found');
        }

        for (let i = 0; i < workflow.steps.length; i++) {
            const step = workflow.steps[i];
            workflow.currentStep = i + 1;

            logger.info(`Executing workflow step ${i + 1}/${workflow.steps.length}: ${step.name || step.type}`);

            try {
                const result = await this.executeStep(step, workflow);
                workflow.results.push({
                    stepIndex: i,
                    stepName: step.name || step.type,
                    result,
                    timestamp: new Date()
                });

                // Update context with step result
                if (step.outputKey) {
                    workflow.context[step.outputKey] = result;
                }

                // Check for conditional execution
                if (step.condition && !this.evaluateCondition(step.condition, workflow.context)) {
                    logger.info(`Step ${i + 1} condition not met, skipping`);
                    continue;
                }

            } catch (error) {
                logger.error(`Workflow step ${i + 1} failed:`, error);
                workflow.results.push({
                    stepIndex: i,
                    stepName: step.name || step.type,
                    error: error.message,
                    timestamp: new Date()
                });
                workflow.errors.push(`Step ${i + 1}: ${error.message}`);

                // Handle error strategy
                if (step.onError === 'stop') {
                    workflow.status = 'error';
                    break;
                } else if (step.onError === 'continue') {
                    continue;
                } else if (step.onError === 'retry' && step.retries) {
                    // Retry logic
                    let retried = false;
                    for (let retry = 0; retry < step.retries; retry++) {
                        try {
                            const result = await this.executeStep(step, workflow);
                            workflow.results[workflow.results.length - 1].result = result;
                            workflow.results[workflow.results.length - 1].retries = retry + 1;
                            retried = true;
                            break;
                        } catch (retryError) {
                            logger.warn(`Retry ${retry + 1} failed:`, retryError);
                        }
                    }
                    if (!retried) {
                        workflow.status = 'error';
                        break;
                    }
                }
            }
        }

        if (workflow.status !== 'error') {
            workflow.status = 'completed';
            workflow.endTime = new Date();
        }

        return workflow;
    }

    /**
     * Execute a single workflow step
     */
    async executeStep(step, workflow) {
        switch (step.type) {
            case 'agent_call':
                return await this.executeAgentCall(step, workflow);
            
            case 'tool_call':
                return await this.executeToolCall(step, workflow);
            
            case 'task_plan':
                return await this.executeTaskPlan(step, workflow);
            
            case 'parallel':
                return await this.executeParallel(step, workflow);
            
            case 'condition':
                return await this.executeCondition(step, workflow);
            
            case 'loop':
                return await this.executeLoop(step, workflow);
            
            default:
                throw new Error(`Unknown step type: ${step.type}`);
        }
    }

    /**
     * Execute agent call step
     */
    async executeAgentCall(step, workflow) {
        const agentKey = step.agent || 'researcher';
        const message = this.interpolateContext(step.message, workflow.context);
        const systemPrompt = step.systemPrompt ? 
            this.interpolateContext(step.systemPrompt, workflow.context) : null;

        const response = await this.ollamaClient(agentKey, message, null, 
            systemPrompt ? [{ role: 'system', content: systemPrompt }] : null);

        // Process tool calls if any
        const toolResults = await this.toolSystem.executeToolCalls(response);
        if (toolResults.length > 0) {
            const enhancedResponse = this.toolSystem.replaceToolCalls(response, toolResults);
            return { response: enhancedResponse, toolResults };
        }

        return { response };
    }

    /**
     * Execute tool call step
     */
    async executeToolCall(step, workflow) {
        const toolName = step.tool;
        const params = this.interpolateContext(step.params || {}, workflow.context);

        const tool = this.toolSystem.tools.get(toolName);
        if (!tool) {
            throw new Error(`Tool not found: ${toolName}`);
        }

        return await tool.handler(params);
    }

    /**
     * Execute task plan step
     */
    async executeTaskPlan(step, workflow) {
        const task = this.interpolateContext(step.task, workflow.context);
        const agentKey = step.agent || 'planner';

        const plannedTask = await this.taskPlanner.planTask(task, agentKey);
        const executedTask = await this.taskPlanner.executeTask(plannedTask.id, agentKey);

        return {
            taskId: executedTask.id,
            status: executedTask.status,
            results: executedTask.results,
            synthesis: executedTask.synthesis
        };
    }

    /**
     * Execute parallel steps
     */
    async executeParallel(step, workflow) {
        const steps = step.steps || [];
        const results = await Promise.allSettled(
            steps.map(s => this.executeStep(s, workflow))
        );

        return results.map((result, index) => ({
            stepIndex: index,
            success: result.status === 'fulfilled',
            result: result.status === 'fulfilled' ? result.value : result.reason
        }));
    }

    /**
     * Execute conditional step
     */
    async executeCondition(step, workflow) {
        const condition = this.evaluateCondition(step.condition, workflow.context);
        
        if (condition) {
            return await this.executeStep(step.then, workflow);
        } else if (step.else) {
            return await this.executeStep(step.else, workflow);
        }
        
        return { condition: false, executed: false };
    }

    /**
     * Execute loop step
     */
    async executeLoop(step, workflow) {
        const items = this.interpolateContext(step.items, workflow.context);
        const results = [];

        for (const item of items) {
            workflow.context[step.itemKey || 'item'] = item;
            const result = await this.executeStep(step.step, workflow);
            results.push(result);
        }

        return results;
    }

    /**
     * Interpolate context variables in strings/objects
     */
    interpolateContext(template, context) {
        if (typeof template === 'string') {
            return template.replace(/\{\{(\w+)\}\}/g, (match, key) => {
                return context[key] !== undefined ? String(context[key]) : match;
            });
        } else if (typeof template === 'object' && template !== null) {
            if (Array.isArray(template)) {
                return template.map(item => this.interpolateContext(item, context));
            } else {
                const result = {};
                for (const [key, value] of Object.entries(template)) {
                    result[key] = this.interpolateContext(value, context);
                }
                return result;
            }
        }
        return template;
    }

    /**
     * Evaluate condition
     */
    evaluateCondition(condition, context) {
        if (typeof condition === 'string') {
            // Simple variable check
            return context[condition] !== undefined && context[condition] !== false;
        } else if (typeof condition === 'object') {
            // Complex condition
            if (condition.type === 'equals') {
                return context[condition.key] === condition.value;
            } else if (condition.type === 'exists') {
                return context[condition.key] !== undefined;
            } else if (condition.type === 'and') {
                return condition.conditions.every(c => this.evaluateCondition(c, context));
            } else if (condition.type === 'or') {
                return condition.conditions.some(c => this.evaluateCondition(c, context));
            }
        }
        return false;
    }

    /**
     * Get workflow status
     */
    getWorkflow(workflowId) {
        return this.activeWorkflows.get(workflowId);
    }

    /**
     * Get all workflows
     */
    getAllWorkflows() {
        return Array.from(this.activeWorkflows.values());
    }

    /**
     * Cancel workflow
     */
    cancelWorkflow(workflowId) {
        const workflow = this.activeWorkflows.get(workflowId);
        if (workflow && workflow.status === 'running') {
            workflow.status = 'cancelled';
            workflow.endTime = new Date();
            return true;
        }
        return false;
    }
}

module.exports = WorkflowOrchestrator;

