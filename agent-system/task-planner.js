/**
 * Task Planning and Decomposition System
 * 
 * Breaks down complex tasks into manageable subtasks and orchestrates execution
 */

const logger = require('winston').default || console;

class TaskPlanner {
    constructor(ollamaClient, agents) {
        this.ollamaClient = ollamaClient;
        this.agents = agents;
        this.activeTasks = new Map();
    }

    /**
     * Plan a complex task by breaking it into subtasks
     */
    async planTask(complexTask, agentKey = 'planner') {
        const taskId = `task_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        
        const planningPrompt = `You are an expert task planner. Break down this complex task into clear, actionable subtasks.

Task: ${complexTask}

Analyze the task and create a structured plan with:
1. Subtasks (numbered, specific, actionable)
2. Dependencies (which subtasks depend on others)
3. Estimated complexity (simple/medium/complex)
4. Required resources/tools
5. Success criteria

Format your response as JSON:
{
  "task": "${complexTask}",
  "subtasks": [
    {
      "id": 1,
      "description": "Specific subtask description",
      "dependencies": [],
      "complexity": "simple|medium|complex",
      "requiredTools": ["tool1", "tool2"],
      "successCriteria": "How to know this is complete"
    }
  ],
  "estimatedTime": "X minutes/hours",
  "requiredAgents": ["agent1", "agent2"]
}

Be thorough and break down the task into the smallest actionable steps.`;

        try {
            const response = await this.ollamaClient(agentKey, planningPrompt);
            
            // Extract JSON from response
            let plan;
            try {
                const jsonMatch = response.match(/\{[\s\S]*\}/);
                if (jsonMatch) {
                    plan = JSON.parse(jsonMatch[0]);
                } else {
                    // Fallback: create plan from response
                    plan = this.parsePlanFromText(response, complexTask);
                }
            } catch (e) {
                logger.warn('Failed to parse JSON plan, using text parser');
                plan = this.parsePlanFromText(response, complexTask);
            }

            const task = {
                id: taskId,
                originalTask: complexTask,
                plan: plan,
                status: 'planned',
                currentStep: 0,
                results: [],
                errors: [],
                startTime: new Date(),
                endTime: null
            };

            this.activeTasks.set(taskId, task);
            return task;
        } catch (error) {
            logger.error('Task planning failed:', error);
            throw error;
        }
    }

    /**
     * Parse plan from unstructured text response
     */
    parsePlanFromText(text, originalTask) {
        const subtasks = [];
        const lines = text.split('\n');
        let currentSubtask = null;

        for (const line of lines) {
            // Look for numbered items
            const numberedMatch = line.match(/^(\d+)[\.\)]\s*(.+)/);
            if (numberedMatch) {
                if (currentSubtask) {
                    subtasks.push(currentSubtask);
                }
                currentSubtask = {
                    id: parseInt(numberedMatch[1]),
                    description: numberedMatch[2].trim(),
                    dependencies: [],
                    complexity: 'medium',
                    requiredTools: [],
                    successCriteria: ''
                };
            } else if (currentSubtask && line.trim()) {
                // Add details to current subtask
                if (line.toLowerCase().includes('depend')) {
                    const deps = line.match(/\d+/g);
                    if (deps) {
                        currentSubtask.dependencies = deps.map(d => parseInt(d));
                    }
                }
            }
        }
        if (currentSubtask) {
            subtasks.push(currentSubtask);
        }

        return {
            task: originalTask,
            subtasks: subtasks.length > 0 ? subtasks : [
                {
                    id: 1,
                    description: originalTask,
                    dependencies: [],
                    complexity: 'medium',
                    requiredTools: [],
                    successCriteria: 'Task completed'
                }
            ],
            estimatedTime: 'Unknown',
            requiredAgents: ['researcher', 'coder']
        };
    }

    /**
     * Execute a planned task step by step
     */
    async executeTask(taskId, agentKey = 'planner', options = {}) {
        const task = this.activeTasks.get(taskId);
        if (!task) {
            throw new Error('Task not found');
        }

        task.status = 'executing';
        const plan = task.plan;

        // Sort subtasks by dependencies (topological sort)
        const sortedSubtasks = this.sortByDependencies(plan.subtasks);

        for (let i = 0; i < sortedSubtasks.length; i++) {
            const subtask = sortedSubtasks[i];
            task.currentStep = i + 1;

            logger.info(`Executing subtask ${subtask.id}/${sortedSubtasks.length}: ${subtask.description}`);

            try {
                // Build context from previous results
                const context = this.buildExecutionContext(task, subtask);

                const executionPrompt = `Execute this subtask as part of a larger task.

Original Task: ${task.originalTask}
Current Subtask: ${subtask.description}
Success Criteria: ${subtask.successCriteria}

Previous Results:
${context}

Execute this subtask now. Provide:
1. What you did
2. Results/output
3. Any issues encountered
4. Whether success criteria was met

Be specific and actionable.`;

                const result = await this.ollamaClient(agentKey, executionPrompt);
                
                task.results.push({
                    subtaskId: subtask.id,
                    description: subtask.description,
                    result: result,
                    timestamp: new Date(),
                    success: true
                });

                // Check if we should continue
                if (options.stopOnError && result.toLowerCase().includes('error')) {
                    task.status = 'error';
                    task.errors.push(`Subtask ${subtask.id} failed: ${result}`);
                    break;
                }

            } catch (error) {
                logger.error(`Subtask ${subtask.id} failed:`, error);
                task.results.push({
                    subtaskId: subtask.id,
                    description: subtask.description,
                    result: `Error: ${error.message}`,
                    timestamp: new Date(),
                    success: false
                });
                task.errors.push(`Subtask ${subtask.id}: ${error.message}`);

                if (options.stopOnError) {
                    task.status = 'error';
                    break;
                }
            }
        }

        // Generate final synthesis
        if (task.status !== 'error') {
            task.status = 'completed';
            task.endTime = new Date();
            
            // Synthesize results
            const synthesis = await this.synthesizeResults(task);
            task.synthesis = synthesis;
        }

        return task;
    }

    /**
     * Sort subtasks by dependencies (topological sort)
     */
    sortByDependencies(subtasks) {
        const sorted = [];
        const visited = new Set();
        const visiting = new Set();

        const visit = (subtask) => {
            if (visiting.has(subtask.id)) {
                // Circular dependency detected
                logger.warn(`Circular dependency detected for subtask ${subtask.id}`);
                return;
            }
            if (visited.has(subtask.id)) {
                return;
            }

            visiting.add(subtask.id);

            // Visit dependencies first
            for (const depId of subtask.dependencies || []) {
                const dep = subtasks.find(s => s.id === depId);
                if (dep) {
                    visit(dep);
                }
            }

            visiting.delete(subtask.id);
            visited.add(subtask.id);
            sorted.push(subtask);
        };

        for (const subtask of subtasks) {
            if (!visited.has(subtask.id)) {
                visit(subtask);
            }
        }

        return sorted;
    }

    /**
     * Build execution context from previous results
     */
    buildExecutionContext(task, currentSubtask) {
        if (task.results.length === 0) {
            return 'No previous results yet.';
        }

        const relevantResults = task.results
            .filter(r => {
                // Include results from dependencies
                const deps = currentSubtask.dependencies || [];
                return deps.includes(r.subtaskId);
            })
            .map(r => `Subtask ${r.subtaskId}: ${r.description}\nResult: ${r.result}`)
            .join('\n\n---\n\n');

        return relevantResults || 'No relevant previous results.';
    }

    /**
     * Synthesize final results from all subtasks
     */
    async synthesizeResults(task) {
        const resultsSummary = task.results
            .map(r => `Subtask ${r.subtaskId}: ${r.description}\n${r.result}`)
            .join('\n\n---\n\n');

        const synthesisPrompt = `Synthesize the results from all subtasks into a comprehensive final answer.

Original Task: ${task.originalTask}

All Subtask Results:
${resultsSummary}

Create a comprehensive summary that:
1. Summarizes what was accomplished
2. Highlights key findings/results
3. Identifies any issues or limitations
4. Provides next steps or recommendations

Be thorough and organized.`;

        try {
            return await this.ollamaClient('researcher', synthesisPrompt);
        } catch (error) {
            logger.error('Synthesis failed:', error);
            return `Synthesis failed: ${error.message}\n\nAll results:\n${resultsSummary}`;
        }
    }

    /**
     * Get task status
     */
    getTask(taskId) {
        return this.activeTasks.get(taskId);
    }

    /**
     * Get all active tasks
     */
    getAllTasks() {
        return Array.from(this.activeTasks.values());
    }

    /**
     * Cancel a task
     */
    cancelTask(taskId) {
        const task = this.activeTasks.get(taskId);
        if (task && task.status === 'executing') {
            task.status = 'cancelled';
            task.endTime = new Date();
            return true;
        }
        return false;
    }
}

module.exports = TaskPlanner;

