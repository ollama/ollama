package agent

import (
	"context"
	"fmt"

	"github.com/ollama/ollama/api/providers"
	"github.com/ollama/ollama/workspace"
)

// Controller manages agent execution with dual-model architecture
type Controller struct {
	supervisor providers.Provider
	worker     providers.Provider
	todoMgr    *workspace.TodoManager
	rulesMgr   *workspace.RulesManager
}

// Session represents an agent session
type Session struct {
	ID           string
	WorkspacePath string
	Status       string
	CurrentPhase int
}

// NewController creates a new agent controller
func NewController(supervisor, worker providers.Provider, todoMgr *workspace.TodoManager, rulesMgr *workspace.RulesManager) *Controller {
	return &Controller{
		supervisor: supervisor,
		worker:     worker,
		todoMgr:    todoMgr,
		rulesMgr:   rulesMgr,
	}
}

// StartSession starts an agent session
func (ac *Controller) StartSession(ctx context.Context, workspacePath string) (*Session, error) {
	// Load todo list
	todos, err := ac.todoMgr.GetTodos(workspacePath)
	if err != nil {
		return nil, err
	}

	// Load rules
	rules, err := ac.rulesMgr.GetRules(workspacePath)
	if err != nil {
		return nil, err
	}

	session := &Session{
		ID:            fmt.Sprintf("session-%d", 1),
		WorkspacePath: workspacePath,
		Status:        "running",
		CurrentPhase:  0,
	}

	// Execute phases
	for phaseIdx, phase := range todos.Phases {
		session.CurrentPhase = phaseIdx

		// Supervisor: Plan the phase
		supervisorPlan, err := ac.supervisorPlanPhase(ctx, phase, rules)
		if err != nil {
			return session, err
		}

		// Worker: Execute the plan
		_, err = ac.workerExecute(ctx, supervisorPlan, rules)
		if err != nil {
			return session, err
		}

		// Mark phase as complete
		ac.todoMgr.MarkTaskComplete(workspacePath, phaseIdx, 0)
	}

	session.Status = "completed"
	return session, nil
}

// supervisorPlanPhase creates execution plan
func (ac *Controller) supervisorPlanPhase(ctx context.Context, phase *workspace.Phase, rules *workspace.Rules) (string, error) {
	prompt := fmt.Sprintf(`You are a supervisor AI. Plan the execution of this phase:

Phase: %s

Tasks:
%s

Rules to follow:
%s

Create a detailed execution plan.`, phase.Name, formatTasks(phase.Tasks), rules.ToSystemPrompt())

	req := providers.ChatRequest{
		Model: "claude-sonnet-4-5",
		Messages: []providers.Message{
			{Role: "user", Content: prompt},
		},
	}

	resp, err := ac.supervisor.ChatCompletion(ctx, req)
	if err != nil {
		return "", err
	}

	return resp.Message.Content, nil
}

// workerExecute executes the plan
func (ac *Controller) workerExecute(ctx context.Context, plan string, rules *workspace.Rules) (string, error) {
	prompt := fmt.Sprintf(`Execute the following plan:

%s

Rules:
%s

Provide the implementation.`, plan, rules.ToSystemPrompt())

	req := providers.ChatRequest{
		Model: "gpt-4",
		Messages: []providers.Message{
			{Role: "user", Content: prompt},
		},
	}

	resp, err := ac.worker.ChatCompletion(ctx, req)
	if err != nil {
		return "", err
	}

	return resp.Message.Content, nil
}

func formatTasks(tasks []*workspace.Task) string {
	result := ""
	for _, task := range tasks {
		checkbox := "[ ]"
		if task.Completed {
			checkbox = "[x]"
		}
		result += fmt.Sprintf("- %s %s\n", checkbox, task.Text)
	}
	return result
}
