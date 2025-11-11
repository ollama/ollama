package agent

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/ollama/ollama/api/providers"
	"github.com/ollama/ollama/workspace"
)

// Controller manages agent execution with dual-model architecture
type Controller struct {
	supervisor providers.Provider
	worker     providers.Provider
	todoMgr    *workspace.TodoManager
	rulesMgr   *workspace.RulesManager
	sessions   map[string]*Session
	sessionsMu sync.RWMutex
}

// Session represents an agent session
type Session struct {
	ID              string                 `json:"id"`
	WorkspacePath   string                 `json:"workspace_path"`
	Status          string                 `json:"status"` // pending, running, completed, failed, paused
	CurrentPhase    int                    `json:"current_phase"`
	CurrentTask     string                 `json:"current_task"`
	TotalPhases     int                    `json:"total_phases"`
	CompletedPhases int                    `json:"completed_phases"`
	StartedAt       time.Time              `json:"started_at"`
	CompletedAt     *time.Time             `json:"completed_at,omitempty"`
	Error           string                 `json:"error,omitempty"`
	Logs            []SessionLog           `json:"logs"`
	Metadata        map[string]interface{} `json:"metadata,omitempty"`
}

// SessionLog represents a log entry for a session
type SessionLog struct {
	Timestamp time.Time `json:"timestamp"`
	Level     string    `json:"level"` // info, warning, error
	Message   string    `json:"message"`
	Phase     int       `json:"phase"`
}

// NewController creates a new agent controller
func NewController(supervisor, worker providers.Provider, todoMgr *workspace.TodoManager, rulesMgr *workspace.RulesManager) *Controller {
	return &Controller{
		supervisor: supervisor,
		worker:     worker,
		todoMgr:    todoMgr,
		rulesMgr:   rulesMgr,
		sessions:   make(map[string]*Session),
	}
}

// GetSession retrieves a session by ID
func (ac *Controller) GetSession(sessionID string) (*Session, error) {
	ac.sessionsMu.RLock()
	defer ac.sessionsMu.RUnlock()

	session, ok := ac.sessions[sessionID]
	if !ok {
		return nil, fmt.Errorf("session not found: %s", sessionID)
	}

	return session, nil
}

// ListSessions returns all sessions
func (ac *Controller) ListSessions() []*Session {
	ac.sessionsMu.RLock()
	defer ac.sessionsMu.RUnlock()

	sessions := make([]*Session, 0, len(ac.sessions))
	for _, session := range ac.sessions {
		sessions = append(sessions, session)
	}

	return sessions
}

// PauseSession pauses a running session
func (ac *Controller) PauseSession(sessionID string) error {
	ac.sessionsMu.Lock()
	defer ac.sessionsMu.Unlock()

	session, ok := ac.sessions[sessionID]
	if !ok {
		return fmt.Errorf("session not found: %s", sessionID)
	}

	if session.Status != "running" {
		return fmt.Errorf("session is not running")
	}

	session.Status = "paused"
	ac.addLog(session, "info", "Session paused by user")

	return nil
}

// ResumeSession resumes a paused session
func (ac *Controller) ResumeSession(sessionID string) error {
	ac.sessionsMu.Lock()
	defer ac.sessionsMu.Unlock()

	session, ok := ac.sessions[sessionID]
	if !ok {
		return fmt.Errorf("session not found: %s", sessionID)
	}

	if session.Status != "paused" {
		return fmt.Errorf("session is not paused")
	}

	session.Status = "running"
	ac.addLog(session, "info", "Session resumed")

	return nil
}

// addLog adds a log entry to the session
func (ac *Controller) addLog(session *Session, level, message string) {
	log := SessionLog{
		Timestamp: time.Now(),
		Level:     level,
		Message:   message,
		Phase:     session.CurrentPhase,
	}
	session.Logs = append(session.Logs, log)
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

	// Create session
	session := &Session{
		ID:              fmt.Sprintf("session-%d", time.Now().Unix()),
		WorkspacePath:   workspacePath,
		Status:          "running",
		CurrentPhase:    0,
		TotalPhases:     len(todos.Phases),
		CompletedPhases: 0,
		StartedAt:       time.Now(),
		Logs:            make([]SessionLog, 0),
		Metadata:        make(map[string]interface{}),
	}

	// Store session
	ac.sessionsMu.Lock()
	ac.sessions[session.ID] = session
	ac.sessionsMu.Unlock()

	ac.addLog(session, "info", "Session started")

	// Execute phases in goroutine
	go func() {
		defer func() {
			if r := recover(); r != nil {
				session.Status = "failed"
				session.Error = fmt.Sprintf("panic: %v", r)
				ac.addLog(session, "error", session.Error)
			}
		}()

		for phaseIdx, phase := range todos.Phases {
			session.CurrentPhase = phaseIdx
			session.CurrentTask = phase.Name
			ac.addLog(session, "info", fmt.Sprintf("Starting phase: %s", phase.Name))

			// Supervisor: Plan the phase
			supervisorPlan, err := ac.supervisorPlanPhase(ctx, phase, rules)
			if err != nil {
				session.Status = "failed"
				session.Error = err.Error()
				ac.addLog(session, "error", fmt.Sprintf("Planning failed: %s", err.Error()))
				return
			}

			ac.addLog(session, "info", "Plan created, executing...")

			// Worker: Execute the plan
			_, err = ac.workerExecute(ctx, supervisorPlan, rules)
			if err != nil {
				session.Status = "failed"
				session.Error = err.Error()
				ac.addLog(session, "error", fmt.Sprintf("Execution failed: %s", err.Error()))
				return
			}

			// Mark phase as complete
			ac.todoMgr.MarkTaskComplete(workspacePath, phaseIdx, 0)
			session.CompletedPhases++
			ac.addLog(session, "info", fmt.Sprintf("Phase completed: %s", phase.Name))
		}

		session.Status = "completed"
		now := time.Now()
		session.CompletedAt = &now
		ac.addLog(session, "info", "All phases completed successfully")
	}()

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
