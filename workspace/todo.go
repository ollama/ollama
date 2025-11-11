package workspace

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// TodoManager manages todo.md file
type TodoManager struct {
	manager *Manager
}

// NewTodoManager creates todo manager
func NewTodoManager(m *Manager) *TodoManager {
	return &TodoManager{manager: m}
}

// TodoList represents parsed todo list
type TodoList struct {
	Phases []*Phase `json:"phases"`
}

// Phase represents a phase in todo list
type Phase struct {
	Name   string  `json:"name"`
	Status string  `json:"status"` // pending, in_progress, completed, failed
	Tasks  []*Task `json:"tasks"`
}

// Task represents a task
type Task struct {
	Text      string `json:"text"`
	Completed bool   `json:"completed"`
}

// GetTodos parses todo.md
func (tm *TodoManager) GetTodos(workspacePath string) (*TodoList, error) {
	todoPath := filepath.Join(workspacePath, ".leah", "todo.md")

	content, err := os.ReadFile(todoPath)
	if err != nil {
		return nil, err
	}

	return tm.parseTodos(string(content)), nil
}

// parseTodos parses todo.md content
func (tm *TodoManager) parseTodos(content string) *TodoList {
	todoList := &TodoList{
		Phases: make([]*Phase, 0),
	}

	scanner := bufio.NewScanner(strings.NewReader(content))
	var currentPhase *Phase

	for scanner.Scan() {
		line := scanner.Text()
		trimmed := strings.TrimSpace(line)

		// Detect phase headers
		if strings.HasPrefix(trimmed, "## ") {
			if currentPhase != nil {
				todoList.Phases = append(todoList.Phases, currentPhase)
			}

			phaseName := strings.TrimPrefix(trimmed, "## ")
			currentPhase = &Phase{
				Name:   phaseName,
				Status: "pending",
				Tasks:  make([]*Task, 0),
			}
			continue
		}

		if currentPhase == nil {
			continue
		}

		// Parse tasks
		if strings.HasPrefix(trimmed, "- [") {
			task := parseTask(trimmed)
			if task != nil {
				currentPhase.Tasks = append(currentPhase.Tasks, task)
			}
		}
	}

	// Add last phase
	if currentPhase != nil {
		todoList.Phases = append(todoList.Phases, currentPhase)
	}

	return todoList
}

func parseTask(line string) *Task {
	trimmed := strings.TrimSpace(line)
	if !strings.HasPrefix(trimmed, "- [") {
		return nil
	}

	completed := strings.HasPrefix(trimmed, "- [x]") || strings.HasPrefix(trimmed, "- [X]")
	text := ""

	if completed {
		text = strings.TrimPrefix(trimmed, "- [x] ")
		text = strings.TrimPrefix(text, "- [X] ")
	} else {
		text = strings.TrimPrefix(trimmed, "- [ ] ")
	}

	return &Task{
		Text:      strings.TrimSpace(text),
		Completed: completed,
	}
}

// UpdateTodos updates todo.md file
func (tm *TodoManager) UpdateTodos(workspacePath string, content string) error {
	todoPath := filepath.Join(workspacePath, ".leah", "todo.md")
	return os.WriteFile(todoPath, []byte(content), 0644)
}

// MarkTaskComplete marks a task as complete
func (tm *TodoManager) MarkTaskComplete(workspacePath string, phaseIndex, taskIndex int) error {
	todos, err := tm.GetTodos(workspacePath)
	if err != nil {
		return err
	}

	if phaseIndex >= len(todos.Phases) {
		return fmt.Errorf("invalid phase index")
	}

	phase := todos.Phases[phaseIndex]
	if taskIndex >= len(phase.Tasks) {
		return fmt.Errorf("invalid task index")
	}

	phase.Tasks[taskIndex].Completed = true

	// Check if all tasks completed
	allCompleted := true
	for _, task := range phase.Tasks {
		if !task.Completed {
			allCompleted = false
			break
		}
	}

	if allCompleted {
		phase.Status = "completed"
	}

	// Regenerate todo.md
	return tm.saveTodos(workspacePath, todos)
}

// saveTodos saves todo list back to file
func (tm *TodoManager) saveTodos(workspacePath string, todos *TodoList) error {
	var content strings.Builder

	content.WriteString("# Proje Todo Listesi\n\n")
	content.WriteString(fmt.Sprintf("**Last Update:** %s\n\n", time.Now().Format("2006-01-02 15:04:05")))
	content.WriteString("---\n\n")

	for _, phase := range todos.Phases {
		content.WriteString(fmt.Sprintf("## %s\n\n", phase.Name))

		for _, task := range phase.Tasks {
			checkbox := "[ ]"
			if task.Completed {
				checkbox = "[x]"
			}
			content.WriteString(fmt.Sprintf("- %s %s\n", checkbox, task.Text))
		}
		content.WriteString("\n")
	}

	return tm.UpdateTodos(workspacePath, content.String())
}
