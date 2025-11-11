package workspace

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

var (
	ErrWorkspaceNotFound = errors.New("workspace not found")
	ErrInvalidPath       = errors.New("invalid workspace path")
)

// Manager manages workspace operations
type Manager struct{}

// NewManager creates a new workspace manager
func NewManager() *Manager {
	return &Manager{}
}

// Initialize initializes a workspace directory
func (m *Manager) Initialize(workspacePath string) error {
	absPath, err := filepath.Abs(workspacePath)
	if err != nil {
		return fmt.Errorf("invalid path: %w", err)
	}

	if _, err := os.Stat(absPath); os.IsNotExist(err) {
		return ErrInvalidPath
	}

	leahPath := filepath.Join(absPath, ".leah")
	if err := os.MkdirAll(leahPath, 0755); err != nil {
		return fmt.Errorf("failed to create .leah directory: %w", err)
	}

	// Create subdirectories
	dirs := []string{"templates", "history"}
	for _, dir := range dirs {
		dirPath := filepath.Join(leahPath, dir)
		if err := os.MkdirAll(dirPath, 0755); err != nil {
			return fmt.Errorf("failed to create %s directory: %w", dir, err)
		}
	}

	// Create rules.md if not exists
	rulesPath := filepath.Join(leahPath, "rules.md")
	if _, err := os.Stat(rulesPath); os.IsNotExist(err) {
		if err := os.WriteFile(rulesPath, []byte(getRulesTemplate()), 0644); err != nil {
			return err
		}
	}

	// Create todo.md if not exists
	todoPath := filepath.Join(leahPath, "todo.md")
	if _, err := os.Stat(todoPath); os.IsNotExist(err) {
		if err := os.WriteFile(todoPath, []byte(getTodoTemplate()), 0644); err != nil {
			return err
		}
	}

	return nil
}

func getRulesTemplate() string {
	return `# AI Model Kuralları

## 🚫 YASAKLAR
- CDN kullanma
- Test yazmadan kod teslim etme

## ✅ ZORUNLU KURALLAR
- Her fonksiyon için yorum yaz
- Type safety kullan
`
}

func getTodoTemplate() string {
	return `# Todo Listesi

## Phase 1: Proje Kurulumu
- [ ] Repository oluştur
- [ ] Initial setup
`
}
