package tools

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/agent/skills"
)

func TestSkillToolLoadsSkill(t *testing.T) {
	dir := t.TempDir()
	skillDir := filepath.Join(dir, "go-code")
	if err := os.MkdirAll(skillDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(skillDir, skills.SkillFile), []byte("---\nname: go-code\ndescription: Write Go code.\n---\n\n# Go Code\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	catalog, err := skills.Load(dir)
	if err != nil {
		t.Fatal(err)
	}

	result, err := NewSkill(catalog).Execute(context.Background(), agent.ToolContext{}, map[string]any{"name": "go-code"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.Content, "Loaded skill: go-code") || !strings.Contains(result.Content, "# Go Code") {
		t.Fatalf("content = %q", result.Content)
	}
}
