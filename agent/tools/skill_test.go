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

func TestManualSkillMessagesUseToolCallShape(t *testing.T) {
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
	skill, ok := catalog.Find("go-code")
	if !ok {
		t.Fatal("skill not found")
	}

	messages, err := ManualSkillMessages(skill, "write a test", 7)
	if err != nil {
		t.Fatal(err)
	}
	if len(messages) != 3 {
		t.Fatalf("messages = %d, want 3", len(messages))
	}
	if messages[0].Role != "user" || messages[0].Content != "write a test" {
		t.Fatalf("user message = %#v", messages[0])
	}
	if messages[1].Role != "assistant" || len(messages[1].ToolCalls) != 1 {
		t.Fatalf("assistant tool call = %#v", messages[1])
	}
	call := messages[1].ToolCalls[0]
	if call.ID != "manual-skill-7-go-code" || call.Function.Name != "skill" {
		t.Fatalf("tool call = %#v", call)
	}
	if name, _ := call.Function.Arguments.Get("name"); name != "go-code" {
		t.Fatalf("tool args = %s", call.Function.Arguments.String())
	}
	if messages[2].Role != "tool" || messages[2].ToolName != "skill" || messages[2].ToolCallID != call.ID {
		t.Fatalf("tool result metadata = %#v", messages[2])
	}
	if !strings.Contains(messages[2].Content, "Loaded skill: go-code") || !strings.Contains(messages[2].Content, "# Go Code") {
		t.Fatalf("tool result = %q", messages[2].Content)
	}
}
