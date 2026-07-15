package tools

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/agent"
)

func TestSkillLoadsCoreCatalogWithoutApproval(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "release-notes")
	if err := os.Mkdir(path, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(path, "SKILL.md"), []byte("Use concise bullets."), 0o644); err != nil {
		t.Fatal(err)
	}
	catalog, err := agent.DiscoverSkills(dir)
	if err != nil {
		t.Fatal(err)
	}
	tool := &Skill{Catalog: catalog}
	if agent.ToolRequiresApproval(tool, map[string]any{"name": "release-notes"}) {
		t.Fatal("loading a skill must not change ordinary tool approval semantics")
	}
	result, err := tool.Execute(context.Background(), agent.ToolContext{}, map[string]any{"name": "release-notes"})
	if err != nil || !strings.Contains(result.Content, "Use concise bullets.") {
		t.Fatalf("tool result = %#v, %v", result, err)
	}
}
