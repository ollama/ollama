package tools

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/agent"
)

func TestBashReportsFinalWorkingDir(t *testing.T) {
	root := t.TempDir()
	subdir := filepath.Join(root, "sub")
	if err := os.Mkdir(subdir, 0o755); err != nil {
		t.Fatal(err)
	}

	result, err := NewBash().Execute(context.Background(), agent.ToolContext{WorkingDir: root}, map[string]any{
		"command": "cd sub && pwd",
	})
	if err != nil {
		t.Fatal(err)
	}
	wantDir, err := filepath.EvalSymlinks(subdir)
	if err != nil {
		t.Fatal(err)
	}
	if result.WorkingDir != wantDir {
		t.Fatalf("working dir = %q, want %q", result.WorkingDir, wantDir)
	}
	if !strings.Contains(result.Content, "sub") {
		t.Fatalf("content = %q, want pwd output", result.Content)
	}
}
