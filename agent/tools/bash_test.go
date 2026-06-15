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

func TestBashBoundsOutputWhileRunning(t *testing.T) {
	result, err := NewBash().Execute(context.Background(), agent.ToolContext{WorkingDir: t.TempDir()}, map[string]any{
		"command": "yes x | head -c 70000",
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.Content, "[stdout truncated: omitted ") {
		t.Fatalf("content = %q, want stdout truncation marker", result.Content)
	}
	if count := strings.Count(result.Content, "x"); count != maxBashOutputBytes/2 {
		t.Fatalf("captured x count = %d, want %d", count, maxBashOutputBytes/2)
	}
	if len(result.Content) > maxBashOutputBytes+200 {
		t.Fatalf("content length = %d, want bounded output", len(result.Content))
	}
}

func TestReadFinalWorkingDirRejectsInvalidPaths(t *testing.T) {
	dir := t.TempDir()
	cwdFile := filepath.Join(dir, "cwd")
	notDir := filepath.Join(dir, "file.txt")
	if err := os.WriteFile(notDir, []byte("not a dir"), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := os.WriteFile(cwdFile, []byte(notDir+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := readFinalWorkingDir(cwdFile); got != "" {
		t.Fatalf("regular file cwd = %q, want empty", got)
	}

	if err := os.WriteFile(cwdFile, []byte(filepath.Join(dir, "missing")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := readFinalWorkingDir(cwdFile); got != "" {
		t.Fatalf("missing cwd = %q, want empty", got)
	}

	if err := os.WriteFile(cwdFile, []byte(dir+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := readFinalWorkingDir(cwdFile); got != dir {
		t.Fatalf("directory cwd = %q, want %q", got, dir)
	}
}
