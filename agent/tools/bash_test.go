package tools

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
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

	result, err := (&Bash{}).Execute(context.Background(), agent.ToolContext{WorkingDir: root}, map[string]any{
		"command": shellTestCommand("cd sub && pwd", "Set-Location sub; Get-Location"),
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
	result, err := (&Bash{}).Execute(context.Background(), agent.ToolContext{WorkingDir: t.TempDir()}, map[string]any{
		"command": shellTestCommand("yes x | head -c 70000", "[Console]::Out.Write(('x' * 70000))"),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.Content, "[stdout truncated: omitted ~") || !strings.Contains(result.Content, " tokens]") {
		t.Fatalf("content = %q, want stdout truncation marker", result.Content)
	}
	if count, want := strings.Count(result.Content, "x"), shellTestCapturedXCount(); count != want {
		t.Fatalf("captured x count = %d, want %d", count, want)
	}
	if len(result.Content) > maxBashOutputBytes+200 {
		t.Fatalf("content length = %d, want bounded output", len(result.Content))
	}
}

func TestBashReportsCanceledCommand(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	result, err := (&Bash{}).Execute(ctx, agent.ToolContext{WorkingDir: t.TempDir()}, map[string]any{
		"command": shellTestCommand("sleep 10", "Start-Sleep -Seconds 10"),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.Content, "Error: command was canceled") {
		t.Fatalf("content = %q, want canceled message", result.Content)
	}
	if strings.Contains(result.Content, "Exit code: -1") {
		t.Fatalf("content = %q, should not mask cancellation as exit code", result.Content)
	}
}

func shellTestCommand(unix, windows string) string {
	if runtime.GOOS == "windows" {
		return windows
	}
	return unix
}

func shellTestCapturedXCount() int {
	if runtime.GOOS == "windows" {
		return maxBashOutputBytes
	}
	return maxBashOutputBytes / 2
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

func TestNormalizeBashWorkingDirWindowsDriveLetter(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("windows path normalization")
	}
	got := normalizeBashWorkingDir("/c/Users/jdoe/project")
	want := filepath.Clean(`C:\Users\jdoe\project`)
	if got != want {
		t.Fatalf("working dir = %q, want %q", got, want)
	}
}
