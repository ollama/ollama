package launch

import (
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
)

func TestPoolsideArgs(t *testing.T) {
	p := &Poolside{}

	tests := []struct {
		name  string
		model string
		extra []string
		want  []string
	}{
		{name: "with model", model: "qwen3.5", want: []string{"-m", "qwen3.5"}},
		{name: "without model", extra: []string{"session"}, want: []string{"session"}},
		{name: "with model and extra args", model: "llama3.2", extra: []string{"--foo", "bar"}, want: []string{"-m", "llama3.2", "--foo", "bar"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := p.args(tt.model, tt.extra)
			if !slices.Equal(got, tt.want) {
				t.Fatalf("args(%q, %v) = %v, want %v", tt.model, tt.extra, got, tt.want)
			}
		})
	}
}

func TestPoolsideRunSetsOllamaEnv(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell fake binary")
	}

	tmpDir := t.TempDir()
	logPath := filepath.Join(tmpDir, "pool.log")
	poolPath := filepath.Join(tmpDir, "pool")
	script := "#!/bin/sh\n" +
		"printf 'base=%s\\nkey=%s\\nargs=%s\\n' \"$POOLSIDE_STANDALONE_BASE_URL\" \"$POOLSIDE_API_KEY\" \"$*\" > \"" + logPath + "\"\n"
	if err := os.WriteFile(poolPath, []byte(script), 0o755); err != nil {
		t.Fatalf("failed to write fake pool binary: %v", err)
	}

	t.Setenv("PATH", tmpDir)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

	p := &Poolside{}
	if err := p.Run("qwen3.5", []string{"session"}); err != nil {
		t.Fatalf("Run returned error: %v", err)
	}

	data, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatalf("failed to read pool log: %v", err)
	}

	got := string(data)
	if !strings.Contains(got, "base=http://127.0.0.1:11434/v1") {
		t.Fatalf("expected Poolside base URL override in log, got:\n%s", got)
	}
	if !strings.Contains(got, "key=ollama") {
		t.Fatalf("expected Poolside API key override in log, got:\n%s", got)
	}
	if !strings.Contains(got, "args=-m qwen3.5 session") {
		t.Fatalf("expected model and extra args in log, got:\n%s", got)
	}
}

func TestPoolsideRunWindowsUnsupported(t *testing.T) {
	prev := poolsideGOOS
	poolsideGOOS = "windows"
	t.Cleanup(func() { poolsideGOOS = prev })

	p := &Poolside{}
	err := p.Run("kimi-k2.6:cloud", nil)
	if err == nil {
		t.Fatal("expected Windows unsupported error")
	}
	if !strings.Contains(err.Error(), "not currently supported on Windows") {
		t.Fatalf("expected Windows warning, got %v", err)
	}
}
