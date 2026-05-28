package launch

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestOMPArgs(t *testing.T) {
	o := &OMP{}

	tests := []struct {
		name  string
		model string
		extra []string
		want  []string
	}{
		{"with model", "qwen3.5", nil, []string{"--model", "ollama/qwen3.5"}},
		{"empty model", "", nil, nil},
		{"preserves Ollama prefix", "ollama/qwen3.5", []string{"--debug"}, []string{"--model", "ollama/qwen3.5", "--debug"}},
		{"with extra args", "llama3.2", []string{"--headless", "--try"}, []string{"--model", "ollama/llama3.2", "--headless", "--try"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := o.args(tt.model, tt.extra)
			if strings.Join(got, "\x00") != strings.Join(tt.want, "\x00") {
				t.Fatalf("args() = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestOMPRunSetsOllamaEnv(t *testing.T) {
	tmpDir := t.TempDir()
	logPath := filepath.Join(tmpDir, "omp.log")
	ompPath := filepath.Join(tmpDir, "omp")
	script := "#!/bin/sh\n" +
		"printf 'base=%s\\nkey=%s\\nargs=%s\\n' \"$OPENAI_BASE_URL\" \"$OPENAI_API_KEY\" \"$*\" > \"" + logPath + "\"\n"
	if err := os.WriteFile(ompPath, []byte(script), 0o755); err != nil {
		t.Fatalf("failed to write fake omp binary: %v", err)
	}

	t.Setenv("PATH", tmpDir)
	t.Setenv("OLLAMA_HOST", "http://0.0.0.0:11434")

	o := &OMP{}
	if err := o.Run("qwen3.5", []string{"--headless", "--try"}); err != nil {
		t.Fatalf("Run returned error: %v", err)
	}

	data, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatalf("failed to read omp log: %v", err)
	}

	got := string(data)
	if !strings.Contains(got, "base=http://127.0.0.1:11434/v1") {
		t.Fatalf("expected OPENAI_BASE_URL override in log, got:\n%s", got)
	}
	if !strings.Contains(got, "key=ollama") {
		t.Fatalf("expected OPENAI_API_KEY override in log, got:\n%s", got)
	}
	if !strings.Contains(got, "args=--model ollama/qwen3.5 --headless --try") {
		t.Fatalf("expected passthrough args in log, got:\n%s", got)
	}
}
