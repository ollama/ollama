package launch

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

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
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

	o := &OMP{}
	if err := o.Run("", []string{"--headless", "--try"}); err != nil {
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
	if !strings.Contains(got, "args=--headless --try") {
		t.Fatalf("expected passthrough args in log, got:\n%s", got)
	}
}
