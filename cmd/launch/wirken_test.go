package launch

import (
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
)

func TestWirkenArgs(t *testing.T) {
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
	w := &Wirken{}

	tests := []struct {
		name  string
		model string
		extra []string
		want  []string
	}{
		{name: "with model", model: "glm-5.3:cloud", want: []string{"run", "--provider", "ollama", "--model", "glm-5.3:cloud", "--base-url", "http://127.0.0.1:11434/v1"}},
		{name: "without model", extra: []string{"--port", "18791"}, want: []string{"run", "--port", "18791"}},
		{name: "with model and extra args", model: "qwen3.5", extra: []string{"--port", "18791"}, want: []string{"run", "--provider", "ollama", "--model", "qwen3.5", "--base-url", "http://127.0.0.1:11434/v1", "--port", "18791"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := w.args(tt.model, tt.extra)
			if !slices.Equal(got, tt.want) {
				t.Fatalf("args(%q, %v) = %v, want %v", tt.model, tt.extra, got, tt.want)
			}
		})
	}
}

// writeFakeWirken writes a wirken stand-in that appends each invocation's
// arguments to logPath, one line per call.
func writeFakeWirken(t *testing.T, dir, logPath string) {
	t.Helper()
	script := "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"" + logPath + "\"\n"
	if err := os.WriteFile(filepath.Join(dir, "wirken"), []byte(script), 0o755); err != nil {
		t.Fatalf("failed to write fake wirken binary: %v", err)
	}
}

func TestWirkenRunPassesModelFlags(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell fake binary")
	}

	binDir := t.TempDir()
	logPath := filepath.Join(binDir, "wirken.log")
	writeFakeWirken(t, binDir, logPath)

	dataDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dataDir, "provider.json"), []byte("{}"), 0o600); err != nil {
		t.Fatal(err)
	}

	t.Setenv("PATH", binDir)
	t.Setenv("WIRKEN_DATA_DIR", dataDir)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

	w := &Wirken{}
	if err := w.Run("glm-5.3:cloud", nil, []string{"--port", "18791"}); err != nil {
		t.Fatalf("Run returned error: %v", err)
	}

	data, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatalf("failed to read wirken log: %v", err)
	}
	got := strings.TrimSpace(string(data))
	want := "run --provider ollama --model glm-5.3:cloud --base-url http://127.0.0.1:11434/v1 --port 18791"
	if got != want {
		t.Fatalf("expected a single run invocation\nwant: %s\ngot:  %s", want, got)
	}
}

func TestWirkenRunSetsUpFirst(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell fake binary")
	}

	binDir := t.TempDir()
	logPath := filepath.Join(binDir, "wirken.log")
	writeFakeWirken(t, binDir, logPath)

	t.Setenv("PATH", binDir)
	t.Setenv("WIRKEN_DATA_DIR", t.TempDir())
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

	w := &Wirken{}
	if err := w.Run("qwen3.5", nil, nil); err != nil {
		t.Fatalf("Run returned error: %v", err)
	}

	data, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatalf("failed to read wirken log: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 2 || lines[0] != "setup" || !strings.HasPrefix(lines[1], "run --provider ollama --model qwen3.5") {
		t.Fatalf("expected setup then run, got:\n%s", data)
	}
}

func TestWirkenRunNotInstalled(t *testing.T) {
	t.Setenv("PATH", t.TempDir())

	w := &Wirken{}
	err := w.Run("qwen3.5", nil, nil)
	if err == nil || !strings.Contains(err.Error(), "wirken is not installed") {
		t.Fatalf("expected not-installed error, got %v", err)
	}
}
