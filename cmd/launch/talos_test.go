package launch

import (
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
)

func TestTalosIntegration(t *testing.T) {
	tl := &Talos{}

	t.Run("String", func(t *testing.T) {
		if got := tl.String(); got != "Talos" {
			t.Errorf("String() = %q, want %q", got, "Talos")
		}
	})

	t.Run("implements Runner", func(t *testing.T) {
		var _ Runner = tl
	})
}

func TestTalosFindPath(t *testing.T) {
	tl := &Talos{}

	t.Run("finds talos in PATH", func(t *testing.T) {
		tmpDir := t.TempDir()
		name := "talos"
		if runtime.GOOS == "windows" {
			name = "talos.exe"
		}
		fakeBin := filepath.Join(tmpDir, name)
		os.WriteFile(fakeBin, []byte("#!/bin/sh\n"), 0o755)
		t.Setenv("PATH", tmpDir)

		got, viaPython, err := tl.findPath()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != fakeBin {
			t.Errorf("findPath() = %q, want %q", got, fakeBin)
		}
		if viaPython {
			t.Error("a wrapper on PATH should not be invoked through -m talos")
		}
	})

	t.Run("falls back to the virtual environment under the install prefix", func(t *testing.T) {
		prefix := t.TempDir()
		bin := filepath.Join(prefix, ".venv", "bin")
		name := "python"
		if runtime.GOOS == "windows" {
			bin = filepath.Join(prefix, ".venv", "Scripts")
			name = "python.exe"
		}
		if err := os.MkdirAll(bin, 0o755); err != nil {
			t.Fatal(err)
		}
		python := filepath.Join(bin, name)
		os.WriteFile(python, []byte("#!/bin/sh\n"), 0o755)
		t.Setenv("TALOS_PREFIX", prefix)
		t.Setenv("PATH", t.TempDir()) // empty dir, no talos wrapper

		got, viaPython, err := tl.findPath()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != python {
			t.Errorf("findPath() = %q, want %q", got, python)
		}
		if !viaPython {
			t.Error("the interpreter must be invoked through -m talos")
		}
	})

	t.Run("returns error when not installed", func(t *testing.T) {
		t.Setenv("TALOS_PREFIX", t.TempDir()) // empty prefix, no .venv
		t.Setenv("PATH", t.TempDir())         // empty dir, no talos wrapper

		if _, _, err := tl.findPath(); err == nil {
			t.Fatal("expected error, got nil")
		}
	})
}

func TestTalosArgs(t *testing.T) {
	tl := &Talos{}

	tests := []struct {
		name      string
		viaPython bool
		args      []string
		want      []string
	}{
		{"wrapper on PATH", false, nil, []string{"chat"}},
		{"through the interpreter", true, nil, []string{"-m", "talos", "chat"}},
		{"extra args are passed through", false, []string{"--verbose"}, []string{"chat", "--verbose"}},
		{"interpreter with extra args", true, []string{"--verbose"}, []string{"-m", "talos", "chat", "--verbose"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tl.args(tt.viaPython, tt.args)
			if !slices.Equal(got, tt.want) {
				t.Errorf("args(%v, %v) = %v, want %v", tt.viaPython, tt.args, got, tt.want)
			}
		})
	}
}

func TestTalosEnvVars(t *testing.T) {
	tl := &Talos{}

	envMap := func(envs []string) map[string]string {
		m := make(map[string]string)
		for _, e := range envs {
			k, v, _ := strings.Cut(e, "=")
			m[k] = v
		}
		return m
	}

	t.Run("points Talos at Ollama through its OpenAI-compatible provider", func(t *testing.T) {
		got := envMap(tl.envVars("llama3.2"))
		if got["TALOS_MODEL_PROVIDER"] != "openai-api" {
			t.Errorf("TALOS_MODEL_PROVIDER = %q, want %q", got["TALOS_MODEL_PROVIDER"], "openai-api")
		}
		if got["TALOS_BASE_URL_OPENAI_API"] == "" {
			t.Error("TALOS_BASE_URL_OPENAI_API should be set")
		}
		if !strings.HasSuffix(got["TALOS_BASE_URL_OPENAI_API"], "/v1") {
			t.Errorf("TALOS_BASE_URL_OPENAI_API = %q, want /v1 suffix", got["TALOS_BASE_URL_OPENAI_API"])
		}
		if got["TALOS_MODEL"] != "llama3.2" {
			t.Errorf("TALOS_MODEL = %q, want %q", got["TALOS_MODEL"], "llama3.2")
		}
	})

	t.Run("omits TALOS_MODEL when model is empty", func(t *testing.T) {
		got := envMap(tl.envVars(""))
		if _, ok := got["TALOS_MODEL"]; ok {
			t.Errorf("TALOS_MODEL should not be set for empty model, got %q", got["TALOS_MODEL"])
		}
	})

	t.Run("uses custom OLLAMA_HOST", func(t *testing.T) {
		t.Setenv("OLLAMA_HOST", "http://myhost:9999")
		got := envMap(tl.envVars("test"))
		if !strings.Contains(got["TALOS_BASE_URL_OPENAI_API"], "myhost:9999") {
			t.Errorf("TALOS_BASE_URL_OPENAI_API = %q, want custom host", got["TALOS_BASE_URL_OPENAI_API"])
		}
	})

	t.Run("translates a wildcard bind address to a connectable one", func(t *testing.T) {
		t.Setenv("OLLAMA_HOST", "http://0.0.0.0:11434")
		got := envMap(tl.envVars("test"))
		if strings.Contains(got["TALOS_BASE_URL_OPENAI_API"], "0.0.0.0") {
			t.Errorf("TALOS_BASE_URL_OPENAI_API = %q, must not pass a wildcard bind address through", got["TALOS_BASE_URL_OPENAI_API"])
		}
		if !strings.Contains(got["TALOS_BASE_URL_OPENAI_API"], "127.0.0.1:11434") {
			t.Errorf("TALOS_BASE_URL_OPENAI_API = %q, want loopback with the same port", got["TALOS_BASE_URL_OPENAI_API"])
		}
	})
}
