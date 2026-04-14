package launch

import (
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
)

func TestCopilotIntegration(t *testing.T) {
	c := &Copilot{}

	t.Run("String", func(t *testing.T) {
		if got := c.String(); got != "Copilot CLI" {
			t.Errorf("String() = %q, want %q", got, "Copilot CLI")
		}
	})

	t.Run("implements Runner", func(t *testing.T) {
		var _ Runner = c
	})
}

func TestCopilotFindPath(t *testing.T) {
	c := &Copilot{}

	t.Run("finds copilot in PATH", func(t *testing.T) {
		tmpDir := t.TempDir()
		name := "copilot"
		if runtime.GOOS == "windows" {
			name = "copilot.exe"
		}
		fakeBin := filepath.Join(tmpDir, name)
		os.WriteFile(fakeBin, []byte("#!/bin/sh\n"), 0o755)
		t.Setenv("PATH", tmpDir)

		got, err := c.findPath()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != fakeBin {
			t.Errorf("findPath() = %q, want %q", got, fakeBin)
		}
	})

	t.Run("returns error when not in PATH", func(t *testing.T) {
		t.Setenv("PATH", t.TempDir()) // empty dir, no copilot binary

		_, err := c.findPath()
		if err == nil {
			t.Fatal("expected error, got nil")
		}
	})

	t.Run("falls back to ~/.local/bin/copilot", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		t.Setenv("PATH", t.TempDir()) // empty dir, no copilot binary

		name := "copilot"
		if runtime.GOOS == "windows" {
			name = "copilot.exe"
		}
		fallback := filepath.Join(tmpDir, ".local", "bin", name)
		os.MkdirAll(filepath.Dir(fallback), 0o755)
		os.WriteFile(fallback, []byte("#!/bin/sh\n"), 0o755)

		got, err := c.findPath()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != fallback {
			t.Errorf("findPath() = %q, want %q", got, fallback)
		}
	})

	t.Run("returns error when neither PATH nor fallback exists", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		t.Setenv("PATH", t.TempDir()) // empty dir, no copilot binary

		_, err := c.findPath()
		if err == nil {
			t.Fatal("expected error, got nil")
		}
	})
}

func TestCopilotArgs(t *testing.T) {
	c := &Copilot{}

	tests := []struct {
		name  string
		model string
		args  []string
		want  []string
	}{
		{"with model", "llama3.2", nil, []string{"--model", "llama3.2"}},
		{"empty model", "", nil, nil},
		{"with model and extra", "llama3.2", []string{"--verbose"}, []string{"--model", "llama3.2", "--verbose"}},
		{"empty model with help", "", []string{"--help"}, []string{"--help"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := c.args(tt.model, tt.args)
			if !slices.Equal(got, tt.want) {
				t.Errorf("args(%q, %v) = %v, want %v", tt.model, tt.args, got, tt.want)
			}
		})
	}
}

func TestCopilotEnvVars(t *testing.T) {
	c := &Copilot{}

	envMap := func(envs []string) map[string]string {
		m := make(map[string]string)
		for _, e := range envs {
			k, v, _ := strings.Cut(e, "=")
			m[k] = v
		}
		return m
	}

	t.Run("sets required provider env vars with model", func(t *testing.T) {
		got := envMap(c.envVars("llama3.2"))
		if got["COPILOT_PROVIDER_BASE_URL"] == "" {
			t.Error("COPILOT_PROVIDER_BASE_URL should be set")
		}
		if !strings.HasSuffix(got["COPILOT_PROVIDER_BASE_URL"], "/v1") {
			t.Errorf("COPILOT_PROVIDER_BASE_URL = %q, want /v1 suffix", got["COPILOT_PROVIDER_BASE_URL"])
		}
		if _, ok := got["COPILOT_PROVIDER_API_KEY"]; !ok {
			t.Error("COPILOT_PROVIDER_API_KEY should be set (empty)")
		}
		if got["COPILOT_PROVIDER_WIRE_API"] != "responses" {
			t.Errorf("COPILOT_PROVIDER_WIRE_API = %q, want %q", got["COPILOT_PROVIDER_WIRE_API"], "responses")
		}
		if got["COPILOT_MODEL"] != "llama3.2" {
			t.Errorf("COPILOT_MODEL = %q, want %q", got["COPILOT_MODEL"], "llama3.2")
		}
	})

	t.Run("omits COPILOT_MODEL when model is empty", func(t *testing.T) {
		got := envMap(c.envVars(""))
		if _, ok := got["COPILOT_MODEL"]; ok {
			t.Errorf("COPILOT_MODEL should not be set for empty model, got %q", got["COPILOT_MODEL"])
		}
	})

	t.Run("uses custom OLLAMA_HOST", func(t *testing.T) {
		t.Setenv("OLLAMA_HOST", "http://myhost:9999")
		got := envMap(c.envVars("test"))
		if !strings.Contains(got["COPILOT_PROVIDER_BASE_URL"], "myhost:9999") {
			t.Errorf("COPILOT_PROVIDER_BASE_URL = %q, want custom host", got["COPILOT_PROVIDER_BASE_URL"])
		}
	})
}
