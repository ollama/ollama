package launch

import (
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
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

func TestCopilotRunPassesTokenEnvVars(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell stub uses /bin/sh")
	}

	tmpDir := t.TempDir()
	capturePath := filepath.Join(tmpDir, "capture")
	fakeBin := filepath.Join(tmpDir, "copilot")
	script := `#!/bin/sh
{
  echo "ARGS:$*"
  echo "COPILOT_MODEL=$COPILOT_MODEL"
  echo "COPILOT_PROVIDER_MAX_PROMPT_TOKENS=$COPILOT_PROVIDER_MAX_PROMPT_TOKENS"
  echo "COPILOT_PROVIDER_MAX_OUTPUT_TOKENS=$COPILOT_PROVIDER_MAX_OUTPUT_TOKENS"
} > "$COPILOT_CAPTURE"
`
	if err := os.WriteFile(fakeBin, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", tmpDir)
	t.Setenv("COPILOT_CAPTURE", capturePath)
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "")

	c := &Copilot{}
	err := c.Run("gemma4:31b-nvfp4", []LaunchModel{
		{Name: "gemma4:31b-nvfp4", ContextLength: 262_144},
	}, []string{"-p", "hello"})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	data, err := os.ReadFile(capturePath)
	if err != nil {
		t.Fatal(err)
	}
	got := string(data)
	for _, want := range []string{
		"ARGS:--model gemma4:31b-nvfp4 -p hello",
		"COPILOT_MODEL=gemma4:31b-nvfp4",
		"COPILOT_PROVIDER_MAX_PROMPT_TOKENS=262144",
		"COPILOT_PROVIDER_MAX_OUTPUT_TOKENS=64000",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("captured output missing %q:\n%s", want, got)
		}
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
		t.Setenv("OLLAMA_CONTEXT_LENGTH", "")
		got := envMap(c.envVars("llama3.2", []LaunchModel{
			{Name: "llama3.2:latest", ContextLength: 65_536, MaxOutputTokens: 4_096},
		}))
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
		if got["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"] != "65536" {
			t.Errorf("COPILOT_PROVIDER_MAX_PROMPT_TOKENS = %q, want %q", got["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"], "65536")
		}
		if got["COPILOT_PROVIDER_MAX_OUTPUT_TOKENS"] != "4096" {
			t.Errorf("COPILOT_PROVIDER_MAX_OUTPUT_TOKENS = %q, want %q", got["COPILOT_PROVIDER_MAX_OUTPUT_TOKENS"], "4096")
		}
		if got["COPILOT_MODEL"] != "llama3.2" {
			t.Errorf("COPILOT_MODEL = %q, want %q", got["COPILOT_MODEL"], "llama3.2")
		}
	})

	t.Run("omits COPILOT_MODEL when model is empty", func(t *testing.T) {
		t.Setenv("OLLAMA_CONTEXT_LENGTH", "")
		got := envMap(c.envVars("", nil))
		if _, ok := got["COPILOT_MODEL"]; ok {
			t.Errorf("COPILOT_MODEL should not be set for empty model, got %q", got["COPILOT_MODEL"])
		}
	})

	t.Run("uses custom OLLAMA_HOST", func(t *testing.T) {
		t.Setenv("OLLAMA_HOST", "http://myhost:9999")
		t.Setenv("OLLAMA_CONTEXT_LENGTH", "")
		got := envMap(c.envVars("test", nil))
		if !strings.Contains(got["COPILOT_PROVIDER_BASE_URL"], "myhost:9999") {
			t.Errorf("COPILOT_PROVIDER_BASE_URL = %q, want custom host", got["COPILOT_PROVIDER_BASE_URL"])
		}
	})

	t.Run("uses details context length when direct context is absent", func(t *testing.T) {
		t.Setenv("OLLAMA_CONTEXT_LENGTH", "")
		got := envMap(c.envVars("llama3.2", []LaunchModel{
			{Name: "llama3.2", Details: api.ModelDetails{ContextLength: 32_768}},
		}))
		if got["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"] != "32768" {
			t.Errorf("COPILOT_PROVIDER_MAX_PROMPT_TOKENS = %q, want %q", got["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"], "32768")
		}
		if got["COPILOT_PROVIDER_MAX_OUTPUT_TOKENS"] != "64000" {
			t.Errorf("COPILOT_PROVIDER_MAX_OUTPUT_TOKENS = %q, want %q", got["COPILOT_PROVIDER_MAX_OUTPUT_TOKENS"], "64000")
		}
	})

	t.Run("uses default output tokens for custom model metadata", func(t *testing.T) {
		t.Setenv("OLLAMA_CONTEXT_LENGTH", "")
		got := envMap(c.envVars("gemma4:31b-nvfp4", []LaunchModel{
			{Name: "gemma4:31b-nvfp4", ContextLength: 262_144},
		}))
		if got["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"] != "262144" {
			t.Errorf("COPILOT_PROVIDER_MAX_PROMPT_TOKENS = %q, want %q", got["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"], "262144")
		}
		if got["COPILOT_PROVIDER_MAX_OUTPUT_TOKENS"] != "64000" {
			t.Errorf("COPILOT_PROVIDER_MAX_OUTPUT_TOKENS = %q, want %q", got["COPILOT_PROVIDER_MAX_OUTPUT_TOKENS"], "64000")
		}
	})

	t.Run("uses known cloud limits when inventory metadata is absent", func(t *testing.T) {
		t.Setenv("OLLAMA_CONTEXT_LENGTH", "64000")
		got := envMap(c.envVars("qwen3.5:cloud", nil))
		if got["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"] != "262144" {
			t.Errorf("COPILOT_PROVIDER_MAX_PROMPT_TOKENS = %q, want %q", got["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"], "262144")
		}
		if got["COPILOT_PROVIDER_MAX_OUTPUT_TOKENS"] != "32768" {
			t.Errorf("COPILOT_PROVIDER_MAX_OUTPUT_TOKENS = %q, want %q", got["COPILOT_PROVIDER_MAX_OUTPUT_TOKENS"], "32768")
		}
	})

	t.Run("uses fallback limits when metadata is absent", func(t *testing.T) {
		t.Setenv("OLLAMA_CONTEXT_LENGTH", "")
		got := envMap(c.envVars("custom-model", nil))
		if got["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"] != "4096" {
			t.Errorf("COPILOT_PROVIDER_MAX_PROMPT_TOKENS = %q, want %q", got["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"], "4096")
		}
		if got["COPILOT_PROVIDER_MAX_OUTPUT_TOKENS"] != "64000" {
			t.Errorf("COPILOT_PROVIDER_MAX_OUTPUT_TOKENS = %q, want %q", got["COPILOT_PROVIDER_MAX_OUTPUT_TOKENS"], "64000")
		}
	})

	t.Run("uses explicit local context override", func(t *testing.T) {
		t.Setenv("OLLAMA_CONTEXT_LENGTH", "64000")
		got := envMap(c.envVars("llama3.2", []LaunchModel{
			{Name: "llama3.2", ContextLength: 131_072, Details: api.ModelDetails{Format: "gguf"}},
		}))
		if got["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"] != "64000" {
			t.Errorf("COPILOT_PROVIDER_MAX_PROMPT_TOKENS = %q, want %q", got["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"], "64000")
		}
	})
}
