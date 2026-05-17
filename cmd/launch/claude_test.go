package launch

import (
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
)

func TestClaudeIntegration(t *testing.T) {
	c := &Claude{}

	t.Run("String", func(t *testing.T) {
		if got := c.String(); got != "Claude Code" {
			t.Errorf("String() = %q, want %q", got, "Claude Code")
		}
	})

	t.Run("implements Runner", func(t *testing.T) {
		var _ Runner = c
	})
}

func TestClaudeFindPath(t *testing.T) {
	c := &Claude{}

	t.Run("finds claude in PATH", func(t *testing.T) {
		tmpDir := t.TempDir()
		name := "claude"
		if runtime.GOOS == "windows" {
			name = "claude.exe"
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

	t.Run("falls back to ~/.claude/local/claude", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		t.Setenv("PATH", t.TempDir()) // empty dir, no claude binary

		name := "claude"
		if runtime.GOOS == "windows" {
			name = "claude.exe"
		}
		fallback := filepath.Join(tmpDir, ".claude", "local", name)
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
		t.Setenv("PATH", t.TempDir()) // empty dir, no claude binary

		_, err := c.findPath()
		if err == nil {
			t.Fatal("expected error, got nil")
		}
	})
}

func TestClaudeArgs(t *testing.T) {
	c := &Claude{}

	tests := []struct {
		name  string
		model string
		args  []string
		want  []string
	}{
		{"with model", "llama3.2", nil, []string{"--model", "llama3.2"}},
		{"empty model", "", nil, nil},
		{"with model and verbose", "llama3.2", []string{"--verbose"}, []string{"--model", "llama3.2", "--verbose"}},
		{"empty model with help", "", []string{"--help"}, []string{"--help"}},
		{"with allowed tools", "llama3.2", []string{"--allowedTools", "Read,Write,Bash"}, []string{"--model", "llama3.2", "--allowedTools", "Read,Write,Bash"}},
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

func TestClaudeModelEnvVars(t *testing.T) {
	c := &Claude{}

	envMap := func(envs []string) map[string]string {
		m := make(map[string]string)
		for _, e := range envs {
			k, v, _ := strings.Cut(e, "=")
			m[k] = v
		}
		return m
	}

	t.Run("maps all Claude model env vars to the provided model", func(t *testing.T) {
		got := envMap(c.modelEnvVars("llama3.2"))
		if got["ANTHROPIC_DEFAULT_OPUS_MODEL"] != "llama3.2" {
			t.Errorf("OPUS = %q, want llama3.2", got["ANTHROPIC_DEFAULT_OPUS_MODEL"])
		}
		if got["ANTHROPIC_DEFAULT_SONNET_MODEL"] != "llama3.2" {
			t.Errorf("SONNET = %q, want llama3.2", got["ANTHROPIC_DEFAULT_SONNET_MODEL"])
		}
		if got["ANTHROPIC_DEFAULT_HAIKU_MODEL"] != "llama3.2" {
			t.Errorf("HAIKU = %q, want llama3.2", got["ANTHROPIC_DEFAULT_HAIKU_MODEL"])
		}
		if got["CLAUDE_CODE_SUBAGENT_MODEL"] != "llama3.2" {
			t.Errorf("SUBAGENT = %q, want llama3.2", got["CLAUDE_CODE_SUBAGENT_MODEL"])
		}
		if got["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] != "" {
			t.Errorf("AUTO_COMPACT_WINDOW = %q, want empty for local models", got["CLAUDE_CODE_AUTO_COMPACT_WINDOW"])
		}
	})

	t.Run("supports empty model", func(t *testing.T) {
		got := envMap(c.modelEnvVars(""))
		if got["ANTHROPIC_DEFAULT_OPUS_MODEL"] != "" {
			t.Errorf("OPUS = %q, want empty", got["ANTHROPIC_DEFAULT_OPUS_MODEL"])
		}
		if got["ANTHROPIC_DEFAULT_SONNET_MODEL"] != "" {
			t.Errorf("SONNET = %q, want empty", got["ANTHROPIC_DEFAULT_SONNET_MODEL"])
		}
		if got["ANTHROPIC_DEFAULT_HAIKU_MODEL"] != "" {
			t.Errorf("HAIKU = %q, want empty", got["ANTHROPIC_DEFAULT_HAIKU_MODEL"])
		}
		if got["CLAUDE_CODE_SUBAGENT_MODEL"] != "" {
			t.Errorf("SUBAGENT = %q, want empty", got["CLAUDE_CODE_SUBAGENT_MODEL"])
		}
		if got["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] != "" {
			t.Errorf("AUTO_COMPACT_WINDOW = %q, want empty", got["CLAUDE_CODE_AUTO_COMPACT_WINDOW"])
		}
	})

	t.Run("sets auto compact window for known cloud models", func(t *testing.T) {
		got := envMap(c.modelEnvVars("glm-5:cloud"))
		if got["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] != "202752" {
			t.Errorf("AUTO_COMPACT_WINDOW = %q, want 202752", got["CLAUDE_CODE_AUTO_COMPACT_WINDOW"])
		}
	})

	t.Run("does not set auto compact window for unknown cloud models", func(t *testing.T) {
		got := envMap(c.modelEnvVars("unknown-model:cloud"))
		if got["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] != "" {
			t.Errorf("AUTO_COMPACT_WINDOW = %q, want empty", got["CLAUDE_CODE_AUTO_COMPACT_WINDOW"])
		}
	})
}
