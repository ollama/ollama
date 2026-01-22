package config

import (
	"fmt"
	"strings"
	"testing"
)

func TestClaudeIntegration(t *testing.T) {
	t.Run("EnvVars", func(t *testing.T) {
		envVars := claudeIntegration.EnvVars("llama3.2")

		expected := map[string]string{
			"ANTHROPIC_BASE_URL":   "http://localhost:11434",
			"ANTHROPIC_API_KEY":    "ollama",
			"ANTHROPIC_AUTH_TOKEN": "ollama",
		}

		if len(envVars) != len(expected) {
			t.Errorf("expected %d env vars, got %d", len(expected), len(envVars))
		}

		for _, env := range envVars {
			if expected[env.Name] != env.Value {
				t.Errorf("env %s: expected %q, got %q", env.Name, expected[env.Name], env.Value)
			}
		}
	})

	// Documents that integration env vars are appended to os.Environ(), meaning
	// they will override any existing env vars with the same name. This is the
	// intended behavior - we want to ensure Claude Code connects to Ollama
	// regardless of what ANTHROPIC_* vars the user may have set.
	t.Run("EnvVars_OverrideExisting", func(t *testing.T) {
		// Simulate building the environment like runIntegration does:

		// User has existing Anthropic credentials
		existingEnv := []string{
			"PATH=/usr/bin",
			"HOME=/home/user",
			"ANTHROPIC_API_KEY=sk-ant-user-real-key",
			"ANTHROPIC_BASE_URL=https://api.anthropic.com",
		}

		// Integration appends its env vars (these come after existing ones)
		env := append(existingEnv, envVarsToStrings(claudeIntegration.EnvVars("llama3.2"))...)

		// When exec.Cmd runs, later values override earlier ones for duplicate keys
		// Verify our values come last (and thus take precedence)
		result := resolveEnv(env)

		if result["ANTHROPIC_API_KEY"] != "ollama" {
			t.Errorf("ANTHROPIC_API_KEY should be overridden to 'ollama', got %q", result["ANTHROPIC_API_KEY"])
		}
		if result["ANTHROPIC_BASE_URL"] != "http://localhost:11434" {
			t.Errorf("ANTHROPIC_BASE_URL should be overridden to localhost, got %q", result["ANTHROPIC_BASE_URL"])
		}
		// Non-overridden vars should be preserved
		if result["HOME"] != "/home/user" {
			t.Errorf("HOME should be preserved, got %q", result["HOME"])
		}
	})

	t.Run("EnvVars_ModelIndependent", func(t *testing.T) {
		envVars1 := claudeIntegration.EnvVars("model-a")
		envVars2 := claudeIntegration.EnvVars("model-b")

		if len(envVars1) != len(envVars2) {
			t.Error("env vars should be model-independent")
		}
		for i := range envVars1 {
			if envVars1[i] != envVars2[i] {
				t.Error("env vars should be model-independent")
			}
		}
	})

	t.Run("Args_WithModel", func(t *testing.T) {
		args := claudeIntegration.Args("llama3.2")

		if len(args) != 2 {
			t.Fatalf("expected 2 args, got %d", len(args))
		}
		if args[0] != "--model" || args[1] != "llama3.2" {
			t.Errorf("expected [--model llama3.2], got %v", args)
		}
	})

	t.Run("Args_EmptyModel", func(t *testing.T) {
		args := claudeIntegration.Args("")

		if args != nil {
			t.Errorf("expected nil args for empty model, got %v", args)
		}
	})

	t.Run("Metadata", func(t *testing.T) {
		if claudeIntegration.Name != "Claude" {
			t.Errorf("expected Name 'Claude', got %q", claudeIntegration.Name)
		}
		if claudeIntegration.DisplayName != "Claude Code" {
			t.Errorf("expected DisplayName 'Claude Code', got %q", claudeIntegration.DisplayName)
		}
		if claudeIntegration.Command != "claude" {
			t.Errorf("expected Command 'claude', got %q", claudeIntegration.Command)
		}
	})
}

// envVarsToStrings converts []envVar to []string in KEY=VALUE format
func envVarsToStrings(vars []envVar) []string {
	result := make([]string, len(vars))
	for i, v := range vars {
		result[i] = fmt.Sprintf("%s=%s", v.Name, v.Value)
	}
	return result
}

// resolveEnv simulates how exec.Cmd resolves duplicate env vars (last wins)
func resolveEnv(env []string) map[string]string {
	result := make(map[string]string)
	for _, e := range env {
		if idx := strings.Index(e, "="); idx != -1 {
			result[e[:idx]] = e[idx+1:]
		}
	}
	return result
}
