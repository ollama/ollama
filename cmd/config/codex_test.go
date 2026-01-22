package config

import (
	"testing"
)

func TestCodexIntegration(t *testing.T) {
	// Documents that Codex uses --oss flag instead of env vars, meaning any
	// existing OPENAI_* env vars the user has set will be preserved (not
	// overridden). This is intentional - the --oss flag tells Codex to use
	// the local Ollama server.
	t.Run("EnvVars_Empty_PreservesExisting", func(t *testing.T) {
		envVars := codexIntegration.EnvVars("llama3.2")

		if len(envVars) != 0 {
			t.Errorf("expected 0 env vars, got %d", len(envVars))
		}

		// Simulate building the environment like runIntegration does
		existingEnv := []string{
			"PATH=/usr/bin",
			"OPENAI_API_KEY=sk-user-real-key",
			"OPENAI_BASE_URL=https://api.openai.com",
		}

		// Since EnvVars returns empty, existing vars are preserved
		env := append(existingEnv, envVarsToStrings(envVars)...)
		result := resolveEnv(env)

		// Verify existing vars are NOT overridden (unlike Claude)
		if result["OPENAI_API_KEY"] != "sk-user-real-key" {
			t.Errorf("OPENAI_API_KEY should be preserved, got %q", result["OPENAI_API_KEY"])
		}
		if result["OPENAI_BASE_URL"] != "https://api.openai.com" {
			t.Errorf("OPENAI_BASE_URL should be preserved, got %q", result["OPENAI_BASE_URL"])
		}
	})

	t.Run("Args_WithModel", func(t *testing.T) {
		args := codexIntegration.Args("llama3.2")

		if len(args) != 3 {
			t.Fatalf("expected 3 args, got %d", len(args))
		}
		if args[0] != "--oss" {
			t.Errorf("expected first arg '--oss', got %q", args[0])
		}
		if args[1] != "-m" {
			t.Errorf("expected second arg '-m', got %q", args[1])
		}
		if args[2] != "llama3.2" {
			t.Errorf("expected third arg 'llama3.2', got %q", args[2])
		}
	})

	t.Run("Args_EmptyModel", func(t *testing.T) {
		args := codexIntegration.Args("")

		if len(args) != 1 {
			t.Fatalf("expected 1 arg, got %d", len(args))
		}
		if args[0] != "--oss" {
			t.Errorf("expected '--oss', got %q", args[0])
		}
	})

	t.Run("Args_AlwaysIncludesOss", func(t *testing.T) {
		for _, model := range []string{"", "model-a", "qwen2.5:latest"} {
			args := codexIntegration.Args(model)
			if len(args) == 0 || args[0] != "--oss" {
				t.Errorf("args for model %q should start with --oss, got %v", model, args)
			}
		}
	})

	t.Run("Metadata", func(t *testing.T) {
		if codexIntegration.Name != "Codex" {
			t.Errorf("expected Name 'Codex', got %q", codexIntegration.Name)
		}
		if codexIntegration.DisplayName != "Codex" {
			t.Errorf("expected DisplayName 'Codex', got %q", codexIntegration.DisplayName)
		}
		if codexIntegration.Command != "codex" {
			t.Errorf("expected Command 'codex', got %q", codexIntegration.Command)
		}
	})
}
