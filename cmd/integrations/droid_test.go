package integrations

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestSetupDroidSettings(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	settingsDir := filepath.Join(tmpDir, ".factory")
	settingsPath := filepath.Join(settingsDir, "settings.json")

	cleanup := func() {
		os.RemoveAll(settingsDir)
	}

	readSettings := func() map[string]any {
		data, _ := os.ReadFile(settingsPath)
		var settings map[string]any
		json.Unmarshal(data, &settings)
		return settings
	}

	getCustomModels := func(settings map[string]any) []map[string]any {
		models, ok := settings["customModels"].([]any)
		if !ok {
			return nil
		}
		var result []map[string]any
		for _, m := range models {
			if entry, ok := m.(map[string]any); ok {
				result = append(result, entry)
			}
		}
		return result
	}

	t.Run("fresh install creates models with sequential indices", func(t *testing.T) {
		cleanup()
		if err := setupDroidSettings([]string{"model-a", "model-b"}); err != nil {
			t.Fatal(err)
		}

		settings := readSettings()
		models := getCustomModels(settings)

		if len(models) != 2 {
			t.Fatalf("expected 2 models, got %d", len(models))
		}

		// Check first model
		if models[0]["model"] != "model-a" {
			t.Errorf("expected model-a, got %s", models[0]["model"])
		}
		if models[0]["id"] != "custom:model-a-[Ollama]-0" {
			t.Errorf("expected custom:model-a-[Ollama]-0, got %s", models[0]["id"])
		}
		if models[0]["index"] != float64(0) {
			t.Errorf("expected index 0, got %v", models[0]["index"])
		}

		// Check second model
		if models[1]["model"] != "model-b" {
			t.Errorf("expected model-b, got %s", models[1]["model"])
		}
		if models[1]["id"] != "custom:model-b-[Ollama]-1" {
			t.Errorf("expected custom:model-b-[Ollama]-1, got %s", models[1]["id"])
		}
		if models[1]["index"] != float64(1) {
			t.Errorf("expected index 1, got %v", models[1]["index"])
		}
	})

	t.Run("sets sessionDefaultSettings.model to first model ID", func(t *testing.T) {
		cleanup()
		if err := setupDroidSettings([]string{"model-a", "model-b"}); err != nil {
			t.Fatal(err)
		}

		settings := readSettings()
		session, ok := settings["sessionDefaultSettings"].(map[string]any)
		if !ok {
			t.Fatal("sessionDefaultSettings not found")
		}
		if session["model"] != "custom:model-a-[Ollama]-0" {
			t.Errorf("expected custom:model-a-[Ollama]-0, got %s", session["model"])
		}
	})

	t.Run("re-indexes when models removed", func(t *testing.T) {
		cleanup()
		// Add three models
		setupDroidSettings([]string{"model-a", "model-b", "model-c"})

		// Remove middle model
		setupDroidSettings([]string{"model-a", "model-c"})

		settings := readSettings()
		models := getCustomModels(settings)

		if len(models) != 2 {
			t.Fatalf("expected 2 models, got %d", len(models))
		}

		// Check indices are sequential 0, 1
		if models[0]["index"] != float64(0) {
			t.Errorf("expected index 0, got %v", models[0]["index"])
		}
		if models[1]["index"] != float64(1) {
			t.Errorf("expected index 1, got %v", models[1]["index"])
		}

		// Check IDs match new indices
		if models[0]["id"] != "custom:model-a-[Ollama]-0" {
			t.Errorf("expected custom:model-a-[Ollama]-0, got %s", models[0]["id"])
		}
		if models[1]["id"] != "custom:model-c-[Ollama]-1" {
			t.Errorf("expected custom:model-c-[Ollama]-1, got %s", models[1]["id"])
		}
	})

	t.Run("preserves non-Ollama custom models", func(t *testing.T) {
		cleanup()
		os.MkdirAll(settingsDir, 0o755)
		// Pre-existing non-Ollama model
		os.WriteFile(settingsPath, []byte(`{
			"customModels": [
				{"model": "gpt-4", "displayName": "GPT-4", "provider": "openai"}
			]
		}`), 0o644)

		setupDroidSettings([]string{"model-a"})

		settings := readSettings()
		models := getCustomModels(settings)

		if len(models) != 2 {
			t.Fatalf("expected 2 models (1 Ollama + 1 non-Ollama), got %d", len(models))
		}

		// Ollama model should be first
		if models[0]["model"] != "model-a" {
			t.Errorf("expected Ollama model first, got %s", models[0]["model"])
		}

		// Non-Ollama model should be preserved at end
		if models[1]["model"] != "gpt-4" {
			t.Errorf("expected gpt-4 preserved, got %s", models[1]["model"])
		}
	})

	t.Run("preserves other settings", func(t *testing.T) {
		cleanup()
		os.MkdirAll(settingsDir, 0o755)
		os.WriteFile(settingsPath, []byte(`{
			"theme": "dark",
			"enableHooks": true,
			"sessionDefaultSettings": {"autonomyMode": "auto-high"}
		}`), 0o644)

		setupDroidSettings([]string{"model-a"})

		settings := readSettings()

		if settings["theme"] != "dark" {
			t.Error("theme was not preserved")
		}
		if settings["enableHooks"] != true {
			t.Error("enableHooks was not preserved")
		}

		session := settings["sessionDefaultSettings"].(map[string]any)
		if session["autonomyMode"] != "auto-high" {
			t.Error("autonomyMode was not preserved")
		}
	})

	t.Run("required fields present", func(t *testing.T) {
		cleanup()
		setupDroidSettings([]string{"test-model"})

		settings := readSettings()
		models := getCustomModels(settings)

		if len(models) != 1 {
			t.Fatal("expected 1 model")
		}

		model := models[0]
		requiredFields := []string{"model", "displayName", "baseUrl", "apiKey", "provider", "maxOutputTokens", "id", "index"}
		for _, field := range requiredFields {
			if model[field] == nil {
				t.Errorf("missing required field: %s", field)
			}
		}

		if model["baseUrl"] != "http://localhost:11434/v1" {
			t.Errorf("unexpected baseUrl: %s", model["baseUrl"])
		}
		if model["apiKey"] != "ollama" {
			t.Errorf("unexpected apiKey: %s", model["apiKey"])
		}
		if model["provider"] != "generic-chat-completion-api" {
			t.Errorf("unexpected provider: %s", model["provider"])
		}
	})

	t.Run("fixes invalid reasoningEffort", func(t *testing.T) {
		cleanup()
		os.MkdirAll(settingsDir, 0o755)
		// Pre-existing settings with invalid reasoningEffort
		os.WriteFile(settingsPath, []byte(`{
			"sessionDefaultSettings": {"reasoningEffort": "off"}
		}`), 0o644)

		setupDroidSettings([]string{"model-a"})

		settings := readSettings()
		session := settings["sessionDefaultSettings"].(map[string]any)

		if session["reasoningEffort"] != "none" {
			t.Errorf("expected reasoningEffort to be fixed to 'none', got %s", session["reasoningEffort"])
		}
	})

	t.Run("preserves valid reasoningEffort", func(t *testing.T) {
		cleanup()
		os.MkdirAll(settingsDir, 0o755)
		os.WriteFile(settingsPath, []byte(`{
			"sessionDefaultSettings": {"reasoningEffort": "high"}
		}`), 0o644)

		setupDroidSettings([]string{"model-a"})

		settings := readSettings()
		session := settings["sessionDefaultSettings"].(map[string]any)

		if session["reasoningEffort"] != "high" {
			t.Errorf("expected reasoningEffort to remain 'high', got %s", session["reasoningEffort"])
		}
	})
}
