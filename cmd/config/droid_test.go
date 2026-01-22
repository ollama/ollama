package config

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

// Edge case tests for droid.go

// TestSetupDroidSettings_CorruptedJSON verifies that corrupted settings.json does not cause panic.
// User's existing non-Ollama settings must be preserved; corrupted file should be handled gracefully.
func TestSetupDroidSettings_CorruptedJSON(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	settingsDir := filepath.Join(tmpDir, ".factory")
	settingsPath := filepath.Join(settingsDir, "settings.json")

	os.MkdirAll(settingsDir, 0o755)
	os.WriteFile(settingsPath, []byte(`{corrupted json content`), 0o644)

	// Should not panic - corrupted JSON should be treated as empty
	err := setupDroidSettings([]string{"model-a"})
	if err != nil {
		t.Fatalf("setupDroidSettings failed with corrupted JSON: %v", err)
	}

	// Verify new config was created
	data, _ := os.ReadFile(settingsPath)
	var settings map[string]any
	if err := json.Unmarshal(data, &settings); err != nil {
		t.Errorf("resulting settings.json is not valid JSON: %v", err)
	}
}

// TestSetupDroidSettings_WrongTypeCustomModels verifies handling when customModels is wrong type.
// Type assertion safety - customModels might be string instead of array.
func TestSetupDroidSettings_WrongTypeCustomModels(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	settingsDir := filepath.Join(tmpDir, ".factory")
	settingsPath := filepath.Join(settingsDir, "settings.json")

	os.MkdirAll(settingsDir, 0o755)
	// customModels is a string instead of array
	os.WriteFile(settingsPath, []byte(`{"customModels": "not an array"}`), 0o644)

	// Should not panic - wrong type should be handled gracefully
	err := setupDroidSettings([]string{"model-a"})
	if err != nil {
		t.Fatalf("setupDroidSettings failed with wrong type customModels: %v", err)
	}

	// Verify models were added correctly
	data, _ := os.ReadFile(settingsPath)
	var settings map[string]any
	json.Unmarshal(data, &settings)

	customModels, ok := settings["customModels"].([]any)
	if !ok {
		t.Fatalf("customModels should be array after setup, got %T", settings["customModels"])
	}
	if len(customModels) != 1 {
		t.Errorf("expected 1 model, got %d", len(customModels))
	}
}

// TestSetupDroidSettings_EmptyModels documents intentional behavior: empty models = no-op (early return).
// Prevents accidental clearing of user's model config.
func TestSetupDroidSettings_EmptyModels(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	settingsDir := filepath.Join(tmpDir, ".factory")
	settingsPath := filepath.Join(settingsDir, "settings.json")

	os.MkdirAll(settingsDir, 0o755)
	originalContent := `{"customModels": [{"model": "existing"}]}`
	os.WriteFile(settingsPath, []byte(originalContent), 0o644)

	// Empty models should be no-op
	err := setupDroidSettings([]string{})
	if err != nil {
		t.Fatalf("setupDroidSettings with empty models failed: %v", err)
	}

	// Original content should be preserved (file not modified)
	data, _ := os.ReadFile(settingsPath)
	if string(data) != originalContent {
		t.Errorf("empty models should not modify file, but content changed")
	}
}

// TestSetupDroidSettings_DuplicateModels verifies handling of duplicate model names in input.
// Should handle gracefully - current implementation keeps duplicates.
func TestSetupDroidSettings_DuplicateModels(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	// Add same model twice
	err := setupDroidSettings([]string{"model-a", "model-a"})
	if err != nil {
		t.Fatalf("setupDroidSettings with duplicates failed: %v", err)
	}

	settings, err := droidSettings()
	if err != nil {
		t.Fatalf("droidSettings failed: %v", err)
	}

	customModels, _ := settings["customModels"].([]any)
	// Document current behavior: duplicates are kept as separate entries
	if len(customModels) != 2 {
		t.Logf("Note: duplicates result in %d entries (documenting behavior)", len(customModels))
	}
}

// TestSetupDroidSettings_MalformedModelEntry verifies handling when model entry is not a map.
// xisting entries might be malformed.
func TestSetupDroidSettings_MalformedModelEntry(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	settingsDir := filepath.Join(tmpDir, ".factory")
	settingsPath := filepath.Join(settingsDir, "settings.json")

	os.MkdirAll(settingsDir, 0o755)
	// Model entry is a string instead of a map
	os.WriteFile(settingsPath, []byte(`{"customModels": ["not a map", 123]}`), 0o644)

	err := setupDroidSettings([]string{"model-a"})
	if err != nil {
		t.Fatalf("setupDroidSettings with malformed entries failed: %v", err)
	}

	// Malformed entries should be preserved in nonOllamaModels
	settings, _ := droidSettings()
	customModels, _ := settings["customModels"].([]any)

	// Should have: 1 new Ollama model + 2 preserved malformed entries
	if len(customModels) != 3 {
		t.Errorf("expected 3 entries (1 new + 2 preserved malformed), got %d", len(customModels))
	}
}

// TestSetupDroidSettings_WrongTypeSessionSettings verifies handling when sessionDefaultSettings is wrong type.
func TestSetupDroidSettings_WrongTypeSessionSettings(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	settingsDir := filepath.Join(tmpDir, ".factory")
	settingsPath := filepath.Join(settingsDir, "settings.json")

	os.MkdirAll(settingsDir, 0o755)
	// sessionDefaultSettings is a string instead of map
	os.WriteFile(settingsPath, []byte(`{"sessionDefaultSettings": "not a map"}`), 0o644)

	err := setupDroidSettings([]string{"model-a"})
	if err != nil {
		t.Fatalf("setupDroidSettings with wrong type sessionDefaultSettings failed: %v", err)
	}

	// Should create proper sessionDefaultSettings
	settings, _ := droidSettings()
	session, ok := settings["sessionDefaultSettings"].(map[string]any)
	if !ok {
		t.Fatalf("sessionDefaultSettings should be map after setup, got %T", settings["sessionDefaultSettings"])
	}
	if session["model"] == nil {
		t.Error("expected model to be set in sessionDefaultSettings")
	}
}

// TestIsValidReasoningEffort documents the valid values for reasoningEffort.
func TestIsValidReasoningEffort(t *testing.T) {
	tests := []struct {
		effort string
		valid  bool
	}{
		{"high", true},
		{"medium", true},
		{"low", true},
		{"none", true},
		{"off", false},
		{"", false},
		{"HIGH", false}, // case sensitive
		{"max", false},
	}

	for _, tt := range tests {
		t.Run(tt.effort, func(t *testing.T) {
			got := isValidReasoningEffort(tt.effort)
			if got != tt.valid {
				t.Errorf("isValidReasoningEffort(%q) = %v, want %v", tt.effort, got, tt.valid)
			}
		})
	}
}
