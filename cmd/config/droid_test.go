package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestDroidIntegration(t *testing.T) {
	d := &Droid{}

	t.Run("String", func(t *testing.T) {
		if got := d.String(); got != "Droid" {
			t.Errorf("String() = %q, want %q", got, "Droid")
		}
	})

	t.Run("implements Runner", func(t *testing.T) {
		var _ Runner = d
	})

	t.Run("implements Editor", func(t *testing.T) {
		var _ Editor = d
	})
}

func TestDroidEdit(t *testing.T) {
	d := &Droid{}
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
		if err := d.Edit([]string{"model-a", "model-b"}); err != nil {
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
		if err := d.Edit([]string{"model-a", "model-b"}); err != nil {
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
		d.Edit([]string{"model-a", "model-b", "model-c"})

		// Remove middle model
		d.Edit([]string{"model-a", "model-c"})

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

		d.Edit([]string{"model-a"})

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

		d.Edit([]string{"model-a"})

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
		d.Edit([]string{"test-model"})

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

		d.Edit([]string{"model-a"})

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

		d.Edit([]string{"model-a"})

		settings := readSettings()
		session := settings["sessionDefaultSettings"].(map[string]any)

		if session["reasoningEffort"] != "high" {
			t.Errorf("expected reasoningEffort to remain 'high', got %s", session["reasoningEffort"])
		}
	})
}

// Edge case tests for droid.go

func TestDroidEdit_CorruptedJSON(t *testing.T) {
	d := &Droid{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	settingsDir := filepath.Join(tmpDir, ".factory")
	settingsPath := filepath.Join(settingsDir, "settings.json")

	os.MkdirAll(settingsDir, 0o755)
	os.WriteFile(settingsPath, []byte(`{corrupted json content`), 0o644)

	// Corrupted JSON should return an error so user knows something is wrong
	err := d.Edit([]string{"model-a"})
	if err == nil {
		t.Fatal("expected error for corrupted JSON, got nil")
	}

	// Original corrupted file should be preserved (not overwritten)
	data, _ := os.ReadFile(settingsPath)
	if string(data) != `{corrupted json content` {
		t.Errorf("corrupted file was modified: got %s", string(data))
	}
}

func TestDroidEdit_WrongTypeCustomModels(t *testing.T) {
	d := &Droid{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	settingsDir := filepath.Join(tmpDir, ".factory")
	settingsPath := filepath.Join(settingsDir, "settings.json")

	os.MkdirAll(settingsDir, 0o755)
	// customModels is a string instead of array
	os.WriteFile(settingsPath, []byte(`{"customModels": "not an array"}`), 0o644)

	// Should not panic - wrong type should be handled gracefully
	err := d.Edit([]string{"model-a"})
	if err != nil {
		t.Fatalf("Edit failed with wrong type customModels: %v", err)
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

func TestDroidEdit_EmptyModels(t *testing.T) {
	d := &Droid{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	settingsDir := filepath.Join(tmpDir, ".factory")
	settingsPath := filepath.Join(settingsDir, "settings.json")

	os.MkdirAll(settingsDir, 0o755)
	originalContent := `{"customModels": [{"model": "existing"}]}`
	os.WriteFile(settingsPath, []byte(originalContent), 0o644)

	// Empty models should be no-op
	err := d.Edit([]string{})
	if err != nil {
		t.Fatalf("Edit with empty models failed: %v", err)
	}

	// Original content should be preserved (file not modified)
	data, _ := os.ReadFile(settingsPath)
	if string(data) != originalContent {
		t.Errorf("empty models should not modify file, but content changed")
	}
}

func TestDroidEdit_DuplicateModels(t *testing.T) {
	d := &Droid{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	settingsDir := filepath.Join(tmpDir, ".factory")
	settingsPath := filepath.Join(settingsDir, "settings.json")

	// Add same model twice
	err := d.Edit([]string{"model-a", "model-a"})
	if err != nil {
		t.Fatalf("Edit with duplicates failed: %v", err)
	}

	settings, err := readJSONFile(settingsPath)
	if err != nil {
		t.Fatalf("readJSONFile failed: %v", err)
	}

	customModels, _ := settings["customModels"].([]any)
	// Document current behavior: duplicates are kept as separate entries
	if len(customModels) != 2 {
		t.Logf("Note: duplicates result in %d entries (documenting behavior)", len(customModels))
	}
}

func TestDroidEdit_MalformedModelEntry(t *testing.T) {
	d := &Droid{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	settingsDir := filepath.Join(tmpDir, ".factory")
	settingsPath := filepath.Join(settingsDir, "settings.json")

	os.MkdirAll(settingsDir, 0o755)
	// Model entry is a string instead of a map
	os.WriteFile(settingsPath, []byte(`{"customModels": ["not a map", 123]}`), 0o644)

	err := d.Edit([]string{"model-a"})
	if err != nil {
		t.Fatalf("Edit with malformed entries failed: %v", err)
	}

	// Malformed entries should be preserved in nonOllamaModels
	settings, _ := readJSONFile(settingsPath)
	customModels, _ := settings["customModels"].([]any)

	// Should have: 1 new Ollama model + 2 preserved malformed entries
	if len(customModels) != 3 {
		t.Errorf("expected 3 entries (1 new + 2 preserved malformed), got %d", len(customModels))
	}
}

func TestDroidEdit_WrongTypeSessionSettings(t *testing.T) {
	d := &Droid{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	settingsDir := filepath.Join(tmpDir, ".factory")
	settingsPath := filepath.Join(settingsDir, "settings.json")

	os.MkdirAll(settingsDir, 0o755)
	// sessionDefaultSettings is a string instead of map
	os.WriteFile(settingsPath, []byte(`{"sessionDefaultSettings": "not a map"}`), 0o644)

	err := d.Edit([]string{"model-a"})
	if err != nil {
		t.Fatalf("Edit with wrong type sessionDefaultSettings failed: %v", err)
	}

	// Should create proper sessionDefaultSettings
	settings, _ := readJSONFile(settingsPath)
	session, ok := settings["sessionDefaultSettings"].(map[string]any)
	if !ok {
		t.Fatalf("sessionDefaultSettings should be map after setup, got %T", settings["sessionDefaultSettings"])
	}
	if session["model"] == nil {
		t.Error("expected model to be set in sessionDefaultSettings")
	}
}

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
