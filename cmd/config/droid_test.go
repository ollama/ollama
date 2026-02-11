package config

import (
	"encoding/json"
	"fmt"
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
		if models[0]["id"] != "custom:model-a-0" {
			t.Errorf("expected custom:model-a-0, got %s", models[0]["id"])
		}
		if models[0]["index"] != float64(0) {
			t.Errorf("expected index 0, got %v", models[0]["index"])
		}

		// Check second model
		if models[1]["model"] != "model-b" {
			t.Errorf("expected model-b, got %s", models[1]["model"])
		}
		if models[1]["id"] != "custom:model-b-1" {
			t.Errorf("expected custom:model-b-1, got %s", models[1]["id"])
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
		if session["model"] != "custom:model-a-0" {
			t.Errorf("expected custom:model-a-0, got %s", session["model"])
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
		if models[0]["id"] != "custom:model-a-0" {
			t.Errorf("expected custom:model-a-0, got %s", models[0]["id"])
		}
		if models[1]["id"] != "custom:model-c-1" {
			t.Errorf("expected custom:model-c-1, got %s", models[1]["id"])
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

		if model["baseUrl"] != "http://127.0.0.1:11434/v1" {
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

	// Malformed entries (non-object) are dropped - only valid model objects are preserved
	settings, _ := readJSONFile(settingsPath)
	customModels, _ := settings["customModels"].([]any)

	// Should have: 1 new Ollama model only (malformed entries dropped)
	if len(customModels) != 1 {
		t.Errorf("expected 1 entry (malformed entries dropped), got %d", len(customModels))
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

// testDroidSettingsFixture is a representative settings.json fixture for testing.
// It covers: simple fields, arrays, nested objects, and customModels.
const testDroidSettingsFixture = `{
  "commandAllowlist": ["ls", "pwd", "git status"],
  "diffMode": "github",
  "enableHooks": true,
  "hooks": {
    "claudeHooksImported": true,
    "importedClaudeHooks": ["uv run ruff check", "echo test"]
  },
  "ideExtensionPromptedAt": {
    "cursor": 1763081579486,
    "vscode": 1762992990179
  },
  "customModels": [
    {
      "model": "existing-ollama-model",
      "displayName": "existing-ollama-model",
      "baseUrl": "http://127.0.0.1:11434/v1",
      "apiKey": "ollama",
      "provider": "generic-chat-completion-api",
      "maxOutputTokens": 64000,
      "supportsImages": false,
      "id": "custom:existing-ollama-model-0",
      "index": 0
    },
    {
      "model": "gpt-4",
      "displayName": "GPT-4",
      "baseUrl": "https://api.openai.com/v1",
      "apiKey": "sk-xxx",
      "provider": "openai",
      "maxOutputTokens": 4096,
      "supportsImages": true,
      "id": "openai-gpt4",
      "index": 1,
      "customField": "should be preserved"
    }
  ],
  "sessionDefaultSettings": {
    "autonomyMode": "auto-medium",
    "model": "custom:existing-ollama-model-0",
    "reasoningEffort": "high"
  },
  "todoDisplayMode": "pinned"
}`

func TestDroidEdit_RoundTrip(t *testing.T) {
	d := &Droid{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	settingsDir := filepath.Join(tmpDir, ".factory")
	settingsPath := filepath.Join(settingsDir, "settings.json")

	os.MkdirAll(settingsDir, 0o755)
	os.WriteFile(settingsPath, []byte(testDroidSettingsFixture), 0o644)

	// Edit with new models
	if err := d.Edit([]string{"llama3", "mistral"}); err != nil {
		t.Fatal(err)
	}

	// Read back and verify
	data, _ := os.ReadFile(settingsPath)
	var settings map[string]any
	json.Unmarshal(data, &settings)

	// Verify unknown top-level fields preserved
	if settings["diffMode"] != "github" {
		t.Error("diffMode not preserved")
	}
	if settings["enableHooks"] != true {
		t.Error("enableHooks not preserved")
	}
	if settings["todoDisplayMode"] != "pinned" {
		t.Error("todoDisplayMode not preserved")
	}

	// Verify arrays preserved
	allowlist, ok := settings["commandAllowlist"].([]any)
	if !ok || len(allowlist) != 3 {
		t.Error("commandAllowlist not preserved")
	}

	// Verify nested objects preserved
	hooks, ok := settings["hooks"].(map[string]any)
	if !ok {
		t.Fatal("hooks not preserved")
	}
	if hooks["claudeHooksImported"] != true {
		t.Error("hooks.claudeHooksImported not preserved")
	}
	importedHooks, ok := hooks["importedClaudeHooks"].([]any)
	if !ok || len(importedHooks) != 2 {
		t.Error("hooks.importedClaudeHooks not preserved")
	}

	// Verify deeply nested numeric values preserved
	idePrompted, ok := settings["ideExtensionPromptedAt"].(map[string]any)
	if !ok {
		t.Fatal("ideExtensionPromptedAt not preserved")
	}
	if idePrompted["cursor"] != float64(1763081579486) {
		t.Error("ideExtensionPromptedAt.cursor not preserved")
	}

	// Verify sessionDefaultSettings unknown fields preserved
	session, ok := settings["sessionDefaultSettings"].(map[string]any)
	if !ok {
		t.Fatal("sessionDefaultSettings not preserved")
	}
	if session["autonomyMode"] != "auto-medium" {
		t.Error("sessionDefaultSettings.autonomyMode not preserved")
	}
	if session["reasoningEffort"] != "high" {
		t.Error("sessionDefaultSettings.reasoningEffort not preserved (was valid)")
	}
	// model should be updated
	if session["model"] != "custom:llama3-0" {
		t.Errorf("sessionDefaultSettings.model not updated, got %s", session["model"])
	}

	// Verify customModels: old ollama replaced, non-ollama preserved with extra fields
	models, ok := settings["customModels"].([]any)
	if !ok {
		t.Fatal("customModels not preserved")
	}
	if len(models) != 3 { // 2 new ollama + 1 non-ollama
		t.Fatalf("expected 3 models, got %d", len(models))
	}

	// First two should be new Ollama models
	m0 := models[0].(map[string]any)
	if m0["model"] != "llama3" || m0["apiKey"] != "ollama" {
		t.Error("first model should be llama3")
	}
	m1 := models[1].(map[string]any)
	if m1["model"] != "mistral" || m1["apiKey"] != "ollama" {
		t.Error("second model should be mistral")
	}

	// Third should be preserved non-Ollama with extra field
	m2 := models[2].(map[string]any)
	if m2["model"] != "gpt-4" {
		t.Error("non-Ollama model not preserved")
	}
	if m2["customField"] != "should be preserved" {
		t.Error("non-Ollama model's extra field not preserved")
	}
}

func TestDroidEdit_PreservesUnknownFields(t *testing.T) {
	d := &Droid{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	settingsDir := filepath.Join(tmpDir, ".factory")
	settingsPath := filepath.Join(settingsDir, "settings.json")

	readSettings := func() map[string]any {
		data, _ := os.ReadFile(settingsPath)
		var settings map[string]any
		json.Unmarshal(data, &settings)
		return settings
	}

	t.Run("preserves all JSON value types", func(t *testing.T) {
		os.RemoveAll(settingsDir)
		os.MkdirAll(settingsDir, 0o755)

		original := `{
			"stringField": "value",
			"numberField": 42,
			"floatField": 3.14,
			"boolField": true,
			"nullField": null,
			"arrayField": [1, "two", true],
			"objectField": {"nested": "value"},
			"customModels": [],
			"sessionDefaultSettings": {}
		}`
		os.WriteFile(settingsPath, []byte(original), 0o644)

		if err := d.Edit([]string{"model-a"}); err != nil {
			t.Fatal(err)
		}

		settings := readSettings()

		if settings["stringField"] != "value" {
			t.Error("stringField not preserved")
		}
		if settings["numberField"] != float64(42) {
			t.Error("numberField not preserved")
		}
		if settings["floatField"] != 3.14 {
			t.Error("floatField not preserved")
		}
		if settings["boolField"] != true {
			t.Error("boolField not preserved")
		}
		if settings["nullField"] != nil {
			t.Error("nullField not preserved")
		}
		arr, ok := settings["arrayField"].([]any)
		if !ok || len(arr) != 3 {
			t.Error("arrayField not preserved")
		}
		obj, ok := settings["objectField"].(map[string]any)
		if !ok || obj["nested"] != "value" {
			t.Error("objectField not preserved")
		}
	})

	t.Run("preserves extra fields in non-Ollama models", func(t *testing.T) {
		os.RemoveAll(settingsDir)
		os.MkdirAll(settingsDir, 0o755)

		original := `{
			"customModels": [{
				"model": "gpt-4",
				"apiKey": "sk-xxx",
				"extraField": "preserved",
				"nestedExtra": {"foo": "bar"}
			}]
		}`
		os.WriteFile(settingsPath, []byte(original), 0o644)

		if err := d.Edit([]string{"llama3"}); err != nil {
			t.Fatal(err)
		}

		settings := readSettings()
		models := settings["customModels"].([]any)
		gpt4 := models[1].(map[string]any) // non-Ollama is second

		if gpt4["extraField"] != "preserved" {
			t.Error("extraField not preserved")
		}
		nested := gpt4["nestedExtra"].(map[string]any)
		if nested["foo"] != "bar" {
			t.Error("nestedExtra not preserved")
		}
	})
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

func TestDroidEdit_Idempotent(t *testing.T) {
	d := &Droid{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	settingsDir := filepath.Join(tmpDir, ".factory")
	settingsPath := filepath.Join(settingsDir, "settings.json")

	os.MkdirAll(settingsDir, 0o755)
	os.WriteFile(settingsPath, []byte(testDroidSettingsFixture), 0o644)

	// Edit twice with same models
	d.Edit([]string{"llama3", "mistral"})
	firstData, _ := os.ReadFile(settingsPath)

	d.Edit([]string{"llama3", "mistral"})
	secondData, _ := os.ReadFile(settingsPath)

	// Results should be identical
	if string(firstData) != string(secondData) {
		t.Error("repeated edits with same models produced different results")
	}
}

func TestDroidEdit_MultipleConsecutiveEdits(t *testing.T) {
	d := &Droid{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	settingsDir := filepath.Join(tmpDir, ".factory")
	settingsPath := filepath.Join(settingsDir, "settings.json")

	os.MkdirAll(settingsDir, 0o755)
	os.WriteFile(settingsPath, []byte(testDroidSettingsFixture), 0o644)

	// Multiple edits shouldn't accumulate garbage or lose data
	for i := range 10 {
		models := []string{"model-a", "model-b"}
		if i%2 == 0 {
			models = []string{"model-x", "model-y", "model-z"}
		}
		if err := d.Edit(models); err != nil {
			t.Fatalf("edit %d failed: %v", i, err)
		}
	}

	// Verify file is still valid JSON and preserves original fields
	data, _ := os.ReadFile(settingsPath)
	var settings map[string]any
	if err := json.Unmarshal(data, &settings); err != nil {
		t.Fatalf("file is not valid JSON after multiple edits: %v", err)
	}

	// Original fields should still be there
	if settings["diffMode"] != "github" {
		t.Error("diffMode lost after multiple edits")
	}
	if settings["enableHooks"] != true {
		t.Error("enableHooks lost after multiple edits")
	}

	// Non-Ollama model should still be preserved
	models := settings["customModels"].([]any)
	foundOther := false
	for _, m := range models {
		if entry, ok := m.(map[string]any); ok {
			if entry["model"] == "gpt-4" {
				foundOther = true
				if entry["customField"] != "should be preserved" {
					t.Error("other customField lost after multiple edits")
				}
			}
		}
	}
	if !foundOther {
		t.Error("other model lost after multiple edits")
	}
}

func TestDroidEdit_UnicodeAndSpecialCharacters(t *testing.T) {
	d := &Droid{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	settingsDir := filepath.Join(tmpDir, ".factory")
	settingsPath := filepath.Join(settingsDir, "settings.json")

	os.MkdirAll(settingsDir, 0o755)

	// Settings with unicode and special characters
	original := `{
		"userName": "日本語テスト",
		"emoji": "🚀🎉💻",
		"specialChars": "quotes: \"test\" and 'test', backslash: \\, newline: \n, tab: \t",
		"unicodeEscape": "\u0048\u0065\u006c\u006c\u006f",
		"customModels": [],
		"sessionDefaultSettings": {}
	}`
	os.WriteFile(settingsPath, []byte(original), 0o644)

	if err := d.Edit([]string{"model-a"}); err != nil {
		t.Fatal(err)
	}

	data, _ := os.ReadFile(settingsPath)
	var settings map[string]any
	json.Unmarshal(data, &settings)

	if settings["userName"] != "日本語テスト" {
		t.Error("Japanese characters not preserved")
	}
	if settings["emoji"] != "🚀🎉💻" {
		t.Error("emoji not preserved")
	}
	// Note: JSON encoding will normalize escape sequences
	if settings["unicodeEscape"] != "Hello" {
		t.Error("unicode escape sequence not preserved")
	}
}

func TestDroidEdit_LargeNumbers(t *testing.T) {
	d := &Droid{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	settingsDir := filepath.Join(tmpDir, ".factory")
	settingsPath := filepath.Join(settingsDir, "settings.json")

	os.MkdirAll(settingsDir, 0o755)

	// Large numbers and timestamps (common in settings files)
	original := `{
		"timestamp": 1763081579486,
		"largeInt": 9007199254740991,
		"negativeNum": -12345,
		"floatNum": 3.141592653589793,
		"scientificNotation": 1.23e10,
		"customModels": [],
		"sessionDefaultSettings": {}
	}`
	os.WriteFile(settingsPath, []byte(original), 0o644)

	if err := d.Edit([]string{"model-a"}); err != nil {
		t.Fatal(err)
	}

	data, _ := os.ReadFile(settingsPath)
	var settings map[string]any
	json.Unmarshal(data, &settings)

	if settings["timestamp"] != float64(1763081579486) {
		t.Errorf("timestamp not preserved: got %v", settings["timestamp"])
	}
	if settings["largeInt"] != float64(9007199254740991) {
		t.Errorf("largeInt not preserved: got %v", settings["largeInt"])
	}
	if settings["negativeNum"] != float64(-12345) {
		t.Error("negativeNum not preserved")
	}
	if settings["floatNum"] != 3.141592653589793 {
		t.Error("floatNum not preserved")
	}
}

func TestDroidEdit_EmptyAndNullValues(t *testing.T) {
	d := &Droid{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	settingsDir := filepath.Join(tmpDir, ".factory")
	settingsPath := filepath.Join(settingsDir, "settings.json")

	os.MkdirAll(settingsDir, 0o755)

	original := `{
		"emptyString": "",
		"nullValue": null,
		"emptyArray": [],
		"emptyObject": {},
		"falseBool": false,
		"zeroNumber": 0,
		"customModels": [],
		"sessionDefaultSettings": {}
	}`
	os.WriteFile(settingsPath, []byte(original), 0o644)

	if err := d.Edit([]string{"model-a"}); err != nil {
		t.Fatal(err)
	}

	data, _ := os.ReadFile(settingsPath)
	var settings map[string]any
	json.Unmarshal(data, &settings)

	if settings["emptyString"] != "" {
		t.Error("emptyString not preserved")
	}
	if settings["nullValue"] != nil {
		t.Error("nullValue not preserved as null")
	}
	if arr, ok := settings["emptyArray"].([]any); !ok || len(arr) != 0 {
		t.Error("emptyArray not preserved")
	}
	if obj, ok := settings["emptyObject"].(map[string]any); !ok || len(obj) != 0 {
		t.Error("emptyObject not preserved")
	}
	if settings["falseBool"] != false {
		t.Error("falseBool not preserved (false vs missing)")
	}
	if settings["zeroNumber"] != float64(0) {
		t.Error("zeroNumber not preserved")
	}
}

func TestDroidEdit_DeeplyNestedStructures(t *testing.T) {
	d := &Droid{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	settingsDir := filepath.Join(tmpDir, ".factory")
	settingsPath := filepath.Join(settingsDir, "settings.json")

	os.MkdirAll(settingsDir, 0o755)

	original := `{
		"level1": {
			"level2": {
				"level3": {
					"level4": {
						"deepValue": "found me",
						"deepArray": [1, 2, {"nested": true}]
					}
				}
			}
		},
		"customModels": [],
		"sessionDefaultSettings": {}
	}`
	os.WriteFile(settingsPath, []byte(original), 0o644)

	if err := d.Edit([]string{"model-a"}); err != nil {
		t.Fatal(err)
	}

	data, _ := os.ReadFile(settingsPath)
	var settings map[string]any
	json.Unmarshal(data, &settings)

	// Navigate to deeply nested value
	l1 := settings["level1"].(map[string]any)
	l2 := l1["level2"].(map[string]any)
	l3 := l2["level3"].(map[string]any)
	l4 := l3["level4"].(map[string]any)

	if l4["deepValue"] != "found me" {
		t.Error("deeply nested value not preserved")
	}

	deepArray := l4["deepArray"].([]any)
	if len(deepArray) != 3 {
		t.Error("deeply nested array not preserved")
	}
	nestedInArray := deepArray[2].(map[string]any)
	if nestedInArray["nested"] != true {
		t.Error("object nested in array not preserved")
	}
}

func TestDroidEdit_ModelNamesWithSpecialCharacters(t *testing.T) {
	d := &Droid{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	settingsDir := filepath.Join(tmpDir, ".factory")
	settingsPath := filepath.Join(settingsDir, "settings.json")

	// Test model names with colons, slashes, special chars
	specialModels := []string{
		"qwen3:480b-cloud",
		"llama3.2:70b",
		"model/with/slashes",
		"model-with-dashes",
		"model_with_underscores",
	}

	if err := d.Edit(specialModels); err != nil {
		t.Fatal(err)
	}

	data, _ := os.ReadFile(settingsPath)
	var settings map[string]any
	json.Unmarshal(data, &settings)

	models := settings["customModels"].([]any)
	if len(models) != len(specialModels) {
		t.Fatalf("expected %d models, got %d", len(specialModels), len(models))
	}

	for i, expected := range specialModels {
		m := models[i].(map[string]any)
		if m["model"] != expected {
			t.Errorf("model %d: expected %s, got %s", i, expected, m["model"])
		}
	}
}

func TestDroidEdit_MissingCustomModelsKey(t *testing.T) {
	d := &Droid{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	settingsDir := filepath.Join(tmpDir, ".factory")
	settingsPath := filepath.Join(settingsDir, "settings.json")

	os.MkdirAll(settingsDir, 0o755)

	// No customModels key at all
	original := `{
		"diffMode": "github",
		"sessionDefaultSettings": {"autonomyMode": "auto-high"}
	}`
	os.WriteFile(settingsPath, []byte(original), 0o644)

	if err := d.Edit([]string{"model-a"}); err != nil {
		t.Fatal(err)
	}

	data, _ := os.ReadFile(settingsPath)
	var settings map[string]any
	json.Unmarshal(data, &settings)

	// Original fields preserved
	if settings["diffMode"] != "github" {
		t.Error("diffMode not preserved")
	}

	// customModels created
	models, ok := settings["customModels"].([]any)
	if !ok || len(models) != 1 {
		t.Error("customModels not created properly")
	}
}

func TestDroidEdit_NullCustomModels(t *testing.T) {
	d := &Droid{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	settingsDir := filepath.Join(tmpDir, ".factory")
	settingsPath := filepath.Join(settingsDir, "settings.json")

	os.MkdirAll(settingsDir, 0o755)

	original := `{
		"customModels": null,
		"sessionDefaultSettings": {}
	}`
	os.WriteFile(settingsPath, []byte(original), 0o644)

	if err := d.Edit([]string{"model-a"}); err != nil {
		t.Fatal(err)
	}

	data, _ := os.ReadFile(settingsPath)
	var settings map[string]any
	json.Unmarshal(data, &settings)

	models, ok := settings["customModels"].([]any)
	if !ok || len(models) != 1 {
		t.Error("null customModels not handled properly")
	}
}

func TestDroidEdit_MinifiedJSON(t *testing.T) {
	d := &Droid{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	settingsDir := filepath.Join(tmpDir, ".factory")
	settingsPath := filepath.Join(settingsDir, "settings.json")

	os.MkdirAll(settingsDir, 0o755)

	// Minified JSON (no whitespace)
	original := `{"diffMode":"github","enableHooks":true,"hooks":{"imported":["cmd1","cmd2"]},"customModels":[],"sessionDefaultSettings":{}}`
	os.WriteFile(settingsPath, []byte(original), 0o644)

	if err := d.Edit([]string{"model-a"}); err != nil {
		t.Fatal(err)
	}

	data, _ := os.ReadFile(settingsPath)
	var settings map[string]any
	if err := json.Unmarshal(data, &settings); err != nil {
		t.Fatal("output is not valid JSON")
	}

	if settings["diffMode"] != "github" {
		t.Error("diffMode not preserved from minified JSON")
	}
	if settings["enableHooks"] != true {
		t.Error("enableHooks not preserved from minified JSON")
	}
}

func TestDroidEdit_CreatesDirectoryIfMissing(t *testing.T) {
	d := &Droid{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	settingsDir := filepath.Join(tmpDir, ".factory")

	// Directory doesn't exist
	if _, err := os.Stat(settingsDir); !os.IsNotExist(err) {
		t.Fatal("directory should not exist before test")
	}

	if err := d.Edit([]string{"model-a"}); err != nil {
		t.Fatal(err)
	}

	// Directory should be created
	if _, err := os.Stat(settingsDir); os.IsNotExist(err) {
		t.Fatal("directory was not created")
	}

	// File should exist and be valid
	settingsPath := filepath.Join(settingsDir, "settings.json")
	data, err := os.ReadFile(settingsPath)
	if err != nil {
		t.Fatal("settings file not created")
	}

	var settings map[string]any
	if err := json.Unmarshal(data, &settings); err != nil {
		t.Fatal("created file is not valid JSON")
	}
}

func TestDroidEdit_PreservesFileAfterError(t *testing.T) {
	d := &Droid{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	settingsDir := filepath.Join(tmpDir, ".factory")
	settingsPath := filepath.Join(settingsDir, "settings.json")

	os.MkdirAll(settingsDir, 0o755)

	// Valid original content
	original := `{"diffMode": "github", "customModels": [], "sessionDefaultSettings": {}}`
	os.WriteFile(settingsPath, []byte(original), 0o644)

	// Empty models list is a no-op, should not modify file
	d.Edit([]string{})

	data, _ := os.ReadFile(settingsPath)
	if string(data) != original {
		t.Error("file was modified when it should not have been")
	}
}

func TestDroidEdit_BackupCreated(t *testing.T) {
	d := &Droid{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	settingsDir := filepath.Join(tmpDir, ".factory")
	settingsPath := filepath.Join(settingsDir, "settings.json")
	backupDir := filepath.Join(os.TempDir(), "ollama-backups")

	os.MkdirAll(settingsDir, 0o755)

	// Use a unique marker to identify our backup
	uniqueMarker := fmt.Sprintf("test-marker-%d", os.Getpid())
	original := fmt.Sprintf(`{"diffMode": "%s", "customModels": [], "sessionDefaultSettings": {}}`, uniqueMarker)
	os.WriteFile(settingsPath, []byte(original), 0o644)

	if err := d.Edit([]string{"model-a"}); err != nil {
		t.Fatal(err)
	}

	// Find backup containing our unique marker
	backups, _ := filepath.Glob(filepath.Join(backupDir, "settings.json.*"))
	foundBackup := false
	for _, backup := range backups {
		data, err := os.ReadFile(backup)
		if err != nil {
			continue
		}
		if string(data) == original {
			foundBackup = true
			break
		}
	}

	if !foundBackup {
		t.Error("backup with original content not found")
	}

	// Main file should be modified
	newData, _ := os.ReadFile(settingsPath)
	var settings map[string]any
	json.Unmarshal(newData, &settings)

	models := settings["customModels"].([]any)
	if len(models) != 1 {
		t.Error("main file was not updated")
	}
}

func TestDroidEdit_LargeNumberOfModels(t *testing.T) {
	d := &Droid{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	settingsDir := filepath.Join(tmpDir, ".factory")
	settingsPath := filepath.Join(settingsDir, "settings.json")

	os.MkdirAll(settingsDir, 0o755)
	os.WriteFile(settingsPath, []byte(`{"customModels": [], "sessionDefaultSettings": {}}`), 0o644)

	// Add many models
	var models []string
	for i := range 100 {
		models = append(models, fmt.Sprintf("model-%d", i))
	}

	if err := d.Edit(models); err != nil {
		t.Fatal(err)
	}

	data, _ := os.ReadFile(settingsPath)
	var settings map[string]any
	json.Unmarshal(data, &settings)

	customModels := settings["customModels"].([]any)
	if len(customModels) != 100 {
		t.Errorf("expected 100 models, got %d", len(customModels))
	}

	// Verify indices are correct
	for i, m := range customModels {
		entry := m.(map[string]any)
		if entry["index"] != float64(i) {
			t.Errorf("model %d has wrong index: %v", i, entry["index"])
		}
	}
}

func TestDroidEdit_LocalModelDefaultMaxOutput(t *testing.T) {
	d := &Droid{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	settingsDir := filepath.Join(tmpDir, ".factory")
	settingsPath := filepath.Join(settingsDir, "settings.json")

	if err := d.Edit([]string{"llama3.2"}); err != nil {
		t.Fatal(err)
	}

	data, _ := os.ReadFile(settingsPath)
	var settings map[string]any
	json.Unmarshal(data, &settings)

	models := settings["customModels"].([]any)
	entry := models[0].(map[string]any)
	if entry["maxOutputTokens"] != float64(64000) {
		t.Errorf("local model maxOutputTokens = %v, want 64000", entry["maxOutputTokens"])
	}
}

func TestDroidEdit_CloudModelLimitsUsed(t *testing.T) {
	// Verify that every cloud model in cloudModelLimits has a valid output
	// value that would be used for maxOutputTokens when isCloudModel returns true.
	// :cloud suffix stripping must also work since that's how users specify them.
	for name, expected := range cloudModelLimits {
		t.Run(name, func(t *testing.T) {
			l, ok := lookupCloudModelLimit(name)
			if !ok {
				t.Fatalf("lookupCloudModelLimit(%q) returned false", name)
			}
			if l.Output != expected.Output {
				t.Errorf("output = %d, want %d", l.Output, expected.Output)
			}
			// Also verify :cloud suffix lookup
			cloudName := name + ":cloud"
			l2, ok := lookupCloudModelLimit(cloudName)
			if !ok {
				t.Fatalf("lookupCloudModelLimit(%q) returned false", cloudName)
			}
			if l2.Output != expected.Output {
				t.Errorf(":cloud output = %d, want %d", l2.Output, expected.Output)
			}
		})
	}
}

func TestDroidEdit_ArraysWithMixedTypes(t *testing.T) {
	d := &Droid{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	settingsDir := filepath.Join(tmpDir, ".factory")
	settingsPath := filepath.Join(settingsDir, "settings.json")

	os.MkdirAll(settingsDir, 0o755)

	// Arrays with mixed types (valid JSON)
	original := `{
		"mixedArray": [1, "two", true, null, {"nested": "obj"}, [1,2,3]],
		"customModels": [],
		"sessionDefaultSettings": {}
	}`
	os.WriteFile(settingsPath, []byte(original), 0o644)

	if err := d.Edit([]string{"model-a"}); err != nil {
		t.Fatal(err)
	}

	data, _ := os.ReadFile(settingsPath)
	var settings map[string]any
	json.Unmarshal(data, &settings)

	arr := settings["mixedArray"].([]any)
	if len(arr) != 6 {
		t.Error("mixedArray length not preserved")
	}
	if arr[0] != float64(1) {
		t.Error("number in mixed array not preserved")
	}
	if arr[1] != "two" {
		t.Error("string in mixed array not preserved")
	}
	if arr[2] != true {
		t.Error("bool in mixed array not preserved")
	}
	if arr[3] != nil {
		t.Error("null in mixed array not preserved")
	}
	if nested, ok := arr[4].(map[string]any); !ok || nested["nested"] != "obj" {
		t.Error("object in mixed array not preserved")
	}
	if innerArr, ok := arr[5].([]any); !ok || len(innerArr) != 3 {
		t.Error("array in mixed array not preserved")
	}
}
