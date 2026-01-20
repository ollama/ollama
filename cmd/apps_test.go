package cmd

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestSetupOpenCodeSettings(t *testing.T) {
	tmpDir := t.TempDir()
	origHome := os.Getenv("HOME")
	os.Setenv("HOME", tmpDir)
	defer os.Setenv("HOME", origHome)

	configDir := filepath.Join(tmpDir, ".config", "opencode")
	configPath := filepath.Join(configDir, "opencode.json")
	stateDir := filepath.Join(tmpDir, ".local", "state", "opencode")
	statePath := filepath.Join(stateDir, "model.json")

	cleanup := func() {
		os.RemoveAll(configDir)
		os.RemoveAll(stateDir)
	}

	t.Run("fresh install", func(t *testing.T) {
		cleanup()
		if err := setupOpenCodeSettings([]string{"llama3.2"}); err != nil {
			t.Fatal(err)
		}
		assertOpenCodeModelExists(t, configPath, "llama3.2")
		assertOpenCodeRecentModel(t, statePath, 0, "ollama", "llama3.2")
	})

	t.Run("preserve other providers", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0755)
		os.WriteFile(configPath, []byte(`{"provider":{"anthropic":{"apiKey":"xxx"}}}`), 0644)
		if err := setupOpenCodeSettings([]string{"llama3.2"}); err != nil {
			t.Fatal(err)
		}
		data, _ := os.ReadFile(configPath)
		var cfg map[string]any
		json.Unmarshal(data, &cfg)
		provider := cfg["provider"].(map[string]any)
		if provider["anthropic"] == nil {
			t.Error("anthropic provider was removed")
		}
		assertOpenCodeModelExists(t, configPath, "llama3.2")
	})

	t.Run("preserve other models", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0755)
		os.WriteFile(configPath, []byte(`{"provider":{"ollama":{"models":{"mistral":{"name":"Mistral"}}}}}`), 0644)
		if err := setupOpenCodeSettings([]string{"llama3.2"}); err != nil {
			t.Fatal(err)
		}
		assertOpenCodeModelExists(t, configPath, "mistral")
		assertOpenCodeModelExists(t, configPath, "llama3.2")
	})

	t.Run("update existing model", func(t *testing.T) {
		cleanup()
		setupOpenCodeSettings([]string{"llama3.2"})
		setupOpenCodeSettings([]string{"llama3.2"})
		assertOpenCodeModelExists(t, configPath, "llama3.2")
	})

	t.Run("preserve top-level keys", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0755)
		os.WriteFile(configPath, []byte(`{"theme":"dark","keybindings":{}}`), 0644)
		if err := setupOpenCodeSettings([]string{"llama3.2"}); err != nil {
			t.Fatal(err)
		}
		data, _ := os.ReadFile(configPath)
		var cfg map[string]any
		json.Unmarshal(data, &cfg)
		if cfg["theme"] != "dark" {
			t.Error("theme was removed")
		}
		if cfg["keybindings"] == nil {
			t.Error("keybindings was removed")
		}
	})

	t.Run("model state - insert at index 0", func(t *testing.T) {
		cleanup()
		os.MkdirAll(stateDir, 0755)
		os.WriteFile(statePath, []byte(`{"recent":[{"providerID":"anthropic","modelID":"claude"}],"favorite":[],"variant":{}}`), 0644)
		if err := setupOpenCodeSettings([]string{"llama3.2"}); err != nil {
			t.Fatal(err)
		}
		assertOpenCodeRecentModel(t, statePath, 0, "ollama", "llama3.2")
		assertOpenCodeRecentModel(t, statePath, 1, "anthropic", "claude")
	})

	t.Run("model state - preserve favorites and variants", func(t *testing.T) {
		cleanup()
		os.MkdirAll(stateDir, 0755)
		os.WriteFile(statePath, []byte(`{"recent":[],"favorite":[{"providerID":"x","modelID":"y"}],"variant":{"a":"b"}}`), 0644)
		if err := setupOpenCodeSettings([]string{"llama3.2"}); err != nil {
			t.Fatal(err)
		}
		data, _ := os.ReadFile(statePath)
		var state map[string]any
		json.Unmarshal(data, &state)
		if len(state["favorite"].([]any)) != 1 {
			t.Error("favorite was modified")
		}
		if state["variant"].(map[string]any)["a"] != "b" {
			t.Error("variant was modified")
		}
	})

	t.Run("model state - deduplicate on re-add", func(t *testing.T) {
		cleanup()
		os.MkdirAll(stateDir, 0755)
		os.WriteFile(statePath, []byte(`{"recent":[{"providerID":"ollama","modelID":"llama3.2"},{"providerID":"anthropic","modelID":"claude"}],"favorite":[],"variant":{}}`), 0644)
		if err := setupOpenCodeSettings([]string{"llama3.2"}); err != nil {
			t.Fatal(err)
		}
		data, _ := os.ReadFile(statePath)
		var state map[string]any
		json.Unmarshal(data, &state)
		recent := state["recent"].([]any)
		if len(recent) != 2 {
			t.Errorf("expected 2 recent entries, got %d", len(recent))
		}
		assertOpenCodeRecentModel(t, statePath, 0, "ollama", "llama3.2")
	})

	t.Run("remove model", func(t *testing.T) {
		cleanup()
		// First add two models
		setupOpenCodeSettings([]string{"llama3.2", "mistral"})
		assertOpenCodeModelExists(t, configPath, "llama3.2")
		assertOpenCodeModelExists(t, configPath, "mistral")

		// Then remove one by only selecting the other
		setupOpenCodeSettings([]string{"llama3.2"})
		assertOpenCodeModelExists(t, configPath, "llama3.2")
		assertOpenCodeModelNotExists(t, configPath, "mistral")
	})

	t.Run("remove model preserves non-ollama models", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0755)
		// Add a non-Ollama model manually
		os.WriteFile(configPath, []byte(`{"provider":{"ollama":{"models":{"external":{"name":"External Model"}}}}}`), 0644)

		setupOpenCodeSettings([]string{"llama3.2"})
		assertOpenCodeModelExists(t, configPath, "llama3.2")
		assertOpenCodeModelExists(t, configPath, "external") // Should be preserved
	})
}

func assertOpenCodeModelExists(t *testing.T, path, model string) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var cfg map[string]any
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatal(err)
	}
	provider, ok := cfg["provider"].(map[string]any)
	if !ok {
		t.Fatal("provider not found")
	}
	ollama, ok := provider["ollama"].(map[string]any)
	if !ok {
		t.Fatal("ollama provider not found")
	}
	models, ok := ollama["models"].(map[string]any)
	if !ok {
		t.Fatal("models not found")
	}
	if models[model] == nil {
		t.Errorf("model %s not found", model)
	}
}

func assertOpenCodeModelNotExists(t *testing.T, path, model string) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var cfg map[string]any
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatal(err)
	}
	provider, ok := cfg["provider"].(map[string]any)
	if !ok {
		return // No provider means no model
	}
	ollama, ok := provider["ollama"].(map[string]any)
	if !ok {
		return // No ollama means no model
	}
	models, ok := ollama["models"].(map[string]any)
	if !ok {
		return // No models means no model
	}
	if models[model] != nil {
		t.Errorf("model %s should not exist but was found", model)
	}
}

func assertOpenCodeRecentModel(t *testing.T, path string, index int, providerID, modelID string) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var state map[string]any
	if err := json.Unmarshal(data, &state); err != nil {
		t.Fatal(err)
	}
	recent, ok := state["recent"].([]any)
	if !ok {
		t.Fatal("recent not found")
	}
	if index >= len(recent) {
		t.Fatalf("index %d out of range (len=%d)", index, len(recent))
	}
	entry, ok := recent[index].(map[string]any)
	if !ok {
		t.Fatal("entry is not a map")
	}
	if entry["providerID"] != providerID {
		t.Errorf("expected providerID %s, got %s", providerID, entry["providerID"])
	}
	if entry["modelID"] != modelID {
		t.Errorf("expected modelID %s, got %s", modelID, entry["modelID"])
	}
}

func TestSetupDroidSettings(t *testing.T) {
	tmpDir := t.TempDir()
	origHome := os.Getenv("HOME")
	os.Setenv("HOME", tmpDir)
	defer os.Setenv("HOME", origHome)

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
		os.MkdirAll(settingsDir, 0755)
		// Pre-existing non-Ollama model
		os.WriteFile(settingsPath, []byte(`{
			"customModels": [
				{"model": "gpt-4", "displayName": "GPT-4", "provider": "openai"}
			]
		}`), 0644)

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
		os.MkdirAll(settingsDir, 0755)
		os.WriteFile(settingsPath, []byte(`{
			"theme": "dark",
			"enableHooks": true,
			"sessionDefaultSettings": {"autonomyMode": "auto-high"}
		}`), 0644)

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
		os.MkdirAll(settingsDir, 0755)
		// Pre-existing settings with invalid reasoningEffort
		os.WriteFile(settingsPath, []byte(`{
			"sessionDefaultSettings": {"reasoningEffort": "off"}
		}`), 0644)

		setupDroidSettings([]string{"model-a"})

		settings := readSettings()
		session := settings["sessionDefaultSettings"].(map[string]any)

		if session["reasoningEffort"] != "none" {
			t.Errorf("expected reasoningEffort to be fixed to 'none', got %s", session["reasoningEffort"])
		}
	})

	t.Run("preserves valid reasoningEffort", func(t *testing.T) {
		cleanup()
		os.MkdirAll(settingsDir, 0755)
		os.WriteFile(settingsPath, []byte(`{
			"sessionDefaultSettings": {"reasoningEffort": "high"}
		}`), 0644)

		setupDroidSettings([]string{"model-a"})

		settings := readSettings()
		session := settings["sessionDefaultSettings"].(map[string]any)

		if session["reasoningEffort"] != "high" {
			t.Errorf("expected reasoningEffort to remain 'high', got %s", session["reasoningEffort"])
		}
	})
}

func TestAtomicWriteJSON(t *testing.T) {
	tmpDir := t.TempDir()

	t.Run("creates file", func(t *testing.T) {
		path := filepath.Join(tmpDir, "new.json")
		data := map[string]string{"key": "value"}

		if err := atomicWriteJSON(path, data); err != nil {
			t.Fatal(err)
		}

		content, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}

		var result map[string]string
		if err := json.Unmarshal(content, &result); err != nil {
			t.Fatal(err)
		}
		if result["key"] != "value" {
			t.Errorf("expected value, got %s", result["key"])
		}
	})

	t.Run("creates backup", func(t *testing.T) {
		path := filepath.Join(tmpDir, "backup.json")
		backupPath := path + ".bak"

		// Write initial file
		os.WriteFile(path, []byte(`{"original": true}`), 0644)

		// Update with atomicWriteJSON
		if err := atomicWriteJSON(path, map[string]bool{"updated": true}); err != nil {
			t.Fatal(err)
		}

		// Check backup exists with original content
		backup, err := os.ReadFile(backupPath)
		if err != nil {
			t.Fatal("backup file not created")
		}

		var backupData map[string]bool
		json.Unmarshal(backup, &backupData)
		if !backupData["original"] {
			t.Error("backup doesn't contain original data")
		}

		// Check new file has updated content
		current, _ := os.ReadFile(path)
		var currentData map[string]bool
		json.Unmarshal(current, &currentData)
		if !currentData["updated"] {
			t.Error("file doesn't contain updated data")
		}
	})

	t.Run("no backup for new file", func(t *testing.T) {
		path := filepath.Join(tmpDir, "nobak.json")
		backupPath := path + ".bak"

		if err := atomicWriteJSON(path, map[string]string{"new": "file"}); err != nil {
			t.Fatal(err)
		}

		if _, err := os.Stat(backupPath); !os.IsNotExist(err) {
			t.Error("backup should not exist for new file")
		}
	})

	t.Run("valid JSON output", func(t *testing.T) {
		path := filepath.Join(tmpDir, "valid.json")
		data := map[string]any{
			"string": "hello",
			"number": 42,
			"nested": map[string]string{"a": "b"},
		}

		if err := atomicWriteJSON(path, data); err != nil {
			t.Fatal(err)
		}

		content, _ := os.ReadFile(path)
		var parsed map[string]any
		if err := json.Unmarshal(content, &parsed); err != nil {
			t.Errorf("output is not valid JSON: %v", err)
		}
	})

	t.Run("no backup when content unchanged", func(t *testing.T) {
		path := filepath.Join(tmpDir, "unchanged.json")
		backupPath := path + ".bak"

		data := map[string]string{"key": "value"}

		// First write
		if err := atomicWriteJSON(path, data); err != nil {
			t.Fatal(err)
		}

		// Create a "stale" backup to verify it's not overwritten
		os.WriteFile(backupPath, []byte(`{"stale": "backup"}`), 0644)

		// Second write with same content
		if err := atomicWriteJSON(path, data); err != nil {
			t.Fatal(err)
		}

		// Backup should still contain stale content (not overwritten)
		backup, _ := os.ReadFile(backupPath)
		if string(backup) != `{"stale": "backup"}` {
			t.Errorf("backup was overwritten when content unchanged")
		}
	})
}
