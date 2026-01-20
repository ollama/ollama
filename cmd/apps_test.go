package cmd

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestSetupOpenCodeSettings(t *testing.T) {
	tmpDir := t.TempDir()
	t.Setenv("HOME", tmpDir)

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
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(configPath, []byte(`{"provider":{"anthropic":{"apiKey":"xxx"}}}`), 0o644)
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
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(configPath, []byte(`{"provider":{"ollama":{"models":{"mistral":{"name":"Mistral"}}}}}`), 0o644)
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
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(configPath, []byte(`{"theme":"dark","keybindings":{}}`), 0o644)
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
		os.MkdirAll(stateDir, 0o755)
		os.WriteFile(statePath, []byte(`{"recent":[{"providerID":"anthropic","modelID":"claude"}],"favorite":[],"variant":{}}`), 0o644)
		if err := setupOpenCodeSettings([]string{"llama3.2"}); err != nil {
			t.Fatal(err)
		}
		assertOpenCodeRecentModel(t, statePath, 0, "ollama", "llama3.2")
		assertOpenCodeRecentModel(t, statePath, 1, "anthropic", "claude")
	})

	t.Run("model state - preserve favorites and variants", func(t *testing.T) {
		cleanup()
		os.MkdirAll(stateDir, 0o755)
		os.WriteFile(statePath, []byte(`{"recent":[],"favorite":[{"providerID":"x","modelID":"y"}],"variant":{"a":"b"}}`), 0o644)
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
		os.MkdirAll(stateDir, 0o755)
		os.WriteFile(statePath, []byte(`{"recent":[{"providerID":"ollama","modelID":"llama3.2"},{"providerID":"anthropic","modelID":"claude"}],"favorite":[],"variant":{}}`), 0o644)
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
		os.MkdirAll(configDir, 0o755)
		// Add a non-Ollama model manually
		os.WriteFile(configPath, []byte(`{"provider":{"ollama":{"models":{"external":{"name":"External Model"}}}}}`), 0o644)

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
	t.Setenv("HOME", tmpDir)

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

	t.Run("creates backup in /tmp/ollama-backups", func(t *testing.T) {
		path := filepath.Join(tmpDir, "backup.json")

		// Write initial file
		os.WriteFile(path, []byte(`{"original": true}`), 0o644)

		// Update with atomicWriteJSON
		if err := atomicWriteJSON(path, map[string]bool{"updated": true}); err != nil {
			t.Fatal(err)
		}

		// Check backup exists in /tmp/ollama-backups/ with original content
		entries, err := os.ReadDir(backupDir)
		if err != nil {
			t.Fatal("backup directory not created")
		}

		var foundBackup bool
		for _, entry := range entries {
			if filepath.Ext(entry.Name()) != ".json" {
				// Look for backup.json.<timestamp>
				name := entry.Name()
				if len(name) > len("backup.json.") && name[:len("backup.json.")] == "backup.json." {
					backupPath := filepath.Join(backupDir, name)
					backup, err := os.ReadFile(backupPath)
					if err == nil {
						var backupData map[string]bool
						json.Unmarshal(backup, &backupData)
						if backupData["original"] {
							foundBackup = true
							// Clean up after test
							os.Remove(backupPath)
							break
						}
					}
				}
			}
		}

		if !foundBackup {
			t.Error("backup file not created in /tmp/ollama-backups")
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

		if err := atomicWriteJSON(path, map[string]string{"new": "file"}); err != nil {
			t.Fatal(err)
		}

		// Check no backup was created for this specific file
		entries, _ := os.ReadDir(backupDir)
		for _, entry := range entries {
			if len(entry.Name()) > len("nobak.json.") && entry.Name()[:len("nobak.json.")] == "nobak.json." {
				t.Error("backup should not exist for new file")
			}
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

		data := map[string]string{"key": "value"}

		// First write
		if err := atomicWriteJSON(path, data); err != nil {
			t.Fatal(err)
		}

		// Count backups before
		entries1, _ := os.ReadDir(backupDir)
		countBefore := 0
		for _, e := range entries1 {
			if len(e.Name()) > len("unchanged.json.") && e.Name()[:len("unchanged.json.")] == "unchanged.json." {
				countBefore++
			}
		}

		// Second write with same content
		if err := atomicWriteJSON(path, data); err != nil {
			t.Fatal(err)
		}

		// Count backups after - should be same (no new backup created)
		entries2, _ := os.ReadDir(backupDir)
		countAfter := 0
		for _, e := range entries2 {
			if len(e.Name()) > len("unchanged.json.") && e.Name()[:len("unchanged.json.")] == "unchanged.json." {
				countAfter++
			}
		}

		if countAfter != countBefore {
			t.Errorf("backup was created when content unchanged (before=%d, after=%d)", countBefore, countAfter)
		}
	})

	t.Run("backup filename contains unix timestamp", func(t *testing.T) {
		path := filepath.Join(tmpDir, "timestamped.json")

		os.WriteFile(path, []byte(`{"v": 1}`), 0o644)
		if err := atomicWriteJSON(path, map[string]int{"v": 2}); err != nil {
			t.Fatal(err)
		}

		entries, _ := os.ReadDir(backupDir)
		var found bool
		for _, entry := range entries {
			name := entry.Name()
			if len(name) > len("timestamped.json.") && name[:len("timestamped.json.")] == "timestamped.json." {
				// Extract timestamp part and verify it's numeric
				timestamp := name[len("timestamped.json."):]
				for _, c := range timestamp {
					if c < '0' || c > '9' {
						t.Errorf("backup filename timestamp contains non-numeric character: %s", name)
					}
				}
				found = true
				os.Remove(filepath.Join(backupDir, name))
				break
			}
		}
		if !found {
			t.Error("backup file with timestamp not found")
		}
	})
}

func TestIntegrationConfig(t *testing.T) {
	tmpDir := t.TempDir()
	t.Setenv("HOME", tmpDir)

	t.Run("save and load round-trip", func(t *testing.T) {
		models := []string{"llama3.2", "mistral", "qwen2.5"}
		if err := SaveIntegration("claude", models); err != nil {
			t.Fatal(err)
		}

		config, err := LoadIntegration("claude")
		if err != nil {
			t.Fatal(err)
		}

		if len(config.Models) != len(models) {
			t.Errorf("expected %d models, got %d", len(models), len(config.Models))
		}
		for i, m := range models {
			if config.Models[i] != m {
				t.Errorf("model %d: expected %s, got %s", i, m, config.Models[i])
			}
		}
	})

	t.Run("DefaultModel returns first model", func(t *testing.T) {
		SaveIntegration("codex", []string{"model-a", "model-b"})

		config, _ := LoadIntegration("codex")
		if config.DefaultModel() != "model-a" {
			t.Errorf("expected model-a, got %s", config.DefaultModel())
		}
	})

	t.Run("DefaultModel returns empty for no models", func(t *testing.T) {
		config := &IntegrationConfig{Models: []string{}}
		if config.DefaultModel() != "" {
			t.Errorf("expected empty string, got %s", config.DefaultModel())
		}
	})

	t.Run("app name is case-insensitive", func(t *testing.T) {
		SaveIntegration("Claude", []string{"model-x"})

		config, err := LoadIntegration("claude")
		if err != nil {
			t.Fatal(err)
		}
		if config.DefaultModel() != "model-x" {
			t.Errorf("expected model-x, got %s", config.DefaultModel())
		}
	})
}

func TestGetExistingConfigPaths(t *testing.T) {
	tmpDir := t.TempDir()
	t.Setenv("HOME", tmpDir)

	t.Run("returns empty for claude (no config files)", func(t *testing.T) {
		paths := getExistingConfigPaths("claude")
		if len(paths) != 0 {
			t.Errorf("expected no paths for claude, got %v", paths)
		}
	})

	t.Run("returns empty for codex (no config files)", func(t *testing.T) {
		paths := getExistingConfigPaths("codex")
		if len(paths) != 0 {
			t.Errorf("expected no paths for codex, got %v", paths)
		}
	})

	t.Run("returns empty for droid when no config exists", func(t *testing.T) {
		paths := getExistingConfigPaths("droid")
		if len(paths) != 0 {
			t.Errorf("expected no paths, got %v", paths)
		}
	})

	t.Run("returns path for droid when config exists", func(t *testing.T) {
		settingsDir := filepath.Join(tmpDir, ".factory")
		os.MkdirAll(settingsDir, 0o755)
		os.WriteFile(filepath.Join(settingsDir, "settings.json"), []byte(`{}`), 0o644)

		paths := getExistingConfigPaths("droid")
		if len(paths) != 1 {
			t.Errorf("expected 1 path, got %d", len(paths))
		}
	})

	t.Run("returns paths for opencode when configs exist", func(t *testing.T) {
		configDir := filepath.Join(tmpDir, ".config", "opencode")
		stateDir := filepath.Join(tmpDir, ".local", "state", "opencode")
		os.MkdirAll(configDir, 0o755)
		os.MkdirAll(stateDir, 0o755)
		os.WriteFile(filepath.Join(configDir, "opencode.json"), []byte(`{}`), 0o644)
		os.WriteFile(filepath.Join(stateDir, "model.json"), []byte(`{}`), 0o644)

		paths := getExistingConfigPaths("opencode")
		if len(paths) != 2 {
			t.Errorf("expected 2 paths, got %d: %v", len(paths), paths)
		}
	})

	t.Run("case insensitive app name", func(t *testing.T) {
		paths1 := getExistingConfigPaths("DROID")
		paths2 := getExistingConfigPaths("droid")
		if len(paths1) != len(paths2) {
			t.Error("app name should be case insensitive")
		}
	})
}

func TestListIntegrations(t *testing.T) {
	tmpDir := t.TempDir()
	t.Setenv("HOME", tmpDir)

	t.Run("returns empty when no integrations", func(t *testing.T) {
		configs, err := ListIntegrations()
		if err != nil {
			t.Fatal(err)
		}
		if len(configs) != 0 {
			t.Errorf("expected 0 integrations, got %d", len(configs))
		}
	})

	t.Run("returns all saved integrations", func(t *testing.T) {
		SaveIntegration("claude", []string{"model-1"})
		SaveIntegration("droid", []string{"model-2"})

		configs, err := ListIntegrations()
		if err != nil {
			t.Fatal(err)
		}
		if len(configs) != 2 {
			t.Errorf("expected 2 integrations, got %d", len(configs))
		}
	})
}
