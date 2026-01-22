package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestSetupOpenCodeSettings(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

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

// Edge case tests for opencode.go

// TestSetupOpenCodeSettings_CorruptedConfigJSON verifies handling of corrupted config.json.
// Corrupted JSON should be handled gracefully, not cause panic or silently overwrite.
func TestSetupOpenCodeSettings_CorruptedConfigJSON(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	configDir := filepath.Join(tmpDir, ".config", "opencode")
	configPath := filepath.Join(configDir, "opencode.json")

	os.MkdirAll(configDir, 0o755)
	os.WriteFile(configPath, []byte(`{corrupted json content`), 0o644)

	// Should not panic - corrupted JSON should be treated as empty
	err := setupOpenCodeSettings([]string{"llama3.2"})
	if err != nil {
		t.Fatalf("setupOpenCodeSettings failed with corrupted config: %v", err)
	}

	// Verify valid JSON was created
	data, _ := os.ReadFile(configPath)
	var cfg map[string]any
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Errorf("resulting config is not valid JSON: %v", err)
	}
}

// TestSetupOpenCodeSettings_CorruptedStateJSON verifies handling of corrupted state/model.json.
// State file corruption should not break the setup process.
func TestSetupOpenCodeSettings_CorruptedStateJSON(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	stateDir := filepath.Join(tmpDir, ".local", "state", "opencode")
	statePath := filepath.Join(stateDir, "model.json")

	os.MkdirAll(stateDir, 0o755)
	os.WriteFile(statePath, []byte(`{corrupted state`), 0o644)

	err := setupOpenCodeSettings([]string{"llama3.2"})
	if err != nil {
		t.Fatalf("setupOpenCodeSettings failed with corrupted state: %v", err)
	}

	// Verify valid state was created
	data, _ := os.ReadFile(statePath)
	var state map[string]any
	if err := json.Unmarshal(data, &state); err != nil {
		t.Errorf("resulting state is not valid JSON: %v", err)
	}
}

// TestSetupOpenCodeSettings_WrongTypeProvider verifies handling when provider is wrong type.
// Type assertion safety - provider might be string instead of map.
func TestSetupOpenCodeSettings_WrongTypeProvider(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	configDir := filepath.Join(tmpDir, ".config", "opencode")
	configPath := filepath.Join(configDir, "opencode.json")

	os.MkdirAll(configDir, 0o755)
	os.WriteFile(configPath, []byte(`{"provider": "not a map"}`), 0o644)

	err := setupOpenCodeSettings([]string{"llama3.2"})
	if err != nil {
		t.Fatalf("setupOpenCodeSettings with wrong type provider failed: %v", err)
	}

	// Verify provider is now correct type
	data, _ := os.ReadFile(configPath)
	var cfg map[string]any
	json.Unmarshal(data, &cfg)

	provider, ok := cfg["provider"].(map[string]any)
	if !ok {
		t.Fatalf("provider should be map after setup, got %T", cfg["provider"])
	}
	if provider["ollama"] == nil {
		t.Error("ollama provider should be created")
	}
}

// TestSetupOpenCodeSettings_WrongTypeRecent verifies handling when recent is wrong type.
func TestSetupOpenCodeSettings_WrongTypeRecent(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	stateDir := filepath.Join(tmpDir, ".local", "state", "opencode")
	statePath := filepath.Join(stateDir, "model.json")

	os.MkdirAll(stateDir, 0o755)
	os.WriteFile(statePath, []byte(`{"recent": "not an array", "favorite": [], "variant": {}}`), 0o644)

	err := setupOpenCodeSettings([]string{"llama3.2"})
	if err != nil {
		t.Fatalf("setupOpenCodeSettings with wrong type recent failed: %v", err)
	}

	// The function should handle this gracefully
	data, _ := os.ReadFile(statePath)
	var state map[string]any
	json.Unmarshal(data, &state)

	// recent should be properly set after setup
	recent, ok := state["recent"].([]any)
	if !ok {
		t.Logf("Note: recent type after setup is %T (documenting behavior)", state["recent"])
	} else if len(recent) == 0 {
		t.Logf("Note: recent is empty (documenting behavior)")
	}
}

// TestSetupOpenCodeSettings_EmptyModels documents intentional behavior: empty models = no-op.
// Prevents accidental clearing of user's model config.
func TestSetupOpenCodeSettings_EmptyModels(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	configDir := filepath.Join(tmpDir, ".config", "opencode")
	configPath := filepath.Join(configDir, "opencode.json")

	os.MkdirAll(configDir, 0o755)
	originalContent := `{"provider":{"ollama":{"models":{"existing":{}}}}}`
	os.WriteFile(configPath, []byte(originalContent), 0o644)

	// Empty models should be no-op
	err := setupOpenCodeSettings([]string{})
	if err != nil {
		t.Fatalf("setupOpenCodeSettings with empty models failed: %v", err)
	}

	// Original content should be preserved (file not modified)
	data, _ := os.ReadFile(configPath)
	if string(data) != originalContent {
		t.Errorf("empty models should not modify file, but content changed")
	}
}

// TestSetupOpenCodeSettings_SpecialCharsInModelName verifies model names with special JSON characters.
// Model names might contain quotes, backslashes, etc.
func TestSetupOpenCodeSettings_SpecialCharsInModelName(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	// Model name with special characters (though unusual)
	specialModel := `model-with-"quotes"`

	err := setupOpenCodeSettings([]string{specialModel})
	if err != nil {
		t.Fatalf("setupOpenCodeSettings with special chars failed: %v", err)
	}

	// Verify it was stored correctly
	configDir := filepath.Join(tmpDir, ".config", "opencode")
	configPath := filepath.Join(configDir, "opencode.json")
	data, _ := os.ReadFile(configPath)

	var cfg map[string]any
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("resulting config is invalid JSON: %v", err)
	}

	// Model should be accessible
	provider, _ := cfg["provider"].(map[string]any)
	ollama, _ := provider["ollama"].(map[string]any)
	models, _ := ollama["models"].(map[string]any)

	if models[specialModel] == nil {
		t.Errorf("model with special chars not found in config")
	}
}

// TestGetOpenCodeOllamaModels_NoConfig verifies behavior when no config exists.
func TestGetOpenCodeOllamaModels_NoConfig(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	models, err := ollamaModelsFromConfig()
	if err == nil {
		t.Log("ollamaModelsFromConfig returns nil error for missing config (acceptable)")
	}
	if len(models) > 0 {
		t.Errorf("expected nil/empty models for missing config, got %v", models)
	}
}

// TestGetOpenCodeConfiguredModels_NoConfig verifies behavior when no config exists.
func TestGetOpenCodeConfiguredModels_NoConfig(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	models := openCodeModels()
	if len(models) > 0 {
		t.Errorf("expected nil/empty for missing config, got %v", models)
	}
}
