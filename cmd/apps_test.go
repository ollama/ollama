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
		if err := setupOpenCodeSettings("llama3.2"); err != nil {
			t.Fatal(err)
		}
		assertOpenCodeModelExists(t, configPath, "llama3.2")
		assertOpenCodeRecentModel(t, statePath, 0, "ollama", "llama3.2")
	})

	t.Run("preserve other providers", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0755)
		os.WriteFile(configPath, []byte(`{"provider":{"anthropic":{"apiKey":"xxx"}}}`), 0644)
		if err := setupOpenCodeSettings("llama3.2"); err != nil {
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
		if err := setupOpenCodeSettings("llama3.2"); err != nil {
			t.Fatal(err)
		}
		assertOpenCodeModelExists(t, configPath, "mistral")
		assertOpenCodeModelExists(t, configPath, "llama3.2")
	})

	t.Run("update existing model", func(t *testing.T) {
		cleanup()
		setupOpenCodeSettings("llama3.2")
		setupOpenCodeSettings("llama3.2")
		assertOpenCodeModelExists(t, configPath, "llama3.2")
	})

	t.Run("preserve top-level keys", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0755)
		os.WriteFile(configPath, []byte(`{"theme":"dark","keybindings":{}}`), 0644)
		if err := setupOpenCodeSettings("llama3.2"); err != nil {
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
		if err := setupOpenCodeSettings("llama3.2"); err != nil {
			t.Fatal(err)
		}
		assertOpenCodeRecentModel(t, statePath, 0, "ollama", "llama3.2")
		assertOpenCodeRecentModel(t, statePath, 1, "anthropic", "claude")
	})

	t.Run("model state - preserve favorites and variants", func(t *testing.T) {
		cleanup()
		os.MkdirAll(stateDir, 0755)
		os.WriteFile(statePath, []byte(`{"recent":[],"favorite":[{"providerID":"x","modelID":"y"}],"variant":{"a":"b"}}`), 0644)
		if err := setupOpenCodeSettings("llama3.2"); err != nil {
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
		if err := setupOpenCodeSettings("llama3.2"); err != nil {
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
