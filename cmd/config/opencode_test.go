package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestOpenCodeIntegration(t *testing.T) {
	o := &OpenCode{}

	t.Run("String", func(t *testing.T) {
		if got := o.String(); got != "OpenCode" {
			t.Errorf("String() = %q, want %q", got, "OpenCode")
		}
	})

	t.Run("implements Runner", func(t *testing.T) {
		var _ Runner = o
	})

	t.Run("implements Editor", func(t *testing.T) {
		var _ Editor = o
	})
}

func TestOpenCodeEdit(t *testing.T) {
	o := &OpenCode{}
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
		if err := o.Edit([]string{"llama3.2"}); err != nil {
			t.Fatal(err)
		}
		assertOpenCodeModelExists(t, configPath, "llama3.2")
		assertOpenCodeRecentModel(t, statePath, 0, "ollama", "llama3.2")
	})

	t.Run("preserve other providers", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(configPath, []byte(`{"provider":{"anthropic":{"apiKey":"xxx"}}}`), 0o644)
		if err := o.Edit([]string{"llama3.2"}); err != nil {
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
		if err := o.Edit([]string{"llama3.2"}); err != nil {
			t.Fatal(err)
		}
		assertOpenCodeModelExists(t, configPath, "mistral")
		assertOpenCodeModelExists(t, configPath, "llama3.2")
	})

	t.Run("update existing model", func(t *testing.T) {
		cleanup()
		o.Edit([]string{"llama3.2"})
		o.Edit([]string{"llama3.2"})
		assertOpenCodeModelExists(t, configPath, "llama3.2")
	})

	t.Run("preserve top-level keys", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(configPath, []byte(`{"theme":"dark","keybindings":{}}`), 0o644)
		if err := o.Edit([]string{"llama3.2"}); err != nil {
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
		if err := o.Edit([]string{"llama3.2"}); err != nil {
			t.Fatal(err)
		}
		assertOpenCodeRecentModel(t, statePath, 0, "ollama", "llama3.2")
		assertOpenCodeRecentModel(t, statePath, 1, "anthropic", "claude")
	})

	t.Run("model state - preserve favorites and variants", func(t *testing.T) {
		cleanup()
		os.MkdirAll(stateDir, 0o755)
		os.WriteFile(statePath, []byte(`{"recent":[],"favorite":[{"providerID":"x","modelID":"y"}],"variant":{"a":"b"}}`), 0o644)
		if err := o.Edit([]string{"llama3.2"}); err != nil {
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
		if err := o.Edit([]string{"llama3.2"}); err != nil {
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
		o.Edit([]string{"llama3.2", "mistral"})
		assertOpenCodeModelExists(t, configPath, "llama3.2")
		assertOpenCodeModelExists(t, configPath, "mistral")

		// Then remove one by only selecting the other
		o.Edit([]string{"llama3.2"})
		assertOpenCodeModelExists(t, configPath, "llama3.2")
		assertOpenCodeModelNotExists(t, configPath, "mistral")
	})

	t.Run("preserve user customizations on managed models", func(t *testing.T) {
		cleanup()
		if err := o.Edit([]string{"llama3.2"}); err != nil {
			t.Fatal(err)
		}

		// Add custom fields to the model entry (simulating user edits)
		data, _ := os.ReadFile(configPath)
		var cfg map[string]any
		json.Unmarshal(data, &cfg)
		provider := cfg["provider"].(map[string]any)
		ollama := provider["ollama"].(map[string]any)
		models := ollama["models"].(map[string]any)
		entry := models["llama3.2"].(map[string]any)
		entry["_myPref"] = "custom-value"
		entry["_myNum"] = 42
		configData, _ := json.MarshalIndent(cfg, "", "  ")
		os.WriteFile(configPath, configData, 0o644)

		// Re-run Edit — should preserve custom fields
		if err := o.Edit([]string{"llama3.2"}); err != nil {
			t.Fatal(err)
		}

		data, _ = os.ReadFile(configPath)
		json.Unmarshal(data, &cfg)
		provider = cfg["provider"].(map[string]any)
		ollama = provider["ollama"].(map[string]any)
		models = ollama["models"].(map[string]any)
		entry = models["llama3.2"].(map[string]any)

		if entry["_myPref"] != "custom-value" {
			t.Errorf("_myPref was lost: got %v", entry["_myPref"])
		}
		if entry["_myNum"] != float64(42) {
			t.Errorf("_myNum was lost: got %v", entry["_myNum"])
		}
		if v, ok := entry["_launch"].(bool); !ok || !v {
			t.Errorf("_launch marker missing or false: got %v", entry["_launch"])
		}
	})

	t.Run("migrate legacy [Ollama] suffix entries", func(t *testing.T) {
		cleanup()
		// Write a config with a legacy entry (has [Ollama] suffix but no _launch marker)
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(configPath, []byte(`{"provider":{"ollama":{"models":{"llama3.2":{"name":"llama3.2 [Ollama]"}}}}}`), 0o644)

		if err := o.Edit([]string{"llama3.2"}); err != nil {
			t.Fatal(err)
		}

		data, _ := os.ReadFile(configPath)
		var cfg map[string]any
		json.Unmarshal(data, &cfg)
		provider := cfg["provider"].(map[string]any)
		ollama := provider["ollama"].(map[string]any)
		models := ollama["models"].(map[string]any)
		entry := models["llama3.2"].(map[string]any)

		// _launch marker should be added
		if v, ok := entry["_launch"].(bool); !ok || !v {
			t.Errorf("_launch marker not added during migration: got %v", entry["_launch"])
		}
		// [Ollama] suffix should be stripped
		if name, ok := entry["name"].(string); !ok || name != "llama3.2" {
			t.Errorf("name suffix not stripped: got %q", entry["name"])
		}
	})

	t.Run("remove model preserves non-ollama models", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)
		// Add a non-Ollama model manually
		os.WriteFile(configPath, []byte(`{"provider":{"ollama":{"models":{"external":{"name":"External Model"}}}}}`), 0o644)

		o.Edit([]string{"llama3.2"})
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

func TestOpenCodeEdit_CorruptedConfigJSON(t *testing.T) {
	o := &OpenCode{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	configDir := filepath.Join(tmpDir, ".config", "opencode")
	configPath := filepath.Join(configDir, "opencode.json")

	os.MkdirAll(configDir, 0o755)
	os.WriteFile(configPath, []byte(`{corrupted json content`), 0o644)

	// Should not panic - corrupted JSON should be treated as empty
	err := o.Edit([]string{"llama3.2"})
	if err != nil {
		t.Fatalf("Edit failed with corrupted config: %v", err)
	}

	// Verify valid JSON was created
	data, _ := os.ReadFile(configPath)
	var cfg map[string]any
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Errorf("resulting config is not valid JSON: %v", err)
	}
}

func TestOpenCodeEdit_CorruptedStateJSON(t *testing.T) {
	o := &OpenCode{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	stateDir := filepath.Join(tmpDir, ".local", "state", "opencode")
	statePath := filepath.Join(stateDir, "model.json")

	os.MkdirAll(stateDir, 0o755)
	os.WriteFile(statePath, []byte(`{corrupted state`), 0o644)

	err := o.Edit([]string{"llama3.2"})
	if err != nil {
		t.Fatalf("Edit failed with corrupted state: %v", err)
	}

	// Verify valid state was created
	data, _ := os.ReadFile(statePath)
	var state map[string]any
	if err := json.Unmarshal(data, &state); err != nil {
		t.Errorf("resulting state is not valid JSON: %v", err)
	}
}

func TestOpenCodeEdit_WrongTypeProvider(t *testing.T) {
	o := &OpenCode{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	configDir := filepath.Join(tmpDir, ".config", "opencode")
	configPath := filepath.Join(configDir, "opencode.json")

	os.MkdirAll(configDir, 0o755)
	os.WriteFile(configPath, []byte(`{"provider": "not a map"}`), 0o644)

	err := o.Edit([]string{"llama3.2"})
	if err != nil {
		t.Fatalf("Edit with wrong type provider failed: %v", err)
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

func TestOpenCodeEdit_WrongTypeRecent(t *testing.T) {
	o := &OpenCode{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	stateDir := filepath.Join(tmpDir, ".local", "state", "opencode")
	statePath := filepath.Join(stateDir, "model.json")

	os.MkdirAll(stateDir, 0o755)
	os.WriteFile(statePath, []byte(`{"recent": "not an array", "favorite": [], "variant": {}}`), 0o644)

	err := o.Edit([]string{"llama3.2"})
	if err != nil {
		t.Fatalf("Edit with wrong type recent failed: %v", err)
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

func TestOpenCodeEdit_EmptyModels(t *testing.T) {
	o := &OpenCode{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	configDir := filepath.Join(tmpDir, ".config", "opencode")
	configPath := filepath.Join(configDir, "opencode.json")

	os.MkdirAll(configDir, 0o755)
	originalContent := `{"provider":{"ollama":{"models":{"existing":{}}}}}`
	os.WriteFile(configPath, []byte(originalContent), 0o644)

	// Empty models should be no-op
	err := o.Edit([]string{})
	if err != nil {
		t.Fatalf("Edit with empty models failed: %v", err)
	}

	// Original content should be preserved (file not modified)
	data, _ := os.ReadFile(configPath)
	if string(data) != originalContent {
		t.Errorf("empty models should not modify file, but content changed")
	}
}

func TestOpenCodeEdit_SpecialCharsInModelName(t *testing.T) {
	o := &OpenCode{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	// Model name with special characters (though unusual)
	specialModel := `model-with-"quotes"`

	err := o.Edit([]string{specialModel})
	if err != nil {
		t.Fatalf("Edit with special chars failed: %v", err)
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

func readOpenCodeModel(t *testing.T, configPath, model string) map[string]any {
	t.Helper()
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	var cfg map[string]any
	json.Unmarshal(data, &cfg)
	provider := cfg["provider"].(map[string]any)
	ollama := provider["ollama"].(map[string]any)
	models := ollama["models"].(map[string]any)
	entry, ok := models[model].(map[string]any)
	if !ok {
		t.Fatalf("model %s not found in config", model)
	}
	return entry
}

func TestOpenCodeEdit_LocalModelNoLimit(t *testing.T) {
	o := &OpenCode{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	configPath := filepath.Join(tmpDir, ".config", "opencode", "opencode.json")

	if err := o.Edit([]string{"llama3.2"}); err != nil {
		t.Fatal(err)
	}

	entry := readOpenCodeModel(t, configPath, "llama3.2")
	if entry["limit"] != nil {
		t.Errorf("local model should not have limit set, got %v", entry["limit"])
	}
}

func TestOpenCodeEdit_PreservesUserLimit(t *testing.T) {
	o := &OpenCode{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	configDir := filepath.Join(tmpDir, ".config", "opencode")
	configPath := filepath.Join(configDir, "opencode.json")

	// Set up a model with a user-configured limit
	os.MkdirAll(configDir, 0o755)
	os.WriteFile(configPath, []byte(`{
		"provider": {
			"ollama": {
				"models": {
					"llama3.2": {
						"name": "llama3.2",
						"_launch": true,
						"limit": {"context": 8192, "output": 4096}
					}
				}
			}
		}
	}`), 0o644)

	// Re-edit should preserve the user's limit (not delete it)
	if err := o.Edit([]string{"llama3.2"}); err != nil {
		t.Fatal(err)
	}

	entry := readOpenCodeModel(t, configPath, "llama3.2")
	limit, ok := entry["limit"].(map[string]any)
	if !ok {
		t.Fatal("user-configured limit was removed")
	}
	if limit["context"] != float64(8192) {
		t.Errorf("context limit changed: got %v, want 8192", limit["context"])
	}
	if limit["output"] != float64(4096) {
		t.Errorf("output limit changed: got %v, want 4096", limit["output"])
	}
}

func TestOpenCodeEdit_CloudModelLimitStructure(t *testing.T) {
	// Verify that when a cloud model entry has limits set (as Edit would do),
	// the structure matches what opencode expects and re-edit preserves them.
	o := &OpenCode{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	configDir := filepath.Join(tmpDir, ".config", "opencode")
	configPath := filepath.Join(configDir, "opencode.json")

	expected := cloudModelLimits["glm-4.7"]

	// Simulate a cloud model that already has the limit set by a previous Edit
	os.MkdirAll(configDir, 0o755)
	os.WriteFile(configPath, []byte(fmt.Sprintf(`{
		"provider": {
			"ollama": {
				"models": {
					"glm-4.7:cloud": {
						"name": "glm-4.7:cloud",
						"_launch": true,
						"limit": {"context": %d, "output": %d}
					}
				}
			}
		}
	}`, expected.Context, expected.Output)), 0o644)

	// Re-edit should preserve the cloud model limit
	if err := o.Edit([]string{"glm-4.7:cloud"}); err != nil {
		t.Fatal(err)
	}

	entry := readOpenCodeModel(t, configPath, "glm-4.7:cloud")
	limit, ok := entry["limit"].(map[string]any)
	if !ok {
		t.Fatal("cloud model limit was removed on re-edit")
	}
	if limit["context"] != float64(expected.Context) {
		t.Errorf("context = %v, want %d", limit["context"], expected.Context)
	}
	if limit["output"] != float64(expected.Output) {
		t.Errorf("output = %v, want %d", limit["output"], expected.Output)
	}
}

func TestLookupCloudModelLimit(t *testing.T) {
	tests := []struct {
		name        string
		wantOK      bool
		wantContext int
		wantOutput  int
	}{
		{"glm-4.7", true, 202_752, 131_072},
		{"glm-4.7:cloud", true, 202_752, 131_072},
		{"kimi-k2.5", true, 262_144, 262_144},
		{"kimi-k2.5:cloud", true, 262_144, 262_144},
		{"deepseek-v3.2", true, 163_840, 65_536},
		{"deepseek-v3.2:cloud", true, 163_840, 65_536},
		{"qwen3-coder:480b", true, 262_144, 65_536},
		{"qwen3-coder-next:cloud", true, 262_144, 32_768},
		{"llama3.2", false, 0, 0},
		{"unknown-model:cloud", false, 0, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l, ok := lookupCloudModelLimit(tt.name)
			if ok != tt.wantOK {
				t.Errorf("lookupCloudModelLimit(%q) ok = %v, want %v", tt.name, ok, tt.wantOK)
			}
			if ok {
				if l.Context != tt.wantContext {
					t.Errorf("context = %d, want %d", l.Context, tt.wantContext)
				}
				if l.Output != tt.wantOutput {
					t.Errorf("output = %d, want %d", l.Output, tt.wantOutput)
				}
			}
		})
	}
}

func TestOpenCodeModels_NoConfig(t *testing.T) {
	o := &OpenCode{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	models := o.Models()
	if len(models) > 0 {
		t.Errorf("expected nil/empty for missing config, got %v", models)
	}
}
