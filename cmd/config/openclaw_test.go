package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestOpenclawIntegration(t *testing.T) {
	c := &Openclaw{}

	t.Run("String", func(t *testing.T) {
		if got := c.String(); got != "Openclaw" {
			t.Errorf("String() = %q, want %q", got, "Openclaw")
		}
	})

	t.Run("implements Runner", func(t *testing.T) {
		var _ Runner = c
	})

	t.Run("implements Editor", func(t *testing.T) {
		var _ Editor = c
	})
}

func TestOpenclawEdit(t *testing.T) {
	c := &Openclaw{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	configDir := filepath.Join(tmpDir, ".clawdbot")
	configPath := filepath.Join(configDir, "clawdbot.json")

	cleanup := func() { os.RemoveAll(configDir) }

	t.Run("fresh install", func(t *testing.T) {
		cleanup()
		if err := c.Edit([]string{"llama3.2"}); err != nil {
			t.Fatal(err)
		}
		assertOpenclawModelExists(t, configPath, "llama3.2")
		assertOpenclawPrimaryModel(t, configPath, "ollama/llama3.2")
	})

	t.Run("multiple models - first is primary", func(t *testing.T) {
		cleanup()
		if err := c.Edit([]string{"llama3.2", "mistral"}); err != nil {
			t.Fatal(err)
		}
		assertOpenclawModelExists(t, configPath, "llama3.2")
		assertOpenclawModelExists(t, configPath, "mistral")
		assertOpenclawPrimaryModel(t, configPath, "ollama/llama3.2")
	})

	t.Run("preserve other providers", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(configPath, []byte(`{"models":{"providers":{"anthropic":{"apiKey":"xxx"}}}}`), 0o644)
		if err := c.Edit([]string{"llama3.2"}); err != nil {
			t.Fatal(err)
		}
		data, _ := os.ReadFile(configPath)
		var cfg map[string]any
		json.Unmarshal(data, &cfg)
		models := cfg["models"].(map[string]any)
		providers := models["providers"].(map[string]any)
		if providers["anthropic"] == nil {
			t.Error("anthropic provider was removed")
		}
	})

	t.Run("preserve top-level keys", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(configPath, []byte(`{"theme":"dark","mcp":{"servers":{}}}`), 0o644)
		if err := c.Edit([]string{"llama3.2"}); err != nil {
			t.Fatal(err)
		}
		data, _ := os.ReadFile(configPath)
		var cfg map[string]any
		json.Unmarshal(data, &cfg)
		if cfg["theme"] != "dark" {
			t.Error("theme was removed")
		}
		if cfg["mcp"] == nil {
			t.Error("mcp was removed")
		}
	})

	t.Run("preserve user customizations on models", func(t *testing.T) {
		cleanup()
		c.Edit([]string{"llama3.2"})

		// User adds custom field
		data, _ := os.ReadFile(configPath)
		var cfg map[string]any
		json.Unmarshal(data, &cfg)
		models := cfg["models"].(map[string]any)
		providers := models["providers"].(map[string]any)
		ollama := providers["ollama"].(map[string]any)
		modelList := ollama["models"].([]any)
		entry := modelList[0].(map[string]any)
		entry["customField"] = "user-value"
		configData, _ := json.MarshalIndent(cfg, "", "  ")
		os.WriteFile(configPath, configData, 0o644)

		// Re-run Edit
		c.Edit([]string{"llama3.2"})

		data, _ = os.ReadFile(configPath)
		json.Unmarshal(data, &cfg)
		models = cfg["models"].(map[string]any)
		providers = models["providers"].(map[string]any)
		ollama = providers["ollama"].(map[string]any)
		modelList = ollama["models"].([]any)
		entry = modelList[0].(map[string]any)
		if entry["customField"] != "user-value" {
			t.Error("custom field was lost")
		}
	})

	t.Run("edit replaces models list", func(t *testing.T) {
		cleanup()
		c.Edit([]string{"llama3.2", "mistral"})
		c.Edit([]string{"llama3.2"})

		assertOpenclawModelExists(t, configPath, "llama3.2")
		assertOpenclawModelNotExists(t, configPath, "mistral")
	})

	t.Run("empty models is no-op", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)
		original := `{"existing":"data"}`
		os.WriteFile(configPath, []byte(original), 0o644)

		c.Edit([]string{})

		data, _ := os.ReadFile(configPath)
		if string(data) != original {
			t.Error("empty models should not modify file")
		}
	})

	t.Run("corrupted JSON treated as empty", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(configPath, []byte(`{corrupted`), 0o644)

		if err := c.Edit([]string{"llama3.2"}); err != nil {
			t.Fatal(err)
		}

		data, _ := os.ReadFile(configPath)
		var cfg map[string]any
		if err := json.Unmarshal(data, &cfg); err != nil {
			t.Error("result should be valid JSON")
		}
	})

	t.Run("wrong type models section", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(configPath, []byte(`{"models":"not a map"}`), 0o644)

		if err := c.Edit([]string{"llama3.2"}); err != nil {
			t.Fatal(err)
		}
		assertOpenclawModelExists(t, configPath, "llama3.2")
	})
}

func TestOpenclawModels(t *testing.T) {
	c := &Openclaw{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	t.Run("no config returns nil", func(t *testing.T) {
		if models := c.Models(); len(models) > 0 {
			t.Errorf("expected nil/empty, got %v", models)
		}
	})

	t.Run("returns all ollama models", func(t *testing.T) {
		configDir := filepath.Join(tmpDir, ".clawdbot")
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(filepath.Join(configDir, "clawdbot.json"), []byte(`{
			"models":{"providers":{"ollama":{"models":[
				{"id":"llama3.2"},
				{"id":"mistral"}
			]}}}
		}`), 0o644)

		models := c.Models()
		if len(models) != 2 {
			t.Errorf("expected 2 models, got %v", models)
		}
	})
}

// Helper functions
func assertOpenclawModelExists(t *testing.T, path, model string) {
	t.Helper()
	data, _ := os.ReadFile(path)
	var cfg map[string]any
	json.Unmarshal(data, &cfg)
	models := cfg["models"].(map[string]any)
	providers := models["providers"].(map[string]any)
	ollama := providers["ollama"].(map[string]any)
	modelList := ollama["models"].([]any)
	for _, m := range modelList {
		if entry, ok := m.(map[string]any); ok {
			if entry["id"] == model {
				return
			}
		}
	}
	t.Errorf("model %s not found", model)
}

func assertOpenclawModelNotExists(t *testing.T, path, model string) {
	t.Helper()
	data, _ := os.ReadFile(path)
	var cfg map[string]any
	json.Unmarshal(data, &cfg)
	models, _ := cfg["models"].(map[string]any)
	providers, _ := models["providers"].(map[string]any)
	ollama, _ := providers["ollama"].(map[string]any)
	modelList, _ := ollama["models"].([]any)
	for _, m := range modelList {
		if entry, ok := m.(map[string]any); ok {
			if entry["id"] == model {
				t.Errorf("model %s should not exist", model)
			}
		}
	}
}

func assertOpenclawPrimaryModel(t *testing.T, path, expected string) {
	t.Helper()
	data, _ := os.ReadFile(path)
	var cfg map[string]any
	json.Unmarshal(data, &cfg)
	agents := cfg["agents"].(map[string]any)
	defaults := agents["defaults"].(map[string]any)
	model := defaults["model"].(map[string]any)
	if model["primary"] != expected {
		t.Errorf("primary model = %v, want %v", model["primary"], expected)
	}
}

func TestOpenclawPaths(t *testing.T) {
	c := &Openclaw{}

	t.Run("returns path when config exists", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		configDir := filepath.Join(tmpDir, ".clawdbot")
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(filepath.Join(configDir, "clawdbot.json"), []byte(`{}`), 0o644)

		paths := c.Paths()
		if len(paths) != 1 {
			t.Errorf("expected 1 path, got %d", len(paths))
		}
	})

	t.Run("returns nil when config missing", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		if paths := c.Paths(); paths != nil {
			t.Errorf("expected nil, got %v", paths)
		}
	})
}

func TestOpenclawModelsEdgeCases(t *testing.T) {
	c := &Openclaw{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	configDir := filepath.Join(tmpDir, ".clawdbot")
	configPath := filepath.Join(configDir, "clawdbot.json")
	cleanup := func() { os.RemoveAll(configDir) }

	t.Run("corrupted JSON returns nil", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(configPath, []byte(`{corrupted`), 0o644)
		if models := c.Models(); models != nil {
			t.Errorf("expected nil, got %v", models)
		}
	})

	t.Run("wrong type at models level", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(configPath, []byte(`{"models":"string"}`), 0o644)
		if models := c.Models(); models != nil {
			t.Errorf("expected nil, got %v", models)
		}
	})

	t.Run("wrong type at providers level", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(configPath, []byte(`{"models":{"providers":"string"}}`), 0o644)
		if models := c.Models(); models != nil {
			t.Errorf("expected nil, got %v", models)
		}
	})

	t.Run("wrong type at ollama level", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(configPath, []byte(`{"models":{"providers":{"ollama":"string"}}}`), 0o644)
		if models := c.Models(); models != nil {
			t.Errorf("expected nil, got %v", models)
		}
	})

	t.Run("model entry missing id", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(configPath, []byte(`{"models":{"providers":{"ollama":{"models":[{"name":"test"}]}}}}`), 0o644)
		if len(c.Models()) != 0 {
			t.Error("expected empty for missing id")
		}
	})

	t.Run("model id is not string", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(configPath, []byte(`{"models":{"providers":{"ollama":{"models":[{"id":123}]}}}}`), 0o644)
		if len(c.Models()) != 0 {
			t.Error("expected empty for non-string id")
		}
	})
}

func TestOpenclawEditSchemaFields(t *testing.T) {
	c := &Openclaw{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	configPath := filepath.Join(tmpDir, ".clawdbot", "clawdbot.json")

	if err := c.Edit([]string{"llama3.2"}); err != nil {
		t.Fatal(err)
	}

	data, _ := os.ReadFile(configPath)
	var cfg map[string]any
	json.Unmarshal(data, &cfg)
	models := cfg["models"].(map[string]any)
	providers := models["providers"].(map[string]any)
	ollama := providers["ollama"].(map[string]any)
	modelList := ollama["models"].([]any)
	entry := modelList[0].(map[string]any)

	// Verify required schema fields
	if entry["reasoning"] != false {
		t.Error("reasoning should be false")
	}
	if entry["input"] == nil {
		t.Error("input should be set")
	}
	if entry["contextWindow"] == nil {
		t.Error("contextWindow should be set")
	}
	if entry["maxTokens"] == nil {
		t.Error("maxTokens should be set")
	}
	cost := entry["cost"].(map[string]any)
	if cost["cacheRead"] == nil {
		t.Error("cost.cacheRead should be set")
	}
	if cost["cacheWrite"] == nil {
		t.Error("cost.cacheWrite should be set")
	}
}

func TestOpenclawEditModelNames(t *testing.T) {
	c := &Openclaw{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	configPath := filepath.Join(tmpDir, ".clawdbot", "clawdbot.json")
	cleanup := func() { os.RemoveAll(filepath.Join(tmpDir, ".clawdbot")) }

	t.Run("model with colon tag", func(t *testing.T) {
		cleanup()
		if err := c.Edit([]string{"llama3.2:70b"}); err != nil {
			t.Fatal(err)
		}
		assertOpenclawModelExists(t, configPath, "llama3.2:70b")
		assertOpenclawPrimaryModel(t, configPath, "ollama/llama3.2:70b")
	})

	t.Run("model with slash", func(t *testing.T) {
		cleanup()
		if err := c.Edit([]string{"library/model:tag"}); err != nil {
			t.Fatal(err)
		}
		assertOpenclawModelExists(t, configPath, "library/model:tag")
		assertOpenclawPrimaryModel(t, configPath, "ollama/library/model:tag")
	})

	t.Run("model with hyphen", func(t *testing.T) {
		cleanup()
		if err := c.Edit([]string{"test-model"}); err != nil {
			t.Fatal(err)
		}
		assertOpenclawModelExists(t, configPath, "test-model")
	})
}

func TestOpenclawEditAgentsPreservation(t *testing.T) {
	c := &Openclaw{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	configDir := filepath.Join(tmpDir, ".clawdbot")
	configPath := filepath.Join(configDir, "clawdbot.json")
	cleanup := func() { os.RemoveAll(configDir) }

	t.Run("preserve other agent defaults", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(configPath, []byte(`{"agents":{"defaults":{"model":{"primary":"old"},"temperature":0.7}}}`), 0o644)

		c.Edit([]string{"llama3.2"})

		data, _ := os.ReadFile(configPath)
		var cfg map[string]any
		json.Unmarshal(data, &cfg)
		agents := cfg["agents"].(map[string]any)
		defaults := agents["defaults"].(map[string]any)
		if defaults["temperature"] != 0.7 {
			t.Error("temperature setting was lost")
		}
	})

	t.Run("preserve other agents besides defaults", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)
		os.WriteFile(configPath, []byte(`{"agents":{"defaults":{},"custom-agent":{"foo":"bar"}}}`), 0o644)

		c.Edit([]string{"llama3.2"})

		data, _ := os.ReadFile(configPath)
		var cfg map[string]any
		json.Unmarshal(data, &cfg)
		agents := cfg["agents"].(map[string]any)
		if agents["custom-agent"] == nil {
			t.Error("custom-agent was lost")
		}
	})
}

const testOpenclawFixture = `{
  "theme": "dark",
  "mcp": {"servers": {"custom": {"enabled": true}}},
  "models": {
    "providers": {
      "anthropic": {"apiKey": "xxx"},
      "ollama": {
        "baseUrl": "http://127.0.0.1:11434/v1",
        "models": [{"id": "old-model", "customField": "preserved"}]
      }
    }
  },
  "agents": {
    "defaults": {"model": {"primary": "old"}, "temperature": 0.7},
    "custom-agent": {"foo": "bar"}
  }
}`

func TestOpenclawEdit_RoundTrip(t *testing.T) {
	c := &Openclaw{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	configDir := filepath.Join(tmpDir, ".clawdbot")
	configPath := filepath.Join(configDir, "clawdbot.json")

	os.MkdirAll(configDir, 0o755)
	os.WriteFile(configPath, []byte(testOpenclawFixture), 0o644)

	if err := c.Edit([]string{"llama3.2", "mistral"}); err != nil {
		t.Fatal(err)
	}

	data, _ := os.ReadFile(configPath)
	var cfg map[string]any
	json.Unmarshal(data, &cfg)

	// Verify top-level preserved
	if cfg["theme"] != "dark" {
		t.Error("theme not preserved")
	}
	mcp := cfg["mcp"].(map[string]any)
	servers := mcp["servers"].(map[string]any)
	if servers["custom"] == nil {
		t.Error("mcp.servers.custom not preserved")
	}

	// Verify other providers preserved
	models := cfg["models"].(map[string]any)
	providers := models["providers"].(map[string]any)
	if providers["anthropic"] == nil {
		t.Error("anthropic provider not preserved")
	}

	// Verify agents preserved
	agents := cfg["agents"].(map[string]any)
	if agents["custom-agent"] == nil {
		t.Error("custom-agent not preserved")
	}
	defaults := agents["defaults"].(map[string]any)
	if defaults["temperature"] != 0.7 {
		t.Error("temperature not preserved")
	}
}

func TestOpenclawEdit_Idempotent(t *testing.T) {
	c := &Openclaw{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	configDir := filepath.Join(tmpDir, ".clawdbot")
	configPath := filepath.Join(configDir, "clawdbot.json")

	os.MkdirAll(configDir, 0o755)
	os.WriteFile(configPath, []byte(testOpenclawFixture), 0o644)

	c.Edit([]string{"llama3.2", "mistral"})
	firstData, _ := os.ReadFile(configPath)

	c.Edit([]string{"llama3.2", "mistral"})
	secondData, _ := os.ReadFile(configPath)

	if string(firstData) != string(secondData) {
		t.Error("repeated edits with same models produced different results")
	}
}

func TestOpenclawEdit_MultipleConsecutiveEdits(t *testing.T) {
	c := &Openclaw{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	configDir := filepath.Join(tmpDir, ".clawdbot")
	configPath := filepath.Join(configDir, "clawdbot.json")

	os.MkdirAll(configDir, 0o755)
	os.WriteFile(configPath, []byte(testOpenclawFixture), 0o644)

	for i := range 10 {
		models := []string{"model-a", "model-b"}
		if i%2 == 0 {
			models = []string{"model-x", "model-y", "model-z"}
		}
		if err := c.Edit(models); err != nil {
			t.Fatalf("edit %d failed: %v", i, err)
		}
	}

	data, _ := os.ReadFile(configPath)
	var cfg map[string]any
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("file is not valid JSON after multiple edits: %v", err)
	}

	if cfg["theme"] != "dark" {
		t.Error("theme lost after multiple edits")
	}
}

func TestOpenclawEdit_BackupCreated(t *testing.T) {
	c := &Openclaw{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	configDir := filepath.Join(tmpDir, ".clawdbot")
	configPath := filepath.Join(configDir, "clawdbot.json")
	backupDir := filepath.Join(os.TempDir(), "ollama-backups")

	os.MkdirAll(configDir, 0o755)
	uniqueMarker := fmt.Sprintf("test-marker-%d", os.Getpid())
	original := fmt.Sprintf(`{"theme": "%s"}`, uniqueMarker)
	os.WriteFile(configPath, []byte(original), 0o644)

	if err := c.Edit([]string{"model-a"}); err != nil {
		t.Fatal(err)
	}

	backups, _ := filepath.Glob(filepath.Join(backupDir, "clawdbot.json.*"))
	foundBackup := false
	for _, backup := range backups {
		data, _ := os.ReadFile(backup)
		if string(data) == original {
			foundBackup = true
			break
		}
	}

	if !foundBackup {
		t.Error("backup with original content not found")
	}
}

func TestOpenclawEdit_CreatesDirectoryIfMissing(t *testing.T) {
	c := &Openclaw{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	configDir := filepath.Join(tmpDir, ".clawdbot")

	if _, err := os.Stat(configDir); !os.IsNotExist(err) {
		t.Fatal("directory should not exist before test")
	}

	if err := c.Edit([]string{"model-a"}); err != nil {
		t.Fatal(err)
	}

	if _, err := os.Stat(configDir); os.IsNotExist(err) {
		t.Fatal("directory was not created")
	}
}
