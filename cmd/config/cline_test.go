package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestClineIntegration(t *testing.T) {
	c := &Cline{}

	t.Run("String", func(t *testing.T) {
		if got := c.String(); got != "Cline" {
			t.Errorf("String() = %q, want %q", got, "Cline")
		}
	})

	t.Run("implements Runner", func(t *testing.T) {
		var _ Runner = c
	})

	t.Run("implements Editor", func(t *testing.T) {
		var _ Editor = c
	})
}

func TestClineEdit(t *testing.T) {
	c := &Cline{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	configDir := filepath.Join(tmpDir, ".cline", "data")
	configPath := filepath.Join(configDir, "globalState.json")

	readConfig := func() map[string]any {
		data, _ := os.ReadFile(configPath)
		var config map[string]any
		json.Unmarshal(data, &config)
		return config
	}

	t.Run("creates config from scratch", func(t *testing.T) {
		os.RemoveAll(filepath.Join(tmpDir, ".cline"))

		if err := c.Edit([]string{"kimi-k2.5:cloud"}); err != nil {
			t.Fatal(err)
		}

		config := readConfig()
		if config["actModeApiProvider"] != "ollama" {
			t.Errorf("actModeApiProvider = %v, want ollama", config["actModeApiProvider"])
		}
		if config["actModeOllamaModelId"] != "kimi-k2.5:cloud" {
			t.Errorf("actModeOllamaModelId = %v, want kimi-k2.5:cloud", config["actModeOllamaModelId"])
		}
		if config["planModeApiProvider"] != "ollama" {
			t.Errorf("planModeApiProvider = %v, want ollama", config["planModeApiProvider"])
		}
		if config["planModeOllamaModelId"] != "kimi-k2.5:cloud" {
			t.Errorf("planModeOllamaModelId = %v, want kimi-k2.5:cloud", config["planModeOllamaModelId"])
		}
		if config["welcomeViewCompleted"] != true {
			t.Errorf("welcomeViewCompleted = %v, want true", config["welcomeViewCompleted"])
		}
	})

	t.Run("preserves existing fields", func(t *testing.T) {
		os.RemoveAll(filepath.Join(tmpDir, ".cline"))
		os.MkdirAll(configDir, 0o755)

		existing := map[string]any{
			"remoteRulesToggles":    map[string]any{},
			"remoteWorkflowToggles": map[string]any{},
			"customSetting":         "keep-me",
		}
		data, _ := json.Marshal(existing)
		os.WriteFile(configPath, data, 0o644)

		if err := c.Edit([]string{"glm-5:cloud"}); err != nil {
			t.Fatal(err)
		}

		config := readConfig()
		if config["customSetting"] != "keep-me" {
			t.Errorf("customSetting was not preserved")
		}
		if config["actModeOllamaModelId"] != "glm-5:cloud" {
			t.Errorf("actModeOllamaModelId = %v, want glm-5:cloud", config["actModeOllamaModelId"])
		}
	})

	t.Run("updates model on re-edit", func(t *testing.T) {
		os.RemoveAll(filepath.Join(tmpDir, ".cline"))

		if err := c.Edit([]string{"kimi-k2.5:cloud"}); err != nil {
			t.Fatal(err)
		}
		if err := c.Edit([]string{"glm-5:cloud"}); err != nil {
			t.Fatal(err)
		}

		config := readConfig()
		if config["actModeOllamaModelId"] != "glm-5:cloud" {
			t.Errorf("actModeOllamaModelId = %v, want glm-5:cloud", config["actModeOllamaModelId"])
		}
		if config["planModeOllamaModelId"] != "glm-5:cloud" {
			t.Errorf("planModeOllamaModelId = %v, want glm-5:cloud", config["planModeOllamaModelId"])
		}
	})

	t.Run("empty models is no-op", func(t *testing.T) {
		os.RemoveAll(filepath.Join(tmpDir, ".cline"))

		if err := c.Edit(nil); err != nil {
			t.Fatal(err)
		}

		if _, err := os.Stat(configPath); !os.IsNotExist(err) {
			t.Error("expected no config file to be created for empty models")
		}
	})

	t.Run("uses first model as primary", func(t *testing.T) {
		os.RemoveAll(filepath.Join(tmpDir, ".cline"))

		if err := c.Edit([]string{"kimi-k2.5:cloud", "glm-5:cloud"}); err != nil {
			t.Fatal(err)
		}

		config := readConfig()
		if config["actModeOllamaModelId"] != "kimi-k2.5:cloud" {
			t.Errorf("actModeOllamaModelId = %v, want kimi-k2.5:cloud (first model)", config["actModeOllamaModelId"])
		}
	})
}

func TestClineModels(t *testing.T) {
	c := &Cline{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	configDir := filepath.Join(tmpDir, ".cline", "data")
	configPath := filepath.Join(configDir, "globalState.json")

	t.Run("returns nil when no config", func(t *testing.T) {
		if models := c.Models(); models != nil {
			t.Errorf("Models() = %v, want nil", models)
		}
	})

	t.Run("returns nil when provider is not ollama", func(t *testing.T) {
		os.MkdirAll(configDir, 0o755)
		config := map[string]any{
			"actModeApiProvider":   "anthropic",
			"actModeOllamaModelId": "some-model",
		}
		data, _ := json.Marshal(config)
		os.WriteFile(configPath, data, 0o644)

		if models := c.Models(); models != nil {
			t.Errorf("Models() = %v, want nil", models)
		}
	})

	t.Run("returns model when ollama is configured", func(t *testing.T) {
		os.MkdirAll(configDir, 0o755)
		config := map[string]any{
			"actModeApiProvider":   "ollama",
			"actModeOllamaModelId": "kimi-k2.5:cloud",
		}
		data, _ := json.Marshal(config)
		os.WriteFile(configPath, data, 0o644)

		models := c.Models()
		if len(models) != 1 || models[0] != "kimi-k2.5:cloud" {
			t.Errorf("Models() = %v, want [kimi-k2.5:cloud]", models)
		}
	})
}

func TestClinePaths(t *testing.T) {
	c := &Cline{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	t.Run("returns nil when no config exists", func(t *testing.T) {
		if paths := c.Paths(); paths != nil {
			t.Errorf("Paths() = %v, want nil", paths)
		}
	})

	t.Run("returns path when config exists", func(t *testing.T) {
		configDir := filepath.Join(tmpDir, ".cline", "data")
		os.MkdirAll(configDir, 0o755)
		configPath := filepath.Join(configDir, "globalState.json")
		os.WriteFile(configPath, []byte("{}"), 0o644)

		paths := c.Paths()
		if len(paths) != 1 || paths[0] != configPath {
			t.Errorf("Paths() = %v, want [%s]", paths, configPath)
		}
	})
}
