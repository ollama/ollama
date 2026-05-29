package launch

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
	providersPath := filepath.Join(tmpDir, ".cline", "data", "settings", "providers.json")

	readConfig := func() map[string]any {
		data, _ := os.ReadFile(configPath)
		var config map[string]any
		json.Unmarshal(data, &config)
		return config
	}

	readProvidersConfig := func() map[string]any {
		data, _ := os.ReadFile(providersPath)
		var config map[string]any
		json.Unmarshal(data, &config)
		return config
	}

	t.Run("creates config from scratch", func(t *testing.T) {
		os.RemoveAll(filepath.Join(tmpDir, ".cline"))

		if err := c.Edit(testLaunchModels("kimi-k2.5:cloud")); err != nil {
			t.Fatal(err)
		}

		config := readConfig()
		if config["actModeApiProvider"] != clineLaunchProvider {
			t.Errorf("actModeApiProvider = %v, want %s", config["actModeApiProvider"], clineLaunchProvider)
		}
		if config["actModeOllamaModelId"] != "kimi-k2.5:cloud" {
			t.Errorf("actModeOllamaModelId = %v, want kimi-k2.5:cloud", config["actModeOllamaModelId"])
		}
		if config["actModeOllamaBaseUrl"] != "http://127.0.0.1:11434" {
			t.Errorf("actModeOllamaBaseUrl = %v, want http://127.0.0.1:11434", config["actModeOllamaBaseUrl"])
		}
		if config["planModeApiProvider"] != clineLaunchProvider {
			t.Errorf("planModeApiProvider = %v, want %s", config["planModeApiProvider"], clineLaunchProvider)
		}
		if config["planModeOllamaModelId"] != "kimi-k2.5:cloud" {
			t.Errorf("planModeOllamaModelId = %v, want kimi-k2.5:cloud", config["planModeOllamaModelId"])
		}
		if config["planModeOllamaBaseUrl"] != "http://127.0.0.1:11434" {
			t.Errorf("planModeOllamaBaseUrl = %v, want http://127.0.0.1:11434", config["planModeOllamaBaseUrl"])
		}
		if config["ollamaBaseUrl"] != "http://127.0.0.1:11434" {
			t.Errorf("ollamaBaseUrl = %v, want http://127.0.0.1:11434", config["ollamaBaseUrl"])
		}
		if config["welcomeViewCompleted"] != true {
			t.Errorf("welcomeViewCompleted = %v, want true", config["welcomeViewCompleted"])
		}

		providersConfig := readProvidersConfig()
		if providersConfig["lastUsedProvider"] != clineLaunchProvider {
			t.Errorf("lastUsedProvider = %v, want %s", providersConfig["lastUsedProvider"], clineLaunchProvider)
		}
		providers, _ := providersConfig["providers"].(map[string]any)
		provider, _ := providers[clineLaunchProvider].(map[string]any)
		if provider["updatedAt"] == "" {
			t.Errorf("updatedAt = %v, want timestamp", provider["updatedAt"])
		}
		settings, _ := provider["settings"].(map[string]any)
		if settings["model"] != "kimi-k2.5:cloud" {
			t.Errorf("settings.model = %v, want kimi-k2.5:cloud", settings["model"])
		}
		if _, ok := settings["apiKey"]; ok {
			t.Errorf("settings.apiKey = %v, want omitted for local Ollama", settings["apiKey"])
		}
		if settings["baseUrl"] != "http://127.0.0.1:11434/v1" {
			t.Errorf("settings.baseUrl = %v, want http://127.0.0.1:11434/v1", settings["baseUrl"])
		}
	})

	t.Run("preserves existing fields", func(t *testing.T) {
		os.RemoveAll(filepath.Join(tmpDir, ".cline"))
		os.MkdirAll(configDir, 0o755)
		os.MkdirAll(filepath.Dir(providersPath), 0o755)

		existing := map[string]any{
			"remoteRulesToggles":    map[string]any{},
			"remoteWorkflowToggles": map[string]any{},
			"customSetting":         "keep-me",
		}
		data, _ := json.Marshal(existing)
		os.WriteFile(configPath, data, 0o644)

		existingProviders := map[string]any{
			"customRoot": "keep-me-too",
			"providers": map[string]any{
				clineLaunchProvider: map[string]any{
					"updatedAt": "2026-05-29T16:56:46.111Z",
					"settings": map[string]any{
						"apiKey":  "bad-migrated-key",
						"timeout": float64(30000),
					},
				},
			},
		}
		data, _ = json.Marshal(existingProviders)
		os.WriteFile(providersPath, data, 0o644)

		if err := c.Edit(testLaunchModels("glm-5:cloud")); err != nil {
			t.Fatal(err)
		}

		config := readConfig()
		if config["customSetting"] != "keep-me" {
			t.Errorf("customSetting was not preserved")
		}
		if config["actModeOllamaModelId"] != "glm-5:cloud" {
			t.Errorf("actModeOllamaModelId = %v, want glm-5:cloud", config["actModeOllamaModelId"])
		}

		providersConfig := readProvidersConfig()
		if providersConfig["customRoot"] != "keep-me-too" {
			t.Errorf("customRoot was not preserved")
		}
		providers, _ := providersConfig["providers"].(map[string]any)
		provider, _ := providers[clineLaunchProvider].(map[string]any)
		if provider["updatedAt"] != "2026-05-29T16:56:46.111Z" {
			t.Errorf("updatedAt = %v, want preserved timestamp", provider["updatedAt"])
		}
		settings, _ := provider["settings"].(map[string]any)
		if settings["timeout"] != float64(30000) {
			t.Errorf("settings.timeout = %v, want 30000", settings["timeout"])
		}
		if _, ok := settings["apiKey"]; ok {
			t.Errorf("settings.apiKey = %v, want omitted for local Ollama", settings["apiKey"])
		}
		if settings["model"] != "glm-5:cloud" {
			t.Errorf("settings.model = %v, want glm-5:cloud", settings["model"])
		}
		if settings["baseUrl"] != "http://127.0.0.1:11434/v1" {
			t.Errorf("settings.baseUrl = %v, want http://127.0.0.1:11434/v1", settings["baseUrl"])
		}
	})

	t.Run("updates model on re-edit", func(t *testing.T) {
		os.RemoveAll(filepath.Join(tmpDir, ".cline"))

		if err := c.Edit(testLaunchModels("kimi-k2.5:cloud")); err != nil {
			t.Fatal(err)
		}
		if err := c.Edit(testLaunchModels("glm-5:cloud")); err != nil {
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

		if err := c.Edit(testLaunchModels("kimi-k2.5:cloud", "glm-5:cloud")); err != nil {
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
	providersPath := filepath.Join(tmpDir, ".cline", "data", "settings", "providers.json")

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

	t.Run("prefers CLI provider config", func(t *testing.T) {
		os.MkdirAll(filepath.Dir(providersPath), 0o755)
		config := map[string]any{
			"lastUsedProvider": clineLaunchProvider,
			"providers": map[string]any{
				clineLaunchProvider: map[string]any{
					"settings": map[string]any{
						"model": "glm-5:cloud",
					},
				},
			},
		}
		data, _ := json.Marshal(config)
		os.WriteFile(providersPath, data, 0o644)

		models := c.Models()
		if len(models) != 1 || models[0] != "glm-5:cloud" {
			t.Errorf("Models() = %v, want [glm-5:cloud]", models)
		}
	})

	t.Run("ignores stale CLI provider config when ollama is not active", func(t *testing.T) {
		os.RemoveAll(filepath.Join(tmpDir, ".cline"))
		os.MkdirAll(configDir, 0o755)
		os.MkdirAll(filepath.Dir(providersPath), 0o755)
		legacyConfig := map[string]any{
			"actModeApiProvider":   "anthropic",
			"actModeOllamaModelId": "legacy-ollama-model",
		}
		data, _ := json.Marshal(legacyConfig)
		os.WriteFile(configPath, data, 0o644)
		providerConfig := map[string]any{
			"lastUsedProvider": "openai",
			"providers": map[string]any{
				clineLaunchProvider: map[string]any{
					"settings": map[string]any{
						"model": "stale-ollama-model",
					},
				},
			},
		}
		data, _ = json.Marshal(providerConfig)
		os.WriteFile(providersPath, data, 0o644)

		if models := c.Models(); models != nil {
			t.Errorf("Models() = %v, want nil", models)
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

func TestClineLaunchArgs(t *testing.T) {
	got := clineLaunchArgs("kimi-k2.5:cloud", []string{"--json", "hello"})
	want := []string{"--json", "hello"}
	if len(got) != len(want) {
		t.Fatalf("args length = %d, want %d: %v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("args[%d] = %q, want %q; got %v", i, got[i], want[i], got)
		}
	}
}
