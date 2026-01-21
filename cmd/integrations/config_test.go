package integrations

import (
	"os"
	"testing"
)

// setTestHome sets both HOME (Unix) and USERPROFILE (Windows) for cross-platform tests
func setTestHome(t *testing.T, dir string) {
	t.Setenv("HOME", dir)
	t.Setenv("USERPROFILE", dir)
}

func TestIntegrationConfig(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	t.Run("save and load round-trip", func(t *testing.T) {
		models := []string{"llama3.2", "mistral", "qwen2.5"}
		if err := saveIntegration("claude", models); err != nil {
			t.Fatal(err)
		}

		config, err := loadIntegration("claude")
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

	t.Run("defaultModel returns first model", func(t *testing.T) {
		saveIntegration("codex", []string{"model-a", "model-b"})

		config, _ := loadIntegration("codex")
		if config.defaultModel() != "model-a" {
			t.Errorf("expected model-a, got %s", config.defaultModel())
		}
	})

	t.Run("defaultModel returns empty for no models", func(t *testing.T) {
		config := &integrationConfig{Models: []string{}}
		if config.defaultModel() != "" {
			t.Errorf("expected empty string, got %s", config.defaultModel())
		}
	})

	t.Run("app name is case-insensitive", func(t *testing.T) {
		saveIntegration("Claude", []string{"model-x"})

		config, err := loadIntegration("claude")
		if err != nil {
			t.Fatal(err)
		}
		if config.defaultModel() != "model-x" {
			t.Errorf("expected model-x, got %s", config.defaultModel())
		}
	})
}

func TestListIntegrations(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	t.Run("returns empty when no integrations", func(t *testing.T) {
		configs, err := listIntegrations()
		if err != nil {
			t.Fatal(err)
		}
		if len(configs) != 0 {
			t.Errorf("expected 0 integrations, got %d", len(configs))
		}
	})

	t.Run("returns all saved integrations", func(t *testing.T) {
		saveIntegration("claude", []string{"model-1"})
		saveIntegration("droid", []string{"model-2"})

		configs, err := listIntegrations()
		if err != nil {
			t.Fatal(err)
		}
		if len(configs) != 2 {
			t.Errorf("expected 2 integrations, got %d", len(configs))
		}
	})
}

func TestGetExistingConfigPaths(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

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
		settingsDir, _ := os.UserHomeDir()
		settingsDir += "/.factory"
		os.MkdirAll(settingsDir, 0o755)
		os.WriteFile(settingsDir+"/settings.json", []byte(`{}`), 0o644)

		paths := getExistingConfigPaths("droid")
		if len(paths) != 1 {
			t.Errorf("expected 1 path, got %d", len(paths))
		}
	})

	t.Run("returns paths for opencode when configs exist", func(t *testing.T) {
		home, _ := os.UserHomeDir()
		configDir := home + "/.config/opencode"
		stateDir := home + "/.local/state/opencode"
		os.MkdirAll(configDir, 0o755)
		os.MkdirAll(stateDir, 0o755)
		os.WriteFile(configDir+"/opencode.json", []byte(`{}`), 0o644)
		os.WriteFile(stateDir+"/model.json", []byte(`{}`), 0o644)

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
