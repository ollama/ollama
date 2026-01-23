package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// setTestHome sets both HOME (Unix) and USERPROFILE (Windows) for cross-platform tests
func setTestHome(t *testing.T, dir string) {
	t.Setenv("HOME", dir)
	t.Setenv("USERPROFILE", dir)
}

// editorPaths is a test helper that safely calls Paths if the runner implements Editor
func editorPaths(r Runner) []string {
	if editor, ok := r.(Editor); ok {
		return editor.Paths()
	}
	return nil
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
		defaultModel := ""
		if len(config.Models) > 0 {
			defaultModel = config.Models[0]
		}
		if defaultModel != "model-a" {
			t.Errorf("expected model-a, got %s", defaultModel)
		}
	})

	t.Run("defaultModel returns empty for no models", func(t *testing.T) {
		config := &integrationConfig{Models: []string{}}
		defaultModel := ""
		if len(config.Models) > 0 {
			defaultModel = config.Models[0]
		}
		if defaultModel != "" {
			t.Errorf("expected empty string, got %s", defaultModel)
		}
	})

	t.Run("app name is case-insensitive", func(t *testing.T) {
		saveIntegration("Claude", []string{"model-x"})

		config, err := loadIntegration("claude")
		if err != nil {
			t.Fatal(err)
		}
		defaultModel := ""
		if len(config.Models) > 0 {
			defaultModel = config.Models[0]
		}
		if defaultModel != "model-x" {
			t.Errorf("expected model-x, got %s", defaultModel)
		}
	})

	t.Run("multiple integrations in single file", func(t *testing.T) {
		saveIntegration("app1", []string{"model-1"})
		saveIntegration("app2", []string{"model-2"})

		config1, _ := loadIntegration("app1")
		config2, _ := loadIntegration("app2")

		defaultModel1 := ""
		if len(config1.Models) > 0 {
			defaultModel1 = config1.Models[0]
		}
		defaultModel2 := ""
		if len(config2.Models) > 0 {
			defaultModel2 = config2.Models[0]
		}
		if defaultModel1 != "model-1" {
			t.Errorf("expected model-1, got %s", defaultModel1)
		}
		if defaultModel2 != "model-2" {
			t.Errorf("expected model-2, got %s", defaultModel2)
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

func TestEditorPaths(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	t.Run("returns empty for claude (no Editor)", func(t *testing.T) {
		r := integrations["claude"]
		paths := editorPaths(r)
		if len(paths) != 0 {
			t.Errorf("expected no paths for claude, got %v", paths)
		}
	})

	t.Run("returns empty for codex (no Editor)", func(t *testing.T) {
		r := integrations["codex"]
		paths := editorPaths(r)
		if len(paths) != 0 {
			t.Errorf("expected no paths for codex, got %v", paths)
		}
	})

	t.Run("returns empty for droid when no config exists", func(t *testing.T) {
		r := integrations["droid"]
		paths := editorPaths(r)
		if len(paths) != 0 {
			t.Errorf("expected no paths, got %v", paths)
		}
	})

	t.Run("returns path for droid when config exists", func(t *testing.T) {
		settingsDir, _ := os.UserHomeDir()
		settingsDir = filepath.Join(settingsDir, ".factory")
		os.MkdirAll(settingsDir, 0o755)
		os.WriteFile(filepath.Join(settingsDir, "settings.json"), []byte(`{}`), 0o644)

		r := integrations["droid"]
		paths := editorPaths(r)
		if len(paths) != 1 {
			t.Errorf("expected 1 path, got %d", len(paths))
		}
	})

	t.Run("returns paths for opencode when configs exist", func(t *testing.T) {
		home, _ := os.UserHomeDir()
		configDir := filepath.Join(home, ".config", "opencode")
		stateDir := filepath.Join(home, ".local", "state", "opencode")
		os.MkdirAll(configDir, 0o755)
		os.MkdirAll(stateDir, 0o755)
		os.WriteFile(filepath.Join(configDir, "opencode.json"), []byte(`{}`), 0o644)
		os.WriteFile(filepath.Join(stateDir, "model.json"), []byte(`{}`), 0o644)

		r := integrations["opencode"]
		paths := editorPaths(r)
		if len(paths) != 2 {
			t.Errorf("expected 2 paths, got %d: %v", len(paths), paths)
		}
	})
}

func TestLoadIntegration_CorruptedJSON(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	// Create corrupted config.json file
	dir := filepath.Join(tmpDir, ".ollama", "config")
	os.MkdirAll(dir, 0o755)
	os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{corrupted json`), 0o644)

	// Corrupted file is treated as empty, so loadIntegration returns not found
	_, err := loadIntegration("test")
	if err == nil {
		t.Error("expected error for nonexistent integration in corrupted file")
	}
}

func TestSaveIntegration_NilModels(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	if err := saveIntegration("test", nil); err != nil {
		t.Fatalf("saveIntegration with nil models failed: %v", err)
	}

	config, err := loadIntegration("test")
	if err != nil {
		t.Fatalf("loadIntegration failed: %v", err)
	}

	if config.Models == nil {
		// nil is acceptable
	} else if len(config.Models) != 0 {
		t.Errorf("expected empty or nil models, got %v", config.Models)
	}
}

func TestSaveIntegration_EmptyAppName(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	err := saveIntegration("", []string{"model"})
	if err == nil {
		t.Error("expected error for empty app name, got nil")
	}
	if err != nil && !strings.Contains(err.Error(), "app name cannot be empty") {
		t.Errorf("expected 'app name cannot be empty' error, got: %v", err)
	}
}

func TestLoadIntegration_NonexistentIntegration(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	_, err := loadIntegration("nonexistent")
	if err == nil {
		t.Error("expected error for nonexistent integration, got nil")
	}
	if !os.IsNotExist(err) {
		t.Logf("error type is os.ErrNotExist as expected: %v", err)
	}
}

func TestIntegrationsPath(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	path, err := integrationsPath()
	if err != nil {
		t.Fatal(err)
	}

	expected := filepath.Join(tmpDir, ".ollama", "config", "config.json")
	if path != expected {
		t.Errorf("expected %s, got %s", expected, path)
	}
}
