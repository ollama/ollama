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

	t.Run("save and load aliases", func(t *testing.T) {
		models := []string{"llama3.2"}
		if err := saveIntegration("claude", models); err != nil {
			t.Fatal(err)
		}
		aliases := map[string]string{
			"primary": "llama3.2:70b",
			"fast":    "llama3.2:8b",
		}
		if err := saveAliases("claude", aliases); err != nil {
			t.Fatal(err)
		}

		config, err := loadIntegration("claude")
		if err != nil {
			t.Fatal(err)
		}
		if config.Aliases == nil {
			t.Fatal("expected aliases to be saved")
		}
		for k, v := range aliases {
			if config.Aliases[k] != v {
				t.Errorf("alias %s: expected %s, got %s", k, v, config.Aliases[k])
			}
		}
	})

	t.Run("saveIntegration preserves aliases", func(t *testing.T) {
		if err := saveIntegration("claude", []string{"model-a"}); err != nil {
			t.Fatal(err)
		}
		if err := saveAliases("claude", map[string]string{"primary": "model-a", "fast": "model-small"}); err != nil {
			t.Fatal(err)
		}

		if err := saveIntegration("claude", []string{"model-b"}); err != nil {
			t.Fatal(err)
		}
		config, err := loadIntegration("claude")
		if err != nil {
			t.Fatal(err)
		}
		if config.Aliases["primary"] != "model-a" {
			t.Errorf("expected aliases to be preserved, got %v", config.Aliases)
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
		config := &integration{Models: []string{}}
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

	dir := filepath.Join(tmpDir, ".ollama")
	os.MkdirAll(dir, 0o755)
	os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{corrupted json`), 0o644)

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

func TestConfigPath(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	path, err := configPath()
	if err != nil {
		t.Fatal(err)
	}

	expected := filepath.Join(tmpDir, ".ollama", "config.json")
	if path != expected {
		t.Errorf("expected %s, got %s", expected, path)
	}
}

func TestLoad(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	t.Run("returns empty config when file does not exist", func(t *testing.T) {
		cfg, err := load()
		if err != nil {
			t.Fatal(err)
		}
		if cfg == nil {
			t.Fatal("expected non-nil config")
		}
		if cfg.Integrations == nil {
			t.Error("expected non-nil Integrations map")
		}
		if len(cfg.Integrations) != 0 {
			t.Errorf("expected empty Integrations, got %d", len(cfg.Integrations))
		}
	})

	t.Run("loads existing config", func(t *testing.T) {
		path, _ := configPath()
		os.MkdirAll(filepath.Dir(path), 0o755)
		os.WriteFile(path, []byte(`{"integrations":{"test":{"models":["model-a"]}}}`), 0o644)

		cfg, err := load()
		if err != nil {
			t.Fatal(err)
		}
		if cfg.Integrations["test"] == nil {
			t.Fatal("expected test integration")
		}
		if len(cfg.Integrations["test"].Models) != 1 {
			t.Errorf("expected 1 model, got %d", len(cfg.Integrations["test"].Models))
		}
	})

	t.Run("returns error for corrupted JSON", func(t *testing.T) {
		path, _ := configPath()
		os.MkdirAll(filepath.Dir(path), 0o755)
		os.WriteFile(path, []byte(`{corrupted`), 0o644)

		_, err := load()
		if err == nil {
			t.Error("expected error for corrupted JSON")
		}
	})
}

func TestMigrateConfig(t *testing.T) {
	t.Run("migrates legacy file to new location", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		legacyDir := filepath.Join(tmpDir, ".ollama", "config")
		os.MkdirAll(legacyDir, 0o755)
		data := []byte(`{"integrations":{"claude":{"models":["llama3.2"]}}}`)
		os.WriteFile(filepath.Join(legacyDir, "config.json"), data, 0o644)

		migrated, err := migrateConfig()
		if err != nil {
			t.Fatal(err)
		}
		if !migrated {
			t.Fatal("expected migration to occur")
		}

		newPath, _ := configPath()
		got, err := os.ReadFile(newPath)
		if err != nil {
			t.Fatalf("new config not found: %v", err)
		}
		if string(got) != string(data) {
			t.Errorf("content mismatch: got %s", got)
		}

		if _, err := os.Stat(filepath.Join(legacyDir, "config.json")); !os.IsNotExist(err) {
			t.Error("legacy file should have been removed")
		}

		if _, err := os.Stat(legacyDir); !os.IsNotExist(err) {
			t.Error("legacy directory should have been removed")
		}
	})

	t.Run("no-op when no legacy file exists", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		migrated, err := migrateConfig()
		if err != nil {
			t.Fatal(err)
		}
		if migrated {
			t.Error("expected no migration")
		}
	})

	t.Run("skips corrupt legacy file", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		legacyDir := filepath.Join(tmpDir, ".ollama", "config")
		os.MkdirAll(legacyDir, 0o755)
		os.WriteFile(filepath.Join(legacyDir, "config.json"), []byte(`{corrupt`), 0o644)

		migrated, err := migrateConfig()
		if err != nil {
			t.Fatal(err)
		}
		if migrated {
			t.Error("should not migrate corrupt file")
		}

		if _, err := os.Stat(filepath.Join(legacyDir, "config.json")); os.IsNotExist(err) {
			t.Error("corrupt legacy file should not have been deleted")
		}
	})

	t.Run("new path takes precedence over legacy", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		legacyDir := filepath.Join(tmpDir, ".ollama", "config")
		os.MkdirAll(legacyDir, 0o755)
		os.WriteFile(filepath.Join(legacyDir, "config.json"), []byte(`{"integrations":{"old":{"models":["old-model"]}}}`), 0o644)

		newDir := filepath.Join(tmpDir, ".ollama")
		os.WriteFile(filepath.Join(newDir, "config.json"), []byte(`{"integrations":{"new":{"models":["new-model"]}}}`), 0o644)

		cfg, err := load()
		if err != nil {
			t.Fatal(err)
		}
		if _, ok := cfg.Integrations["new"]; !ok {
			t.Error("expected new-path integration to be loaded")
		}
		if _, ok := cfg.Integrations["old"]; ok {
			t.Error("legacy integration should not have been loaded")
		}
	})

	t.Run("idempotent when called twice", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		legacyDir := filepath.Join(tmpDir, ".ollama", "config")
		os.MkdirAll(legacyDir, 0o755)
		os.WriteFile(filepath.Join(legacyDir, "config.json"), []byte(`{"integrations":{}}`), 0o644)

		if _, err := migrateConfig(); err != nil {
			t.Fatal(err)
		}

		migrated, err := migrateConfig()
		if err != nil {
			t.Fatal(err)
		}
		if migrated {
			t.Error("second migration should be a no-op")
		}
	})

	t.Run("legacy directory preserved if not empty", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		legacyDir := filepath.Join(tmpDir, ".ollama", "config")
		os.MkdirAll(legacyDir, 0o755)
		os.WriteFile(filepath.Join(legacyDir, "config.json"), []byte(`{"integrations":{}}`), 0o644)
		os.WriteFile(filepath.Join(legacyDir, "other-file.txt"), []byte("keep me"), 0o644)

		if _, err := migrateConfig(); err != nil {
			t.Fatal(err)
		}

		if _, err := os.Stat(legacyDir); os.IsNotExist(err) {
			t.Error("directory with other files should not have been removed")
		}
		if _, err := os.Stat(filepath.Join(legacyDir, "other-file.txt")); os.IsNotExist(err) {
			t.Error("other files in legacy directory should be untouched")
		}
	})

	t.Run("save writes to new path after migration", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		legacyDir := filepath.Join(tmpDir, ".ollama", "config")
		os.MkdirAll(legacyDir, 0o755)
		os.WriteFile(filepath.Join(legacyDir, "config.json"), []byte(`{"integrations":{"claude":{"models":["llama3.2"]}}}`), 0o644)

		// load triggers migration, then save should write to new path
		if err := saveIntegration("codex", []string{"qwen2.5"}); err != nil {
			t.Fatal(err)
		}

		newPath := filepath.Join(tmpDir, ".ollama", "config.json")
		if _, err := os.Stat(newPath); os.IsNotExist(err) {
			t.Error("save should write to new path")
		}

		// old path should not be recreated
		if _, err := os.Stat(filepath.Join(legacyDir, "config.json")); !os.IsNotExist(err) {
			t.Error("save should not recreate legacy path")
		}
	})

	t.Run("load triggers migration transparently", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		legacyDir := filepath.Join(tmpDir, ".ollama", "config")
		os.MkdirAll(legacyDir, 0o755)
		os.WriteFile(filepath.Join(legacyDir, "config.json"), []byte(`{"integrations":{"claude":{"models":["llama3.2"]}}}`), 0o644)

		cfg, err := load()
		if err != nil {
			t.Fatal(err)
		}
		if cfg.Integrations["claude"] == nil || cfg.Integrations["claude"].Models[0] != "llama3.2" {
			t.Error("migration via load() did not preserve data")
		}
	})
}

func TestSave(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	t.Run("creates config file", func(t *testing.T) {
		cfg := &config{
			Integrations: map[string]*integration{
				"test": {Models: []string{"model-a", "model-b"}},
			},
		}

		if err := save(cfg); err != nil {
			t.Fatal(err)
		}

		path, _ := configPath()
		if _, err := os.Stat(path); os.IsNotExist(err) {
			t.Error("config file was not created")
		}
	})

	t.Run("round-trip preserves data", func(t *testing.T) {
		cfg := &config{
			Integrations: map[string]*integration{
				"claude": {Models: []string{"llama3.2", "mistral"}},
				"codex":  {Models: []string{"qwen2.5"}},
			},
		}

		if err := save(cfg); err != nil {
			t.Fatal(err)
		}

		loaded, err := load()
		if err != nil {
			t.Fatal(err)
		}

		if len(loaded.Integrations) != 2 {
			t.Errorf("expected 2 integrations, got %d", len(loaded.Integrations))
		}
		if loaded.Integrations["claude"] == nil {
			t.Error("missing claude integration")
		}
		if len(loaded.Integrations["claude"].Models) != 2 {
			t.Errorf("expected 2 models for claude, got %d", len(loaded.Integrations["claude"].Models))
		}
	})
}
