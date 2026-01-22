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

// Edge case tests for config.go

// TestLoadIntegration_CorruptedJSON verifies that corrupted JSON returns a clear error, not a panic.
// Users may have manually edited config files or have disk corruption.
func TestLoadIntegration_CorruptedJSON(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	// Create corrupted JSON file
	dir := filepath.Join(tmpDir, ".ollama", "config", "integrations")
	os.MkdirAll(dir, 0o755)
	os.WriteFile(filepath.Join(dir, "test.json"), []byte(`{corrupted json`), 0o644)

	_, err := loadIntegration("test")
	if err == nil {
		t.Error("expected error for corrupted JSON, got nil")
	}
}

// TestLoadIntegration_EmptyFile verifies that empty files return clear errors.
// Empty file is invalid JSON.
func TestLoadIntegration_EmptyFile(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	dir := filepath.Join(tmpDir, ".ollama", "config", "integrations")
	os.MkdirAll(dir, 0o755)
	os.WriteFile(filepath.Join(dir, "empty.json"), []byte(``), 0o644)

	_, err := loadIntegration("empty")
	if err == nil {
		t.Error("expected error for empty file, got nil")
	}
}

// TestSaveIntegration_NilModels verifies that nil models slice works correctly.
// Both nil and []string{} should work without issues.
func TestSaveIntegration_NilModels(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	// Save with nil models
	if err := saveIntegration("test", nil); err != nil {
		t.Fatalf("saveIntegration with nil models failed: %v", err)
	}

	// Verify it can be loaded back
	config, err := loadIntegration("test")
	if err != nil {
		t.Fatalf("loadIntegration failed: %v", err)
	}

	// nil becomes empty array in JSON, either is acceptable
	if config.Models == nil {
		// Some JSON implementations preserve nil as nil
	} else if len(config.Models) != 0 {
		t.Errorf("expected empty or nil models, got %v", config.Models)
	}
}

// TestSaveIntegration_EmptyAppName verifies that empty app name returns an error.
// Empty app name would create ".json" file which is confusing.
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

// TestListIntegrations_SkipsCorruptedFiles documents intentional behavior: corrupted files are silently skipped.
// Partial config is better than total failure.
func TestListIntegrations_SkipsCorruptedFiles(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	dir := filepath.Join(tmpDir, ".ollama", "config", "integrations")
	os.MkdirAll(dir, 0o755)

	// Create one valid and one corrupted config
	os.WriteFile(filepath.Join(dir, "valid.json"), []byte(`{"app":"valid","models":["m1"]}`), 0o644)
	os.WriteFile(filepath.Join(dir, "corrupted.json"), []byte(`{corrupted`), 0o644)

	configs, err := listIntegrations()
	if err != nil {
		t.Fatalf("listIntegrations failed: %v", err)
	}

	// Should return the valid config, skipping corrupted
	if len(configs) != 1 {
		t.Errorf("expected 1 valid config, got %d", len(configs))
	}
	if len(configs) > 0 && configs[0].App != "valid" {
		t.Errorf("expected 'valid' app, got %s", configs[0].App)
	}
}

// TestListIntegrations_IgnoresNonJSON verifies that non-JSON files in directory are ignored.
// Directory may contain backup files, READMEs, or other non-config files.
func TestListIntegrations_IgnoresNonJSON(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	dir := filepath.Join(tmpDir, ".ollama", "config", "integrations")
	os.MkdirAll(dir, 0o755)

	// Create various non-JSON files
	os.WriteFile(filepath.Join(dir, "readme.txt"), []byte("readme"), 0o644)
	os.WriteFile(filepath.Join(dir, "backup.bak"), []byte("backup"), 0o644)
	os.WriteFile(filepath.Join(dir, ".hidden"), []byte("hidden"), 0o644)
	os.WriteFile(filepath.Join(dir, "valid.json"), []byte(`{"app":"valid","models":[]}`), 0o644)

	configs, err := listIntegrations()
	if err != nil {
		t.Fatalf("listIntegrations failed: %v", err)
	}

	// Should only return the JSON config
	if len(configs) != 1 {
		t.Errorf("expected 1 config (only .json), got %d", len(configs))
	}
}

// TestLoadIntegration_NonexistentFile verifies clear error for missing file.
func TestLoadIntegration_NonexistentFile(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	_, err := loadIntegration("nonexistent")
	if err == nil {
		t.Error("expected error for nonexistent file, got nil")
	}
	if !os.IsNotExist(err) {
		t.Logf("error type is not os.IsNotExist, but that's acceptable: %v", err)
	}
}
