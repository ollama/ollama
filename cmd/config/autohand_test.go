package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestAutohandIntegration(t *testing.T) {
	ah := &Autohand{}

	t.Run("String", func(t *testing.T) {
		if got := ah.String(); got != "Autohand Code" {
			t.Errorf("String() = %q, want %q", got, "Autohand Code")
		}
	})

	t.Run("implements Runner", func(t *testing.T) {
		var _ Runner = ah
	})
}

func TestConfigureAutohand(t *testing.T) {
	t.Run("creates config with provider and model", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		t.Setenv("OLLAMA_HOST", "http://localhost:11434")

		if err := configureAutoHand("llama3.2"); err != nil {
			t.Fatalf("configureAutoHand() error = %v", err)
		}

		configPath := filepath.Join(tmpDir, ".autohand", "config.json")
		data, err := os.ReadFile(configPath)
		if err != nil {
			t.Fatalf("Failed to read config: %v", err)
		}

		var cfg map[string]any
		if err := json.Unmarshal(data, &cfg); err != nil {
			t.Fatalf("Failed to parse config: %v", err)
		}

		if cfg["provider"] != "ollama" {
			t.Errorf("provider = %v, want ollama", cfg["provider"])
		}

		ollama, ok := cfg["ollama"].(map[string]any)
		if !ok {
			t.Fatal("Config missing ollama section")
		}

		if ollama["baseUrl"] != "http://localhost:11434" {
			t.Errorf("baseUrl = %v, want http://localhost:11434", ollama["baseUrl"])
		}
		if ollama["model"] != "llama3.2" {
			t.Errorf("model = %v, want llama3.2", ollama["model"])
		}
	})

	t.Run("preserves existing fields", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		t.Setenv("OLLAMA_HOST", "http://localhost:11434")

		configDir := filepath.Join(tmpDir, ".autohand")
		if err := os.MkdirAll(configDir, 0o755); err != nil {
			t.Fatal(err)
		}
		configPath := filepath.Join(configDir, "config.json")

		existingConfig := `{
			"provider": "openai",
			"openai": {
				"apiKey": "sk-test",
				"model": "gpt-4o"
			},
			"theme": "dark"
		}`
		if err := os.WriteFile(configPath, []byte(existingConfig), 0o644); err != nil {
			t.Fatal(err)
		}

		if err := configureAutoHand("qwen3:8b"); err != nil {
			t.Fatalf("configureAutoHand() error = %v", err)
		}

		data, err := os.ReadFile(configPath)
		if err != nil {
			t.Fatalf("Failed to read config: %v", err)
		}

		var cfg map[string]any
		if err := json.Unmarshal(data, &cfg); err != nil {
			t.Fatalf("Failed to parse config: %v", err)
		}

		// Provider should be updated to ollama
		if cfg["provider"] != "ollama" {
			t.Errorf("provider = %v, want ollama", cfg["provider"])
		}

		// OpenAI config should be preserved
		openai, ok := cfg["openai"].(map[string]any)
		if !ok {
			t.Error("openai config should be preserved")
		} else {
			if openai["apiKey"] != "sk-test" {
				t.Errorf("openai.apiKey = %v, want sk-test", openai["apiKey"])
			}
		}

		// Custom fields should be preserved
		if cfg["theme"] != "dark" {
			t.Errorf("theme = %v, want dark (preserved)", cfg["theme"])
		}

		// Ollama config should be set
		ollama := cfg["ollama"].(map[string]any)
		if ollama["model"] != "qwen3:8b" {
			t.Errorf("ollama.model = %v, want qwen3:8b", ollama["model"])
		}
	})

	t.Run("preserves existing ollama fields", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		t.Setenv("OLLAMA_HOST", "http://custom:8080")

		configDir := filepath.Join(tmpDir, ".autohand")
		if err := os.MkdirAll(configDir, 0o755); err != nil {
			t.Fatal(err)
		}
		configPath := filepath.Join(configDir, "config.json")

		existingConfig := `{
			"provider": "ollama",
			"ollama": {
				"baseUrl": "http://old:11434",
				"model": "old-model",
				"customField": "preserved"
			}
		}`
		if err := os.WriteFile(configPath, []byte(existingConfig), 0o644); err != nil {
			t.Fatal(err)
		}

		if err := configureAutoHand("new-model"); err != nil {
			t.Fatalf("configureAutoHand() error = %v", err)
		}

		data, err := os.ReadFile(configPath)
		if err != nil {
			t.Fatalf("Failed to read config: %v", err)
		}

		var cfg map[string]any
		if err := json.Unmarshal(data, &cfg); err != nil {
			t.Fatalf("Failed to parse config: %v", err)
		}

		ollama := cfg["ollama"].(map[string]any)

		// baseUrl should be updated to current OLLAMA_HOST
		if ollama["baseUrl"] != "http://custom:8080" {
			t.Errorf("baseUrl = %v, want http://custom:8080", ollama["baseUrl"])
		}
		// model should be updated
		if ollama["model"] != "new-model" {
			t.Errorf("model = %v, want new-model", ollama["model"])
		}
		// custom fields should be preserved
		if ollama["customField"] != "preserved" {
			t.Errorf("customField = %v, want preserved", ollama["customField"])
		}
	})

	t.Run("handles corrupt config gracefully", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		t.Setenv("OLLAMA_HOST", "http://localhost:11434")

		configDir := filepath.Join(tmpDir, ".autohand")
		if err := os.MkdirAll(configDir, 0o755); err != nil {
			t.Fatal(err)
		}
		configPath := filepath.Join(configDir, "config.json")
		if err := os.WriteFile(configPath, []byte("{invalid json}"), 0o644); err != nil {
			t.Fatal(err)
		}

		if err := configureAutoHand("test-model"); err != nil {
			t.Fatalf("configureAutoHand() should not fail with corrupt config, got %v", err)
		}

		data, err := os.ReadFile(configPath)
		if err != nil {
			t.Fatalf("Failed to read config: %v", err)
		}

		var cfg map[string]any
		if err := json.Unmarshal(data, &cfg); err != nil {
			t.Fatalf("Config should be valid after configureAutoHand, got parse error: %v", err)
		}

		if cfg["provider"] != "ollama" {
			t.Errorf("provider = %v, want ollama", cfg["provider"])
		}
		ollama := cfg["ollama"].(map[string]any)
		if ollama["model"] != "test-model" {
			t.Errorf("model = %v, want test-model", ollama["model"])
		}
	})

	t.Run("creates config directory if missing", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		t.Setenv("OLLAMA_HOST", "http://localhost:11434")

		// Don't create .autohand directory - let configureAutoHand do it
		if err := configureAutoHand("test-model"); err != nil {
			t.Fatalf("configureAutoHand() error = %v", err)
		}

		configPath := filepath.Join(tmpDir, ".autohand", "config.json")
		if _, err := os.Stat(configPath); os.IsNotExist(err) {
			t.Error("config.json should be created")
		}
	})

	t.Run("updates model on subsequent calls", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)
		t.Setenv("OLLAMA_HOST", "http://localhost:11434")

		if err := configureAutoHand("model-1"); err != nil {
			t.Fatalf("first configureAutoHand() error = %v", err)
		}

		if err := configureAutoHand("model-2"); err != nil {
			t.Fatalf("second configureAutoHand() error = %v", err)
		}

		configPath := filepath.Join(tmpDir, ".autohand", "config.json")
		data, err := os.ReadFile(configPath)
		if err != nil {
			t.Fatalf("Failed to read config: %v", err)
		}

		var cfg map[string]any
		if err := json.Unmarshal(data, &cfg); err != nil {
			t.Fatalf("Failed to parse config: %v", err)
		}

		ollama := cfg["ollama"].(map[string]any)
		if ollama["model"] != "model-2" {
			t.Errorf("model = %v, want model-2 (should be updated)", ollama["model"])
		}
	})
}
