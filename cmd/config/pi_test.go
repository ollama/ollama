package config

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func TestPiIntegration(t *testing.T) {
	pi := &Pi{}

	t.Run("String", func(t *testing.T) {
		if got := pi.String(); got != "Pi" {
			t.Errorf("String() = %q, want %q", got, "Pi")
		}
	})

	t.Run("implements Runner", func(t *testing.T) {
		var _ Runner = pi
	})

	t.Run("implements Editor", func(t *testing.T) {
		var _ Editor = pi
	})
}

func TestPiPaths(t *testing.T) {
	pi := &Pi{}

	t.Run("returns empty when no config exists", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		paths := pi.Paths()
		if len(paths) != 0 {
			t.Errorf("Paths() = %v, want empty", paths)
		}
	})

	t.Run("returns path when config exists", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		configDir := filepath.Join(tmpDir, ".pi", "agent")
		if err := os.MkdirAll(configDir, 0o755); err != nil {
			t.Fatal(err)
		}
		configPath := filepath.Join(configDir, "models.json")
		if err := os.WriteFile(configPath, []byte("{}"), 0o644); err != nil {
			t.Fatal(err)
		}

		paths := pi.Paths()
		if len(paths) != 1 || paths[0] != configPath {
			t.Errorf("Paths() = %v, want [%s]", paths, configPath)
		}
	})
}

func TestPiEdit(t *testing.T) {
	// Mock Ollama server for createConfig calls during Edit
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/show" {
			fmt.Fprintf(w, `{"capabilities":[],"model_info":{}}`)
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	pi := &Pi{}
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	configDir := filepath.Join(tmpDir, ".pi", "agent")
	configPath := filepath.Join(configDir, "models.json")

	cleanup := func() {
		os.RemoveAll(configDir)
	}

	readConfig := func() map[string]any {
		data, _ := os.ReadFile(configPath)
		var cfg map[string]any
		json.Unmarshal(data, &cfg)
		return cfg
	}

	t.Run("returns nil for empty models", func(t *testing.T) {
		if err := pi.Edit([]string{}); err != nil {
			t.Errorf("Edit([]) error = %v, want nil", err)
		}
	})

	t.Run("creates config with models", func(t *testing.T) {
		cleanup()

		models := []string{"llama3.2", "qwen3:8b"}
		if err := pi.Edit(models); err != nil {
			t.Fatalf("Edit() error = %v", err)
		}

		cfg := readConfig()

		providers, ok := cfg["providers"].(map[string]any)
		if !ok {
			t.Error("Config missing providers")
		}

		ollama, ok := providers["ollama"].(map[string]any)
		if !ok {
			t.Error("Providers missing ollama")
		}

		modelsArray, ok := ollama["models"].([]any)
		if !ok || len(modelsArray) != 2 {
			t.Errorf("Expected 2 models, got %v", modelsArray)
		}

		if ollama["baseUrl"] == nil {
			t.Error("Missing baseUrl")
		}
		if ollama["api"] != "openai-completions" {
			t.Errorf("Expected api=openai-completions, got %v", ollama["api"])
		}
		if ollama["apiKey"] != "ollama" {
			t.Errorf("Expected apiKey=ollama, got %v", ollama["apiKey"])
		}
	})

	t.Run("updates existing config preserving ollama provider settings", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)

		existingConfig := `{
			"providers": {
				"ollama": {
					"baseUrl": "http://custom:8080/v1",
					"api": "custom-api",
					"apiKey": "custom-key",
					"models": [
						{"id": "old-model", "_launch": true}
					]
				}
			}
		}`
		if err := os.WriteFile(configPath, []byte(existingConfig), 0o644); err != nil {
			t.Fatal(err)
		}

		models := []string{"new-model"}
		if err := pi.Edit(models); err != nil {
			t.Fatalf("Edit() error = %v", err)
		}

		cfg := readConfig()
		providers := cfg["providers"].(map[string]any)
		ollama := providers["ollama"].(map[string]any)

		if ollama["baseUrl"] != "http://custom:8080/v1" {
			t.Errorf("Custom baseUrl not preserved, got %v", ollama["baseUrl"])
		}
		if ollama["api"] != "custom-api" {
			t.Errorf("Custom api not preserved, got %v", ollama["api"])
		}
		if ollama["apiKey"] != "custom-key" {
			t.Errorf("Custom apiKey not preserved, got %v", ollama["apiKey"])
		}

		modelsArray := ollama["models"].([]any)
		if len(modelsArray) != 1 {
			t.Errorf("Expected 1 model after update, got %d", len(modelsArray))
		} else {
			modelEntry := modelsArray[0].(map[string]any)
			if modelEntry["id"] != "new-model" {
				t.Errorf("Expected new-model, got %v", modelEntry["id"])
			}
			// Verify _launch marker is present
			if modelEntry["_launch"] != true {
				t.Errorf("Expected _launch marker to be true")
			}
		}
	})

	t.Run("replaces old models with new ones", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)

		// Old models must have _launch marker to be managed by us
		existingConfig := `{
			"providers": {
				"ollama": {
					"baseUrl": "http://localhost:11434/v1",
					"api": "openai-completions",
					"apiKey": "ollama",
					"models": [
						{"id": "old-model-1", "_launch": true},
						{"id": "old-model-2", "_launch": true}
					]
				}
			}
		}`
		if err := os.WriteFile(configPath, []byte(existingConfig), 0o644); err != nil {
			t.Fatal(err)
		}

		newModels := []string{"new-model-1", "new-model-2"}
		if err := pi.Edit(newModels); err != nil {
			t.Fatalf("Edit() error = %v", err)
		}

		cfg := readConfig()
		providers := cfg["providers"].(map[string]any)
		ollama := providers["ollama"].(map[string]any)
		modelsArray := ollama["models"].([]any)

		if len(modelsArray) != 2 {
			t.Errorf("Expected 2 models, got %d", len(modelsArray))
		}

		modelIDs := make(map[string]bool)
		for _, m := range modelsArray {
			modelObj := m.(map[string]any)
			id := modelObj["id"].(string)
			modelIDs[id] = true
		}

		if !modelIDs["new-model-1"] || !modelIDs["new-model-2"] {
			t.Errorf("Expected new models, got %v", modelIDs)
		}
		if modelIDs["old-model-1"] || modelIDs["old-model-2"] {
			t.Errorf("Old models should have been removed, got %v", modelIDs)
		}
	})

	t.Run("handles partial overlap in model list", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)

		// Models must have _launch marker to be managed
		existingConfig := `{
			"providers": {
				"ollama": {
					"baseUrl": "http://localhost:11434/v1",
					"api": "openai-completions",
					"apiKey": "ollama",
					"models": [
						{"id": "keep-model", "_launch": true},
						{"id": "remove-model", "_launch": true}
					]
				}
			}
		}`
		if err := os.WriteFile(configPath, []byte(existingConfig), 0o644); err != nil {
			t.Fatal(err)
		}

		newModels := []string{"keep-model", "add-model"}
		if err := pi.Edit(newModels); err != nil {
			t.Fatalf("Edit() error = %v", err)
		}

		cfg := readConfig()
		providers := cfg["providers"].(map[string]any)
		ollama := providers["ollama"].(map[string]any)
		modelsArray := ollama["models"].([]any)

		if len(modelsArray) != 2 {
			t.Errorf("Expected 2 models, got %d", len(modelsArray))
		}

		modelIDs := make(map[string]bool)
		for _, m := range modelsArray {
			modelObj := m.(map[string]any)
			id := modelObj["id"].(string)
			modelIDs[id] = true
		}

		if !modelIDs["keep-model"] || !modelIDs["add-model"] {
			t.Errorf("Expected keep-model and add-model, got %v", modelIDs)
		}
		if modelIDs["remove-model"] {
			t.Errorf("remove-model should have been removed")
		}
	})

	t.Run("handles corrupt config gracefully", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)

		if err := os.WriteFile(configPath, []byte("{invalid json}"), 0o644); err != nil {
			t.Fatal(err)
		}

		models := []string{"test-model"}
		if err := pi.Edit(models); err != nil {
			t.Fatalf("Edit() should not fail with corrupt config, got %v", err)
		}

		data, err := os.ReadFile(configPath)
		if err != nil {
			t.Fatalf("Failed to read config: %v", err)
		}

		var cfg map[string]any
		if err := json.Unmarshal(data, &cfg); err != nil {
			t.Fatalf("Config should be valid after Edit, got parse error: %v", err)
		}

		providers := cfg["providers"].(map[string]any)
		ollama := providers["ollama"].(map[string]any)
		modelsArray := ollama["models"].([]any)

		if len(modelsArray) != 1 {
			t.Errorf("Expected 1 model, got %d", len(modelsArray))
		}
	})

	// CRITICAL SAFETY TEST: verifies we don't stomp on user configs
	t.Run("preserves user-managed models without _launch marker", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)

		// User has manually configured models in ollama provider (no _launch marker)
		existingConfig := `{
			"providers": {
				"ollama": {
					"baseUrl": "http://localhost:11434/v1",
					"api": "openai-completions",
					"apiKey": "ollama",
					"models": [
						{"id": "user-model-1"},
						{"id": "user-model-2", "customField": "preserved"},
						{"id": "ollama-managed", "_launch": true}
					]
				}
			}
		}`
		if err := os.WriteFile(configPath, []byte(existingConfig), 0o644); err != nil {
			t.Fatal(err)
		}

		// Add a new ollama-managed model
		newModels := []string{"new-ollama-model"}
		if err := pi.Edit(newModels); err != nil {
			t.Fatalf("Edit() error = %v", err)
		}

		cfg := readConfig()
		providers := cfg["providers"].(map[string]any)
		ollama := providers["ollama"].(map[string]any)
		modelsArray := ollama["models"].([]any)

		// Should have: new-ollama-model (managed) + 2 user models (preserved)
		if len(modelsArray) != 3 {
			t.Errorf("Expected 3 models (1 new managed + 2 preserved user models), got %d", len(modelsArray))
		}

		modelIDs := make(map[string]map[string]any)
		for _, m := range modelsArray {
			modelObj := m.(map[string]any)
			id := modelObj["id"].(string)
			modelIDs[id] = modelObj
		}

		// Verify new model has _launch marker
		if m, ok := modelIDs["new-ollama-model"]; !ok {
			t.Errorf("new-ollama-model should be present")
		} else if m["_launch"] != true {
			t.Errorf("new-ollama-model should have _launch marker")
		}

		// Verify user models are preserved
		if _, ok := modelIDs["user-model-1"]; !ok {
			t.Errorf("user-model-1 should be preserved")
		}
		if _, ok := modelIDs["user-model-2"]; !ok {
			t.Errorf("user-model-2 should be preserved")
		} else if modelIDs["user-model-2"]["customField"] != "preserved" {
			t.Errorf("user-model-2 customField should be preserved")
		}

		// Verify old ollama-managed model is removed (not in new list)
		if _, ok := modelIDs["ollama-managed"]; ok {
			t.Errorf("ollama-managed should be removed (old ollama model not in new selection)")
		}
	})

	t.Run("updates settings.json with default provider and model", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)

		// Create existing settings with other fields
		settingsPath := filepath.Join(configDir, "settings.json")
		existingSettings := `{
			"theme": "dark",
			"customSetting": "value",
			"defaultProvider": "anthropic",
			"defaultModel": "claude-3"
		}`
		if err := os.WriteFile(settingsPath, []byte(existingSettings), 0o644); err != nil {
			t.Fatal(err)
		}

		models := []string{"llama3.2"}
		if err := pi.Edit(models); err != nil {
			t.Fatalf("Edit() error = %v", err)
		}

		data, err := os.ReadFile(settingsPath)
		if err != nil {
			t.Fatalf("Failed to read settings: %v", err)
		}

		var settings map[string]any
		if err := json.Unmarshal(data, &settings); err != nil {
			t.Fatalf("Failed to parse settings: %v", err)
		}

		// Verify defaultProvider is set to ollama
		if settings["defaultProvider"] != "ollama" {
			t.Errorf("defaultProvider = %v, want ollama", settings["defaultProvider"])
		}

		// Verify defaultModel is set to first model
		if settings["defaultModel"] != "llama3.2" {
			t.Errorf("defaultModel = %v, want llama3.2", settings["defaultModel"])
		}

		// Verify other fields are preserved
		if settings["theme"] != "dark" {
			t.Errorf("theme = %v, want dark (preserved)", settings["theme"])
		}
		if settings["customSetting"] != "value" {
			t.Errorf("customSetting = %v, want value (preserved)", settings["customSetting"])
		}
	})

	t.Run("creates settings.json if it does not exist", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)

		models := []string{"qwen3:8b"}
		if err := pi.Edit(models); err != nil {
			t.Fatalf("Edit() error = %v", err)
		}

		settingsPath := filepath.Join(configDir, "settings.json")
		data, err := os.ReadFile(settingsPath)
		if err != nil {
			t.Fatalf("settings.json should be created: %v", err)
		}

		var settings map[string]any
		if err := json.Unmarshal(data, &settings); err != nil {
			t.Fatalf("Failed to parse settings: %v", err)
		}

		if settings["defaultProvider"] != "ollama" {
			t.Errorf("defaultProvider = %v, want ollama", settings["defaultProvider"])
		}
		if settings["defaultModel"] != "qwen3:8b" {
			t.Errorf("defaultModel = %v, want qwen3:8b", settings["defaultModel"])
		}
	})

	t.Run("handles corrupt settings.json gracefully", func(t *testing.T) {
		cleanup()
		os.MkdirAll(configDir, 0o755)

		// Create corrupt settings
		settingsPath := filepath.Join(configDir, "settings.json")
		if err := os.WriteFile(settingsPath, []byte("{invalid"), 0o644); err != nil {
			t.Fatal(err)
		}

		models := []string{"test-model"}
		if err := pi.Edit(models); err != nil {
			t.Fatalf("Edit() should not fail with corrupt settings, got %v", err)
		}

		data, err := os.ReadFile(settingsPath)
		if err != nil {
			t.Fatalf("Failed to read settings: %v", err)
		}

		var settings map[string]any
		if err := json.Unmarshal(data, &settings); err != nil {
			t.Fatalf("settings.json should be valid after Edit, got parse error: %v", err)
		}

		if settings["defaultProvider"] != "ollama" {
			t.Errorf("defaultProvider = %v, want ollama", settings["defaultProvider"])
		}
		if settings["defaultModel"] != "test-model" {
			t.Errorf("defaultModel = %v, want test-model", settings["defaultModel"])
		}
	})
}

func TestPiModels(t *testing.T) {
	pi := &Pi{}

	t.Run("returns nil when no config exists", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		models := pi.Models()
		if models != nil {
			t.Errorf("Models() = %v, want nil", models)
		}
	})

	t.Run("returns models from config", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		configDir := filepath.Join(tmpDir, ".pi", "agent")
		if err := os.MkdirAll(configDir, 0o755); err != nil {
			t.Fatal(err)
		}
		config := `{
			"providers": {
				"ollama": {
					"models": [
						{"id": "llama3.2"},
						{"id": "qwen3:8b"}
					]
				}
			}
		}`
		configPath := filepath.Join(configDir, "models.json")
		if err := os.WriteFile(configPath, []byte(config), 0o644); err != nil {
			t.Fatal(err)
		}

		models := pi.Models()
		if len(models) != 2 {
			t.Errorf("Models() returned %d models, want 2", len(models))
		}
		if models[0] != "llama3.2" || models[1] != "qwen3:8b" {
			t.Errorf("Models() = %v, want [llama3.2 qwen3:8b] (sorted)", models)
		}
	})

	t.Run("returns sorted models", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		configDir := filepath.Join(tmpDir, ".pi", "agent")
		if err := os.MkdirAll(configDir, 0o755); err != nil {
			t.Fatal(err)
		}
		config := `{
			"providers": {
				"ollama": {
					"models": [
						{"id": "z-model"},
						{"id": "a-model"},
						{"id": "m-model"}
					]
				}
			}
		}`
		configPath := filepath.Join(configDir, "models.json")
		if err := os.WriteFile(configPath, []byte(config), 0o644); err != nil {
			t.Fatal(err)
		}

		models := pi.Models()
		if models[0] != "a-model" || models[1] != "m-model" || models[2] != "z-model" {
			t.Errorf("Models() = %v, want [a-model m-model z-model] (sorted)", models)
		}
	})

	t.Run("returns nil when models array is missing", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		configDir := filepath.Join(tmpDir, ".pi", "agent")
		if err := os.MkdirAll(configDir, 0o755); err != nil {
			t.Fatal(err)
		}
		config := `{
			"providers": {
				"ollama": {}
			}
		}`
		configPath := filepath.Join(configDir, "models.json")
		if err := os.WriteFile(configPath, []byte(config), 0o644); err != nil {
			t.Fatal(err)
		}

		models := pi.Models()
		if models != nil {
			t.Errorf("Models() = %v, want nil when models array is missing", models)
		}
	})

	t.Run("handles corrupt config gracefully", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		configDir := filepath.Join(tmpDir, ".pi", "agent")
		if err := os.MkdirAll(configDir, 0o755); err != nil {
			t.Fatal(err)
		}
		configPath := filepath.Join(configDir, "models.json")
		if err := os.WriteFile(configPath, []byte("{invalid json}"), 0o644); err != nil {
			t.Fatal(err)
		}

		models := pi.Models()
		if models != nil {
			t.Errorf("Models() = %v, want nil for corrupt config", models)
		}
	})
}

func TestIsPiOllamaModel(t *testing.T) {
	tests := []struct {
		name string
		cfg  map[string]any
		want bool
	}{
		{"with _launch true", map[string]any{"id": "m", "_launch": true}, true},
		{"with _launch false", map[string]any{"id": "m", "_launch": false}, false},
		{"without _launch", map[string]any{"id": "m"}, false},
		{"with _launch non-bool", map[string]any{"id": "m", "_launch": "yes"}, false},
		{"empty map", map[string]any{}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isPiOllamaModel(tt.cfg); got != tt.want {
				t.Errorf("isPiOllamaModel(%v) = %v, want %v", tt.cfg, got, tt.want)
			}
		})
	}
}

func TestCreateConfig(t *testing.T) {
	t.Run("sets vision input when model has vision capability", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/show" {
				fmt.Fprintf(w, `{"capabilities":["vision"],"model_info":{}}`)
				return
			}
			w.WriteHeader(http.StatusNotFound)
		}))
		defer srv.Close()

		u, _ := url.Parse(srv.URL)
		client := api.NewClient(u, srv.Client())

		cfg := createConfig(context.Background(), client, "llava:7b")

		if cfg["id"] != "llava:7b" {
			t.Errorf("id = %v, want llava:7b", cfg["id"])
		}
		if cfg["_launch"] != true {
			t.Error("expected _launch = true")
		}
		input, ok := cfg["input"].([]string)
		if !ok || len(input) != 2 || input[0] != "text" || input[1] != "image" {
			t.Errorf("input = %v, want [text image]", cfg["input"])
		}
	})

	t.Run("sets text-only input when model lacks vision", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/show" {
				fmt.Fprintf(w, `{"capabilities":["completion"],"model_info":{}}`)
				return
			}
			w.WriteHeader(http.StatusNotFound)
		}))
		defer srv.Close()

		u, _ := url.Parse(srv.URL)
		client := api.NewClient(u, srv.Client())

		cfg := createConfig(context.Background(), client, "llama3.2")

		input, ok := cfg["input"].([]string)
		if !ok || len(input) != 1 || input[0] != "text" {
			t.Errorf("input = %v, want [text]", cfg["input"])
		}
		if _, ok := cfg["reasoning"]; ok {
			t.Error("reasoning should not be set for non-thinking model")
		}
	})

	t.Run("sets reasoning when model has thinking capability", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/show" {
				fmt.Fprintf(w, `{"capabilities":["thinking"],"model_info":{}}`)
				return
			}
			w.WriteHeader(http.StatusNotFound)
		}))
		defer srv.Close()

		u, _ := url.Parse(srv.URL)
		client := api.NewClient(u, srv.Client())

		cfg := createConfig(context.Background(), client, "qwq")

		if cfg["reasoning"] != true {
			t.Error("expected reasoning = true for thinking model")
		}
	})

	t.Run("extracts context window from model info", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/show" {
				fmt.Fprintf(w, `{"capabilities":[],"model_info":{"llama.context_length":131072}}`)
				return
			}
			w.WriteHeader(http.StatusNotFound)
		}))
		defer srv.Close()

		u, _ := url.Parse(srv.URL)
		client := api.NewClient(u, srv.Client())

		cfg := createConfig(context.Background(), client, "llama3.2")

		if cfg["contextWindow"] != 131072 {
			t.Errorf("contextWindow = %v, want 131072", cfg["contextWindow"])
		}
	})

	t.Run("handles all capabilities together", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/show" {
				fmt.Fprintf(w, `{"capabilities":["vision","thinking"],"model_info":{"qwen3.context_length":32768}}`)
				return
			}
			w.WriteHeader(http.StatusNotFound)
		}))
		defer srv.Close()

		u, _ := url.Parse(srv.URL)
		client := api.NewClient(u, srv.Client())

		cfg := createConfig(context.Background(), client, "qwen3-vision")

		input := cfg["input"].([]string)
		if len(input) != 2 || input[0] != "text" || input[1] != "image" {
			t.Errorf("input = %v, want [text image]", input)
		}
		if cfg["reasoning"] != true {
			t.Error("expected reasoning = true")
		}
		if cfg["contextWindow"] != 32768 {
			t.Errorf("contextWindow = %v, want 32768", cfg["contextWindow"])
		}
	})

	t.Run("returns minimal config when show fails", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprintf(w, `{"error":"model not found"}`)
		}))
		defer srv.Close()

		u, _ := url.Parse(srv.URL)
		client := api.NewClient(u, srv.Client())

		cfg := createConfig(context.Background(), client, "missing-model")

		if cfg["id"] != "missing-model" {
			t.Errorf("id = %v, want missing-model", cfg["id"])
		}
		if cfg["_launch"] != true {
			t.Error("expected _launch = true")
		}
		// Should not have capability fields
		if _, ok := cfg["input"]; ok {
			t.Error("input should not be set when show fails")
		}
		if _, ok := cfg["reasoning"]; ok {
			t.Error("reasoning should not be set when show fails")
		}
		if _, ok := cfg["contextWindow"]; ok {
			t.Error("contextWindow should not be set when show fails")
		}
	})

	t.Run("skips zero context length", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/show" {
				fmt.Fprintf(w, `{"capabilities":[],"model_info":{"llama.context_length":0}}`)
				return
			}
			w.WriteHeader(http.StatusNotFound)
		}))
		defer srv.Close()

		u, _ := url.Parse(srv.URL)
		client := api.NewClient(u, srv.Client())

		cfg := createConfig(context.Background(), client, "test-model")

		if _, ok := cfg["contextWindow"]; ok {
			t.Error("contextWindow should not be set for zero value")
		}
	})
}

// Ensure Capability constants used in createConfig match expected values
func TestPiCapabilityConstants(t *testing.T) {
	if model.CapabilityVision != "vision" {
		t.Errorf("CapabilityVision = %q, want %q", model.CapabilityVision, "vision")
	}
	if model.CapabilityThinking != "thinking" {
		t.Errorf("CapabilityThinking = %q, want %q", model.CapabilityThinking, "thinking")
	}
}
