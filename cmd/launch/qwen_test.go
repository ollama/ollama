package launch

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestQwenEdit(t *testing.T) {
	tmpDir := t.TempDir()
	origWd, _ := os.Getwd()
	defer os.Chdir(origWd)
	os.Chdir(tmpDir)

	q := &Qwen{}

	err := q.Edit([]string{"test-model"})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	configPath := filepath.Join(tmpDir, ".qwen", "settings.json")
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		t.Fatalf("expected config file to be created at %s", configPath)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("failed to read config: %v", err)
	}

	var cfg map[string]any
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("failed to parse json: %v", err)
	}

	modelProviders, ok := cfg["modelProviders"].(map[string]any)
	if !ok {
		t.Fatalf("missing modelProviders")
	}
	openaiArray, ok := modelProviders["openai"].([]any)
	if !ok || len(openaiArray) == 0 {
		t.Fatalf("missing modelProviders.openai array")
	}

	provider := openaiArray[0].(map[string]any)
	if provider["id"] != "test-model" {
		t.Errorf("expected id 'test-model', got %v", provider["id"])
	}
	if provider["envKey"] != "OPENAI_API_KEY" {
		t.Errorf("expected envKey 'OPENAI_API_KEY', got %v", provider["envKey"])
	}

	modelBlock, ok := cfg["model"].(map[string]any)
	if !ok || modelBlock["name"] != "test-model" {
		t.Errorf("expected model.name 'test-model', got %v", modelBlock["name"])
	}

	sec, ok := cfg["security"].(map[string]any)
	if !ok {
		t.Fatalf("missing security")
	}
	auth, ok := sec["auth"].(map[string]any)
	if !ok || auth["selectedType"] != "openai" {
		t.Errorf("expected auth.selectedType 'openai', got %v", auth["selectedType"])
	}
}

func TestQwenEditPreservesAuth(t *testing.T) {
	tmpDir := t.TempDir()
	origWd, _ := os.Getwd()
	defer os.Chdir(origWd)
	os.Chdir(tmpDir)

	configDir := filepath.Join(tmpDir, ".qwen")
	os.MkdirAll(configDir, 0o755)
	initialConfig := []byte(`{
		"security": {
			"auth": {
				"selectedType": "custom_auth"
			}
		}
	}`)
	os.WriteFile(filepath.Join(configDir, "settings.json"), initialConfig, 0o644)

	q := &Qwen{}
	err := q.Edit([]string{"test-model"})
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	data, _ := os.ReadFile(filepath.Join(configDir, "settings.json"))
	var cfg map[string]any
	json.Unmarshal(data, &cfg)

	sec := cfg["security"].(map[string]any)
	auth := sec["auth"].(map[string]any)
	if auth["selectedType"] != "custom_auth" {
		t.Errorf("expected auth.selectedType to be preserved as 'custom_auth', got %v", auth["selectedType"])
	}
}

func TestQwenLegacyMigration(t *testing.T) {
	tmpDir := t.TempDir()
	origWd, _ := os.Getwd()
	defer os.Chdir(origWd)
	os.Chdir(tmpDir)

	configDir := filepath.Join(tmpDir, ".qwen")
	os.MkdirAll(configDir, 0o755)
	initialConfig := []byte(`{
		"modelProviders": {
			"openai": {
				"id": "legacy",
				"baseUrl": "http://legacy"
			}
		}
	}`)
	os.WriteFile(filepath.Join(configDir, "settings.json"), initialConfig, 0o644)

	q := &Qwen{}
	q.Edit([]string{"test-model"})

	data, _ := os.ReadFile(filepath.Join(configDir, "settings.json"))
	var cfg map[string]any
	json.Unmarshal(data, &cfg)

	modelProviders := cfg["modelProviders"].(map[string]any)
	openaiArray, ok := modelProviders["openai"].([]any)
	if !ok || len(openaiArray) != 2 {
		t.Fatalf("expected openai to be migrated to an array of length 2, got %v", openaiArray)
	}
}
