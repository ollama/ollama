package launch

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/envconfig"
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

func TestQwenEditForcesSelectedType(t *testing.T) {
	tmpDir := t.TempDir()
	origWd, _ := os.Getwd()
	defer os.Chdir(origWd)
	os.Chdir(tmpDir)

	configDir := filepath.Join(tmpDir, ".qwen")
	os.MkdirAll(configDir, 0o755)

	// Test 1: selectedType is forced to "openai" even if another type exists
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
	if auth["selectedType"] != "openai" {
		t.Errorf("expected auth.selectedType to be 'openai', got %v", auth["selectedType"])
	}

	// Test 2: selectedType is set to "openai" when not present
	os.Remove(filepath.Join(configDir, "settings.json"))
	q.Edit([]string{"test-model2"})

	data, _ = os.ReadFile(filepath.Join(configDir, "settings.json"))
	json.Unmarshal(data, &cfg)

	sec = cfg["security"].(map[string]any)
	auth = sec["auth"].(map[string]any)
	if auth["selectedType"] != "openai" {
		t.Errorf("expected auth.selectedType to be 'openai', got %v", auth["selectedType"])
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

func TestQwenIntegration(t *testing.T) {
	q := &Qwen{}

	t.Run("String", func(t *testing.T) {
		if got := q.String(); got != "Qwen Code CLI" {
			t.Errorf("String() = %q, want %q", got, "Qwen Code CLI")
		}
	})

	t.Run("implements Runner", func(t *testing.T) {
		var _ Runner = q
	})

	t.Run("implements Editor", func(t *testing.T) {
		var _ Editor = q
	})
}

func TestQwenFindPath(t *testing.T) {
	q := &Qwen{}
	path, err := q.findPath()
	if err != nil {
		t.Skipf("qwen binary not found, skipping: %v", err)
	}
	if path == "" {
		t.Fatal("expected non-empty path")
	}
}

func TestQwenModels(t *testing.T) {
	tmpDir := t.TempDir()
	origWd, _ := os.Getwd()
	defer os.Chdir(origWd)
	os.Chdir(tmpDir)

	// Without config, Models() should return nil
	q := &Qwen{}
	if models := q.Models(); models != nil {
		t.Errorf("expected nil models, got %v", models)
	}

	// With config, should return model name
	configDir := filepath.Join(tmpDir, ".qwen")
	os.MkdirAll(configDir, 0o755)
	config := map[string]any{
		"model": map[string]any{"name": "test-model"},
	}
	data, _ := json.Marshal(config)
	os.WriteFile(filepath.Join(configDir, "settings.json"), data, 0o644)

	models := q.Models()
	if len(models) != 1 || models[0] != "test-model" {
		t.Errorf("expected [test-model], got %v", models)
	}
}

func TestQwenPaths(t *testing.T) {
	// Test that Paths() returns the project config path when it exists.
	testDir := filepath.Join(t.TempDir(), "qwen-paths-test")
	os.MkdirAll(testDir, 0755)
	origWd, _ := os.Getwd()
	defer os.Chdir(origWd)
	os.Chdir(testDir)

	q := &Qwen{}
	// With config file, Paths() should return the project path
	os.MkdirAll(filepath.Join(testDir, ".qwen"), 0755)
	os.WriteFile(filepath.Join(testDir, ".qwen", "settings.json"), []byte("{}"), 0644)

	paths := q.Paths()
	if len(paths) != 1 {
		t.Fatalf("expected 1 path, got %v", paths)
	}
	if paths[0] != filepath.Join(testDir, ".qwen", "settings.json") {
		t.Errorf("expected project config path, got %s", paths[0])
	}
}

func TestQwenBuildEnvForcesVars(t *testing.T) {
	q := &Qwen{}

	// Set hostile parent env values
	t.Setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
	t.Setenv("OPENAI_API_KEY", "real-key")
	t.Setenv("OPENAI_MODEL", "gpt-4")

	env := q.buildEnv("test-model")

	// Check that OPENAI_API_KEY is forced to dummy
	found := false
	for _, e := range env {
		if e == "OPENAI_API_KEY=dummy" {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected OPENAI_API_KEY=dummy in env")
	}

	// Check that OPENAI_BASE_URL is forced to Ollama URL
	host := strings.TrimRight(envconfig.Host().String(), "/")
	expectedBase := host + "/v1"
	found = false
	for _, e := range env {
		if e == "OPENAI_BASE_URL="+expectedBase {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected OPENAI_BASE_URL=%s in env", expectedBase)
	}

	// Check that OPENAI_MODEL is forced to test-model
	found = false
	for _, e := range env {
		if e == "OPENAI_MODEL=test-model" {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected OPENAI_MODEL=test-model in env")
	}

	// Check that hostile parent values are not present
	for _, e := range env {
		switch e {
		case "OPENAI_BASE_URL=https://api.openai.com/v1":
			t.Error("hostile parent OPENAI_BASE_URL should not be present")
		case "OPENAI_API_KEY=real-key":
			t.Error("hostile parent OPENAI_API_KEY should not be present")
		case "OPENAI_MODEL=gpt-4":
			t.Error("hostile parent OPENAI_MODEL should not be present")
		}
	}
}

func TestQwenConfigPathPrecedence(t *testing.T) {
	q := &Qwen{}

	// Test 1: Existing project config takes precedence
	t.Run("existing project config", func(t *testing.T) {
		tmpDir := t.TempDir()
		origWd, _ := os.Getwd()
		defer os.Chdir(origWd)
		os.Chdir(tmpDir)

		// Create project config
		os.MkdirAll(filepath.Join(tmpDir, ".qwen"), 0755)
		os.WriteFile(filepath.Join(tmpDir, ".qwen", "settings.json"), []byte("{}"), 0644)

		path, err := q.configPath()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expected := filepath.Join(tmpDir, ".qwen", "settings.json")
		if path != expected {
			t.Errorf("expected %s, got %s", expected, path)
		}
	})

	// Test 2: Existing user config is used if no project config
	t.Run("existing user config", func(t *testing.T) {
		tmpDir := t.TempDir()
		origWd, _ := os.Getwd()
		defer os.Chdir(origWd)
		os.Chdir(tmpDir)

		// Create user config in fake home
		fakeHome := filepath.Join(tmpDir, "fake-home")
		t.Setenv("HOME", fakeHome)

		userConfigDir := filepath.Join(fakeHome, ".qwen")
		os.MkdirAll(userConfigDir, 0755)
		os.WriteFile(filepath.Join(userConfigDir, "settings.json"), []byte("{}"), 0644)

		path, err := q.configPath()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expected := filepath.Join(fakeHome, ".qwen", "settings.json")
		if path != expected {
			t.Errorf("expected %s, got %s", expected, path)
		}
	})

	// Test 3: New project config is created if neither exists
	t.Run("new project config", func(t *testing.T) {
		tmpDir := t.TempDir()
		origWd, _ := os.Getwd()
		defer os.Chdir(origWd)
		os.Chdir(tmpDir)

		// Set HOME to a directory without .qwen
		fakeHome := filepath.Join(tmpDir, "fake-home-no-config")
		t.Setenv("HOME", fakeHome)

		path, err := q.configPath()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		expected := filepath.Join(tmpDir, ".qwen", "settings.json")
		if path != expected {
			t.Errorf("expected %s, got %s", expected, path)
		}
	})
}
