package launch

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"
)

func setQwenTestHome(t *testing.T, home string) {
	t.Helper()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
}

func TestQwenEdit(t *testing.T) {
	tmpDir := t.TempDir()
	setQwenTestHome(t, tmpDir)

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
	if provider["envKey"] != qwenAPIKeyEnvKey() {
		t.Errorf("expected envKey %q, got %v", qwenAPIKeyEnvKey(), provider["envKey"])
	}
	if provider["name"] != "test-model (Ollama)" {
		t.Errorf("expected name %q, got %v", "test-model (Ollama)", provider["name"])
	}
	if _, ok := provider["description"]; ok {
		t.Errorf("did not expect description for non-recommended model, got %v", provider["description"])
	}

	envCfg, ok := cfg["env"].(map[string]any)
	if !ok {
		t.Fatalf("missing env")
	}
	if envCfg[qwenAPIKeyEnvKey()] != "ollama" {
		t.Errorf("expected env[%q] to be %q, got %v", qwenAPIKeyEnvKey(), "ollama", envCfg[qwenAPIKeyEnvKey()])
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
	if auth["baseUrl"] != qwenBaseURL() {
		t.Errorf("expected auth.baseUrl %q, got %v", qwenBaseURL(), auth["baseUrl"])
	}
}

func TestQwenEditAddsAllSelectedModels(t *testing.T) {
	tmpDir := t.TempDir()
	setQwenTestHome(t, tmpDir)

	q := &Qwen{}
	if err := q.Edit([]string{"qwen3:32b", "qwen3:14b"}); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	data, err := os.ReadFile(filepath.Join(tmpDir, ".qwen", "settings.json"))
	if err != nil {
		t.Fatalf("failed to read config: %v", err)
	}

	var cfg map[string]any
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("failed to parse json: %v", err)
	}

	modelProviders := cfg["modelProviders"].(map[string]any)
	openaiArray := modelProviders["openai"].([]any)
	if len(openaiArray) != 2 {
		t.Fatalf("expected 2 openai providers, got %d", len(openaiArray))
	}

	var gotIDs []string
	for _, item := range openaiArray {
		provider := item.(map[string]any)
		gotIDs = append(gotIDs, provider["id"].(string))
		if provider["envKey"] != qwenAPIKeyEnvKey() {
			t.Fatalf("expected envKey %q, got %v", qwenAPIKeyEnvKey(), provider["envKey"])
		}
	}

	if !slices.Equal(gotIDs, []string{"qwen3:32b", "qwen3:14b"}) {
		t.Fatalf("expected providers %v, got %v", []string{"qwen3:32b", "qwen3:14b"}, gotIDs)
	}
}

func TestQwenAPIKeyEnvKey(t *testing.T) {
	if got := qwenAPIKeyEnvKey(); got != "OLLAMA_API_KEY" {
		t.Fatalf("expected OLLAMA_API_KEY, got %q", got)
	}
}

func TestQwenEditUsesRecommendedModelDescriptionWhenAvailable(t *testing.T) {
	tmpDir := t.TempDir()
	setQwenTestHome(t, tmpDir)

	q := &Qwen{}
	if err := q.Edit([]string{"gemma4"}); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	data, err := os.ReadFile(filepath.Join(tmpDir, ".qwen", "settings.json"))
	if err != nil {
		t.Fatalf("failed to read config: %v", err)
	}

	var cfg map[string]any
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("failed to parse json: %v", err)
	}

	modelProviders := cfg["modelProviders"].(map[string]any)
	openaiArray := modelProviders["openai"].([]any)
	provider := openaiArray[0].(map[string]any)
	if provider["description"] != "Reasoning and code generation locally" {
		t.Fatalf("expected recommended description %q, got %v", "Reasoning and code generation locally", provider["description"])
	}
}

func TestQwenEditForcesOpenAIAuth(t *testing.T) {
	tmpDir := t.TempDir()
	setQwenTestHome(t, tmpDir)

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
	if auth["selectedType"] != "openai" {
		t.Errorf("expected auth.selectedType to be rewritten to 'openai', got %v", auth["selectedType"])
	}
}

func TestQwenLegacyMigration(t *testing.T) {
	tmpDir := t.TempDir()
	setQwenTestHome(t, tmpDir)

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

func TestQwenEditRewritesLegacyEnvKey(t *testing.T) {
	tmpDir := t.TempDir()
	setQwenTestHome(t, tmpDir)

	configDir := filepath.Join(tmpDir, ".qwen")
	if err := os.MkdirAll(configDir, 0o755); err != nil {
		t.Fatalf("failed to create config dir: %v", err)
	}
	initialConfig := []byte(`{
		"modelProviders": {
			"openai": [
				{
					"id": "test-model",
					"envKey": "OPENAI_API_KEY",
					"baseUrl": "http://legacy"
				}
			]
		}
	}`)
	if err := os.WriteFile(filepath.Join(configDir, "settings.json"), initialConfig, 0o644); err != nil {
		t.Fatalf("failed to write initial config: %v", err)
	}

	q := &Qwen{}
	if err := q.Edit([]string{"test-model"}); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	data, err := os.ReadFile(filepath.Join(configDir, "settings.json"))
	if err != nil {
		t.Fatalf("failed to read config: %v", err)
	}

	var cfg map[string]any
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("failed to parse json: %v", err)
	}

	modelProviders := cfg["modelProviders"].(map[string]any)
	openaiArray := modelProviders["openai"].([]any)
	provider := openaiArray[0].(map[string]any)
	if provider["envKey"] != qwenAPIKeyEnvKey() {
		t.Fatalf("expected envKey %q, got %v", qwenAPIKeyEnvKey(), provider["envKey"])
	}
}

func TestQwenEditNormalizesLoopbackOllamaProviders(t *testing.T) {
	tmpDir := t.TempDir()
	setQwenTestHome(t, tmpDir)

	configDir := filepath.Join(tmpDir, ".qwen")
	if err := os.MkdirAll(configDir, 0o755); err != nil {
		t.Fatalf("failed to create config dir: %v", err)
	}
	initialConfig := []byte(`{
		"modelProviders": {
			"openai": [
				{
					"id": "llama3.2",
					"baseUrl": "http://127.0.0.1:11434/v1"
				}
			]
		},
		"model": {
			"name": "llama3.2"
		}
	}`)
	if err := os.WriteFile(filepath.Join(configDir, "settings.json"), initialConfig, 0o644); err != nil {
		t.Fatalf("failed to write initial config: %v", err)
	}

	q := &Qwen{}
	if err := q.Edit([]string{"llama3.2"}); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	data, err := os.ReadFile(filepath.Join(configDir, "settings.json"))
	if err != nil {
		t.Fatalf("failed to read config: %v", err)
	}

	var cfg map[string]any
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("failed to parse json: %v", err)
	}

	modelProviders := cfg["modelProviders"].(map[string]any)
	openaiArray := modelProviders["openai"].([]any)
	provider := openaiArray[0].(map[string]any)
	if provider["baseUrl"] != qwenBaseURL() {
		t.Fatalf("expected baseUrl %q, got %v", qwenBaseURL(), provider["baseUrl"])
	}
	if provider["envKey"] != qwenAPIKeyEnvKey() {
		t.Fatalf("expected envKey %q, got %v", qwenAPIKeyEnvKey(), provider["envKey"])
	}
}

func TestQwenEditDropsStaleOllamaProvidersButKeepsRemoteProviders(t *testing.T) {
	tmpDir := t.TempDir()
	setQwenTestHome(t, tmpDir)

	configDir := filepath.Join(tmpDir, ".qwen")
	if err := os.MkdirAll(configDir, 0o755); err != nil {
		t.Fatalf("failed to create config dir: %v", err)
	}
	initialConfig := []byte(`{
		"modelProviders": {
			"openai": [
				{
					"id": "llama3.2",
					"baseUrl": "http://127.0.0.1:11434/v1"
				},
				{
					"id": "gpt-4.1",
					"baseUrl": "https://api.openai.com/v1",
					"envKey": "OPENAI_API_KEY"
				}
			]
		},
		"model": {
			"name": "gemma4"
		}
	}`)
	if err := os.WriteFile(filepath.Join(configDir, "settings.json"), initialConfig, 0o644); err != nil {
		t.Fatalf("failed to write initial config: %v", err)
	}

	q := &Qwen{}
	if err := q.Edit([]string{"gemma4"}); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	data, err := os.ReadFile(filepath.Join(configDir, "settings.json"))
	if err != nil {
		t.Fatalf("failed to read config: %v", err)
	}

	var cfg map[string]any
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("failed to parse json: %v", err)
	}

	modelProviders := cfg["modelProviders"].(map[string]any)
	openaiArray := modelProviders["openai"].([]any)
	if len(openaiArray) != 2 {
		t.Fatalf("expected 2 providers, got %d", len(openaiArray))
	}

	first := openaiArray[0].(map[string]any)
	if first["id"] != "gemma4" {
		t.Fatalf("expected first provider to be selected ollama model, got %v", first["id"])
	}
	if first["envKey"] != qwenAPIKeyEnvKey() {
		t.Fatalf("expected first provider envKey %q, got %v", qwenAPIKeyEnvKey(), first["envKey"])
	}

	second := openaiArray[1].(map[string]any)
	if second["id"] != "gpt-4.1" {
		t.Fatalf("expected remote provider to be preserved, got %v", second["id"])
	}
	if second["envKey"] != "OPENAI_API_KEY" {
		t.Fatalf("expected remote provider envKey to be preserved, got %v", second["envKey"])
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
	setQwenTestHome(t, tmpDir)

	// Without config, Models() should return nil
	q := &Qwen{}
	if models := q.Models(); models != nil {
		t.Errorf("expected nil models, got %v", models)
	}

	// With config, should return model name
	configDir := filepath.Join(tmpDir, ".qwen")
	os.MkdirAll(configDir, 0o755)
	config := map[string]any{
		"modelProviders": map[string]any{
			"openai": []any{
				map[string]any{
					"id":      "second-model",
					"baseUrl": qwenBaseURL(),
				},
				map[string]any{
					"id":      "remote-model",
					"baseUrl": "https://api.example.com/v1",
				},
			},
		},
		"model": map[string]any{"name": "test-model"},
	}
	data, _ := json.Marshal(config)
	os.WriteFile(filepath.Join(configDir, "settings.json"), data, 0o644)

	models := q.Models()
	if !slices.Equal(models, []string{"test-model", "second-model"}) {
		t.Errorf("expected [test-model second-model], got %v", models)
	}
}

func TestQwenPaths(t *testing.T) {
	testDir := filepath.Join(t.TempDir(), "qwen-paths-test")
	setQwenTestHome(t, testDir)

	q := &Qwen{}
	os.MkdirAll(filepath.Join(testDir, ".qwen"), 0755)
	os.WriteFile(filepath.Join(testDir, ".qwen", "settings.json"), []byte("{}"), 0644)

	paths := q.Paths()
	if len(paths) != 1 {
		t.Fatalf("expected 1 path, got %v", paths)
	}
	want, err := filepath.EvalSymlinks(filepath.Join(testDir, ".qwen", "settings.json"))
	if err != nil {
		t.Fatalf("failed to resolve expected path: %v", err)
	}
	got, err := filepath.EvalSymlinks(paths[0])
	if err != nil {
		t.Fatalf("failed to resolve returned path: %v", err)
	}
	if got != want {
		t.Errorf("expected user config path %s, got %s", want, got)
	}
}

func TestQwenIsOllamaProviderBaseURL(t *testing.T) {
	if !qwenIsOllamaProviderBaseURL("http://127.0.0.1:11434/v1") {
		t.Fatal("expected loopback localhost variant to be treated as Ollama")
	}
	if qwenIsOllamaProviderBaseURL("https://api.openai.com/v1") {
		t.Fatal("did not expect remote OpenAI base URL to be treated as Ollama")
	}
}

func TestQwenLaunchArgs(t *testing.T) {
	got := qwenLaunchArgs("llama3.2", nil)
	want := []string{"--model", "llama3.2", "--auth-type", "openai"}
	if !slices.Equal(got, want) {
		t.Fatalf("expected %v, got %v", want, got)
	}

	got = qwenLaunchArgs("llama3.2", []string{"--auth-type", "openai"})
	want = []string{"--model", "llama3.2", "--auth-type", "openai"}
	if !slices.Equal(got, want) {
		t.Fatalf("expected %v, got %v", want, got)
	}

	got = qwenLaunchArgs("llama3.2", []string{"-m", "gemma4"})
	want = []string{"--auth-type", "openai", "-m", "gemma4"}
	if !slices.Equal(got, want) {
		t.Fatalf("expected %v, got %v", want, got)
	}
}

func TestQwenLaunchEnv(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OPENAI_BASE_URL", "")
	t.Setenv("OPENAI_MODEL", "")

	env := qwenLaunchEnv("llama3.2")
	joined := slices.Clone(env)

	if !slices.Contains(joined, "OPENAI_API_KEY=ollama") {
		t.Fatalf("expected OPENAI_API_KEY fallback, got %v", joined)
	}
	if !slices.Contains(joined, "OPENAI_BASE_URL="+qwenBaseURL()) {
		t.Fatalf("expected OPENAI_BASE_URL fallback, got %v", joined)
	}
	if !slices.Contains(joined, "OPENAI_MODEL=llama3.2") {
		t.Fatalf("expected OPENAI_MODEL fallback, got %v", joined)
	}
}

func TestQwenLaunchEnvOverridesExistingOpenAIEnv(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "real-key")
	t.Setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
	t.Setenv("OPENAI_MODEL", "gpt-4.1")

	env := qwenLaunchEnv("llama3.2")
	if !slices.Contains(env, "OPENAI_API_KEY=ollama") {
		t.Fatalf("expected OPENAI_API_KEY override, got %v", env)
	}
	if !slices.Contains(env, "OPENAI_BASE_URL="+qwenBaseURL()) {
		t.Fatalf("expected OPENAI_BASE_URL override, got %v", env)
	}
	if !slices.Contains(env, "OPENAI_MODEL=llama3.2") {
		t.Fatalf("expected OPENAI_MODEL override, got %v", env)
	}
	for _, entry := range env {
		if entry == "OPENAI_API_KEY=real-key" || entry == "OPENAI_BASE_URL=https://api.openai.com/v1" || entry == "OPENAI_MODEL=gpt-4.1" {
			t.Fatalf("did not expect stale OpenAI env entry %q to survive", entry)
		}
	}
}

func TestQwenRunDoesNotRewriteMultiModelConfig(t *testing.T) {
	tmpDir := t.TempDir()
	setQwenTestHome(t, tmpDir)
	origWd, _ := os.Getwd()
	defer os.Chdir(origWd)
	if err := os.Chdir(tmpDir); err != nil {
		t.Fatalf("failed to chdir: %v", err)
	}

	qwenBinDir := filepath.Join(tmpDir, "bin")
	if err := os.MkdirAll(qwenBinDir, 0o755); err != nil {
		t.Fatalf("failed to create bin dir: %v", err)
	}

	qwenBin := filepath.Join(qwenBinDir, "qwen")
	qwenScript := "#!/bin/sh\nexit 0\n"
	if runtime.GOOS == "windows" {
		qwenBin = filepath.Join(qwenBinDir, "qwen.bat")
		qwenScript = "@echo off\r\nexit /b 0\r\n"
	}
	if err := os.WriteFile(qwenBin, []byte(qwenScript), 0o755); err != nil {
		t.Fatalf("failed to write fake qwen binary: %v", err)
	}
	if runtime.GOOS != "windows" {
		if err := os.Chmod(qwenBin, 0o755); err != nil {
			t.Fatalf("failed to chmod fake qwen binary: %v", err)
		}
	}

	t.Setenv("PATH", qwenBinDir+string(os.PathListSeparator)+os.Getenv("PATH"))

	configDir := filepath.Join(tmpDir, ".qwen")
	if err := os.MkdirAll(configDir, 0o755); err != nil {
		t.Fatalf("failed to create config dir: %v", err)
	}

	initialConfig := map[string]any{
		"env": map[string]any{
			qwenAPIKeyEnvKey(): "ollama",
		},
		"modelProviders": map[string]any{
			"openai": []any{
				map[string]any{
					"id":      "qwen3:32b",
					"baseUrl": qwenBaseURL(),
					"envKey":  qwenAPIKeyEnvKey(),
				},
				map[string]any{
					"id":      "qwen3:14b",
					"baseUrl": qwenBaseURL(),
					"envKey":  qwenAPIKeyEnvKey(),
				},
			},
		},
		"security": map[string]any{
			"auth": map[string]any{
				"selectedType": "openai",
			},
		},
		"model": map[string]any{
			"name": "qwen3:32b",
		},
	}
	data, err := json.Marshal(initialConfig)
	if err != nil {
		t.Fatalf("failed to marshal initial config: %v", err)
	}
	configPath := filepath.Join(configDir, "settings.json")
	if err := os.WriteFile(configPath, data, 0o644); err != nil {
		t.Fatalf("failed to write initial config: %v", err)
	}

	q := &Qwen{}
	if err := q.Run("qwen3:32b", nil); err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	data, err = os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("failed to read config after run: %v", err)
	}

	var cfg map[string]any
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("failed to parse config after run: %v", err)
	}

	modelProviders := cfg["modelProviders"].(map[string]any)
	openaiArray := modelProviders["openai"].([]any)
	if len(openaiArray) != 2 {
		t.Fatalf("expected both Ollama providers to remain after launch, got %d", len(openaiArray))
	}
}
