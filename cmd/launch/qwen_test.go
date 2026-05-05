package launch

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"

	"github.com/ollama/ollama/cmd/config"
)

func setQwenTestHome(t *testing.T, home string) {
	t.Helper()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
}

func TestQwenConfigure(t *testing.T) {
	tmpDir := t.TempDir()
	setQwenTestHome(t, tmpDir)

	q := &Qwen{}
	if err := q.Configure("gemma4"); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	data, err := os.ReadFile(filepath.Join(tmpDir, ".qwen", "settings.json"))
	if err != nil {
		t.Fatalf("failed to read config: %v", err)
	}

	var cfg map[string]any
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("failed to parse config: %v", err)
	}

	envCfg := cfg["env"].(map[string]any)
	if envCfg[qwenOllamaEnvKey] != "ollama" {
		t.Fatalf("expected env[%q] to be ollama, got %v", qwenOllamaEnvKey, envCfg[qwenOllamaEnvKey])
	}

	modelCfg := cfg["model"].(map[string]any)
	if modelCfg["name"] != "gemma4" {
		t.Fatalf("expected model.name gemma4, got %v", modelCfg["name"])
	}

	security := cfg["security"].(map[string]any)
	auth := security["auth"].(map[string]any)
	if auth["selectedType"] != "openai" {
		t.Fatalf("expected auth.selectedType openai, got %v", auth["selectedType"])
	}
	if auth["baseUrl"] != qwenBaseURL() {
		t.Fatalf("expected auth.baseUrl %q, got %v", qwenBaseURL(), auth["baseUrl"])
	}

	modelProviders := cfg["modelProviders"].(map[string]any)
	openai := modelProviders["openai"].([]any)
	if len(openai) != 1 {
		t.Fatalf("expected one openai provider, got %d", len(openai))
	}

	provider := openai[0].(map[string]any)
	if provider["id"] != "gemma4" {
		t.Fatalf("expected provider id gemma4, got %v", provider["id"])
	}
	if provider["name"] != "gemma4 (Ollama)" {
		t.Fatalf("expected provider name %q, got %v", "gemma4 (Ollama)", provider["name"])
	}
	if provider["baseUrl"] != qwenBaseURL() {
		t.Fatalf("expected provider baseUrl %q, got %v", qwenBaseURL(), provider["baseUrl"])
	}
	if provider["envKey"] != qwenOllamaEnvKey {
		t.Fatalf("expected provider envKey %q, got %v", qwenOllamaEnvKey, provider["envKey"])
	}
}

func TestQwenCurrentModel(t *testing.T) {
	tmpDir := t.TempDir()
	setQwenTestHome(t, tmpDir)

	q := &Qwen{}
	if got := q.CurrentModel(); got != "" {
		t.Fatalf("expected empty model without config, got %q", got)
	}

	configDir := filepath.Join(tmpDir, ".qwen")
	if err := os.MkdirAll(configDir, 0o755); err != nil {
		t.Fatalf("failed to create config dir: %v", err)
	}

	configPath := filepath.Join(configDir, "settings.json")
	if err := os.WriteFile(configPath, []byte(`{"model":{"name":"llama3.2"}}`), 0o644); err != nil {
		t.Fatalf("failed to write config: %v", err)
	}

	if got := q.CurrentModel(); got != "llama3.2" {
		t.Fatalf("expected current model llama3.2, got %q", got)
	}
}

func TestQwenCurrentModelFallsBackToProvider(t *testing.T) {
	tmpDir := t.TempDir()
	setQwenTestHome(t, tmpDir)

	configDir := filepath.Join(tmpDir, ".qwen")
	if err := os.MkdirAll(configDir, 0o755); err != nil {
		t.Fatalf("failed to create config dir: %v", err)
	}

	configPath := filepath.Join(configDir, "settings.json")
	if err := os.WriteFile(configPath, []byte(`{"modelProviders":{"openai":[{"id":"mistral"}]}}`), 0o644); err != nil {
		t.Fatalf("failed to write config: %v", err)
	}

	if got := (&Qwen{}).CurrentModel(); got != "mistral" {
		t.Fatalf("expected provider fallback mistral, got %q", got)
	}
}

func TestQwenOnboard(t *testing.T) {
	tmpDir := t.TempDir()
	setQwenTestHome(t, tmpDir)

	if err := (&Qwen{}).Onboard(); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	saved, err := config.LoadIntegration("qwen")
	if err != nil {
		t.Fatalf("failed to load integration config: %v", err)
	}
	if !saved.Onboarded {
		t.Fatal("expected qwen integration to be marked onboarded")
	}
}

func TestQwenIntegration(t *testing.T) {
	q := &Qwen{}

	t.Run("String", func(t *testing.T) {
		if got := q.String(); got != "Qwen Code" {
			t.Fatalf("String() = %q, want %q", got, "Qwen Code")
		}
	})

	t.Run("implements Runner", func(t *testing.T) {
		var _ Runner = q
	})

	t.Run("implements ManagedSingleModel", func(t *testing.T) {
		var _ ManagedSingleModel = q
	})

	t.Run("implements ManagedInteractiveOnboarding", func(t *testing.T) {
		var _ ManagedInteractiveOnboarding = q
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

func TestQwenPaths(t *testing.T) {
	testDir := filepath.Join(t.TempDir(), "qwen-paths-test")
	setQwenTestHome(t, testDir)

	q := &Qwen{}
	os.MkdirAll(filepath.Join(testDir, ".qwen"), 0o755)
	os.WriteFile(filepath.Join(testDir, ".qwen", "settings.json"), []byte("{}"), 0o644)

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
		t.Fatalf("expected user config path %s, got %s", want, got)
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
	if !slices.Contains(env, "OPENAI_API_KEY=ollama") {
		t.Fatalf("expected OPENAI_API_KEY override, got %v", env)
	}
	if !slices.Contains(env, "OPENAI_BASE_URL="+qwenBaseURL()) {
		t.Fatalf("expected OPENAI_BASE_URL override, got %v", env)
	}
	if !slices.Contains(env, "OPENAI_MODEL=llama3.2") {
		t.Fatalf("expected OPENAI_MODEL override, got %v", env)
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
}

func TestQwenRunDoesNotRewriteConfig(t *testing.T) {
	tmpDir := t.TempDir()
	setQwenTestHome(t, tmpDir)
	t.Chdir(tmpDir)

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

	initialConfig := []byte(`{"model":{"name":"qwen3:32b"}}`)
	configPath := filepath.Join(configDir, "settings.json")
	if err := os.WriteFile(configPath, initialConfig, 0o644); err != nil {
		t.Fatalf("failed to write initial config: %v", err)
	}

	if err := (&Qwen{}).Run("qwen3:32b", nil); err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("failed to read config after run: %v", err)
	}
	if string(data) != string(initialConfig) {
		t.Fatalf("expected run not to rewrite config, got %s", string(data))
	}
}
