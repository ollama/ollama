package launch

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"

	modelpkg "github.com/ollama/ollama/types/model"
	"gopkg.in/yaml.v3"
)

func TestMain(m *testing.M) {
	if os.Getenv("OLLAMA_LAUNCH_OMP_TEST_HELPER") == "1" {
		runOMPTestHelper()
		return
	}
	os.Exit(m.Run())
}

func runOMPTestHelper() {
	logPath := os.Getenv("OLLAMA_LAUNCH_OMP_TEST_LOG")
	if logPath != "" {
		f, err := os.OpenFile(logPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
		if err == nil {
			_, _ = fmt.Fprintln(f, strings.Join(os.Args[1:], " "))
			_ = f.Close()
		}
	}

	if len(os.Args) >= 3 && os.Args[1] == "plugin" && os.Args[2] == "list" {
		fmt.Print(os.Getenv("OLLAMA_LAUNCH_OMP_TEST_PLUGIN_LIST"))
		os.Exit(0)
	}
	if len(os.Args) >= 4 && os.Args[1] == "plugin" && os.Args[2] == "install" {
		if os.Getenv("OLLAMA_LAUNCH_OMP_TEST_FAIL_INSTALL") == "1" {
			_, _ = fmt.Fprintln(os.Stderr, "install failed")
			os.Exit(1)
		}
		os.Exit(0)
	}
	os.Exit(0)
}

func setOMPTestHome(t *testing.T, dir string) {
	t.Helper()
	setTestHome(t, dir)
	t.Setenv("PI_CONFIG_DIR", "")
	t.Setenv("PI_CODING_AGENT_DIR", "")
}

func TestOMPIntegration(t *testing.T) {
	o := &OMP{}

	t.Run("String", func(t *testing.T) {
		if got := o.String(); got != "OMP" {
			t.Errorf("String() = %q, want %q", got, "OMP")
		}
	})

	t.Run("implements Runner", func(t *testing.T) {
		var _ Runner = o
	})

	t.Run("implements ManagedSingleModel", func(t *testing.T) {
		var _ ManagedSingleModel = o
	})

	t.Run("implements ManagedModelListConfigurer", func(t *testing.T) {
		var _ ManagedModelListConfigurer = o
	})

	t.Run("does not require interactive onboarding", func(t *testing.T) {
		var _ ManagedInteractiveOnboarding = o
		if o.RequiresInteractiveOnboarding() {
			t.Fatal("OMP onboarding should not require an interactive terminal")
		}
	})
}

func TestOMPArgs(t *testing.T) {
	o := &OMP{}

	tests := []struct {
		name  string
		model string
		args  []string
		want  []string
	}{
		{"with model", "gemma4", nil, []string{"--model", "ollama/gemma4"}},
		{"with cloud model", "kimi-k2.6:cloud", nil, []string{"--model", "ollama/kimi-k2.6:cloud"}},
		{"empty model", "", nil, nil},
		{"with model and extra", "gemma4", []string{"--help"}, []string{"--model", "ollama/gemma4", "--help"}},
		{"already qualified", "ollama/gemma4", nil, []string{"--model", "ollama/gemma4"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := o.args(tt.model, tt.args)
			if !slices.Equal(got, tt.want) {
				t.Errorf("args(%q, %v) = %v, want %v", tt.model, tt.args, got, tt.want)
			}
		})
	}
}

func TestOMPRun_WebSearchPluginLifecycle(t *testing.T) {
	seedOMPHelperBinary := func(t *testing.T, dir string) {
		t.Helper()
		src, err := os.Executable()
		if err != nil {
			t.Fatal(err)
		}
		data, err := os.ReadFile(src)
		if err != nil {
			t.Fatal(err)
		}
		dst := filepath.Join(dir, ompExecutableNames()[0])
		if err := os.WriteFile(dst, data, 0o755); err != nil {
			t.Fatal(err)
		}
	}

	setCloudStatus := func(t *testing.T, disabled bool) {
		t.Helper()
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/status" {
				fmt.Fprintf(w, `{"cloud":{"disabled":%t,"source":"config"}}`, disabled)
				return
			}
			http.NotFound(w, r)
		}))
		t.Cleanup(srv.Close)
		t.Setenv("OLLAMA_HOST", srv.URL)
	}

	setup := func(t *testing.T, pluginList string, cloudDisabled bool) (string, *OMP) {
		t.Helper()
		tmpDir := t.TempDir()
		setOMPTestHome(t, tmpDir)
		t.Setenv("PATH", tmpDir)
		t.Setenv("OLLAMA_LAUNCH_OMP_TEST_HELPER", "1")
		t.Setenv("OLLAMA_LAUNCH_OMP_TEST_PLUGIN_LIST", pluginList)
		logPath := filepath.Join(tmpDir, "omp.log")
		t.Setenv("OLLAMA_LAUNCH_OMP_TEST_LOG", logPath)
		setCloudStatus(t, cloudDisabled)
		seedOMPHelperBinary(t, tmpDir)
		return logPath, &OMP{}
	}

	t.Run("web search missing installs before launch", func(t *testing.T) {
		logPath, o := setup(t, "No plugins installed\n", false)

		if err := o.Run("kimi-k2.6:cloud", nil, []string{"session"}); err != nil {
			t.Fatalf("Run() error = %v", err)
		}

		calls, err := os.ReadFile(logPath)
		if err != nil {
			t.Fatal(err)
		}
		got := string(calls)
		if !strings.Contains(got, "plugin list\n") {
			t.Fatalf("expected plugin list call, got:\n%s", got)
		}
		if !strings.Contains(got, "plugin install "+ompWebSearchPlugin+"\n") {
			t.Fatalf("expected plugin install call, got:\n%s", got)
		}
		if !strings.Contains(got, "--model ollama/kimi-k2.6:cloud session\n") {
			t.Fatalf("expected final omp launch call, got:\n%s", got)
		}
	})

	t.Run("web search present refreshes before launch", func(t *testing.T) {
		logPath, o := setup(t, "npm Plugins:\n\n● "+ompWebSearchPlugin+"@0.0.5\n", false)

		if err := o.Run("gemma4", nil, []string{"chat"}); err != nil {
			t.Fatalf("Run() error = %v", err)
		}

		calls, err := os.ReadFile(logPath)
		if err != nil {
			t.Fatal(err)
		}
		got := string(calls)
		if !strings.Contains(got, "plugin install "+ompWebSearchPlugin+"\n") {
			t.Fatalf("expected plugin refresh install call, got:\n%s", got)
		}
		if !strings.Contains(got, "--model ollama/gemma4 chat\n") {
			t.Fatalf("expected final omp launch call, got:\n%s", got)
		}
	})

	t.Run("web search install failure warns and continues", func(t *testing.T) {
		logPath, o := setup(t, "No plugins installed\n", false)
		t.Setenv("OLLAMA_LAUNCH_OMP_TEST_FAIL_INSTALL", "1")

		stderr := captureStderr(t, func() {
			if err := o.Run("gemma4", nil, []string{"chat"}); err != nil {
				t.Fatalf("Run() should continue after plugin install failure, got %v", err)
			}
		})
		if !strings.Contains(stderr, "Warning: could not install "+ompWebSearchPlugin) {
			t.Fatalf("expected install warning, got:\n%s", stderr)
		}

		calls, err := os.ReadFile(logPath)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(calls), "--model ollama/gemma4 chat\n") {
			t.Fatalf("expected final omp launch call, got:\n%s", calls)
		}
	})

	t.Run("cloud disabled skips web search plugin management", func(t *testing.T) {
		logPath, o := setup(t, "No plugins installed\n", true)

		stderr := captureStderr(t, func() {
			if err := o.Run("gemma4", nil, []string{"chat"}); err != nil {
				t.Fatalf("Run() error = %v", err)
			}
		})
		if !strings.Contains(stderr, "Cloud is disabled; skipping "+ompWebSearchPlugin+" setup.") {
			t.Fatalf("expected cloud-disabled skip message, got:\n%s", stderr)
		}

		calls, err := os.ReadFile(logPath)
		if err != nil {
			t.Fatal(err)
		}
		got := string(calls)
		if strings.Contains(got, "plugin list\n") || strings.Contains(got, "plugin install "+ompWebSearchPlugin+"\n") {
			t.Fatalf("did not expect plugin management calls, got:\n%s", got)
		}
		if !strings.Contains(got, "--model ollama/gemma4 chat\n") {
			t.Fatalf("expected final omp launch call, got:\n%s", got)
		}
	})
}

func TestOMPFindPath(t *testing.T) {
	o := &OMP{}

	t.Run("finds omp in PATH", func(t *testing.T) {
		tmpDir := t.TempDir()
		name := "omp"
		if runtime.GOOS == "windows" {
			name = "omp.exe"
		}
		fakeBin := filepath.Join(tmpDir, name)
		os.WriteFile(fakeBin, []byte("#!/bin/sh\n"), 0o755)
		t.Setenv("PATH", tmpDir)

		got, err := o.findPath()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != fakeBin {
			t.Errorf("findPath() = %q, want %q", got, fakeBin)
		}
	})

	t.Run("falls back to ~/.local/bin/omp", func(t *testing.T) {
		home := t.TempDir()
		setOMPTestHome(t, home)
		t.Setenv("PATH", t.TempDir())

		fallback := filepath.Join(home, ".local", "bin", ompExecutableNames()[0])
		os.MkdirAll(filepath.Dir(fallback), 0o755)
		os.WriteFile(fallback, []byte("#!/bin/sh\n"), 0o755)

		got, err := o.findPath()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != fallback {
			t.Errorf("findPath() = %q, want %q", got, fallback)
		}
	})

	t.Run("falls back to ~/.bun/bin/omp", func(t *testing.T) {
		home := t.TempDir()
		setOMPTestHome(t, home)
		t.Setenv("PATH", t.TempDir())

		fallback := filepath.Join(home, ".bun", "bin", ompExecutableNames()[0])
		os.MkdirAll(filepath.Dir(fallback), 0o755)
		os.WriteFile(fallback, []byte("#!/bin/sh\n"), 0o755)

		got, err := o.findPath()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != fallback {
			t.Errorf("findPath() = %q, want %q", got, fallback)
		}
	})

	t.Run("returns error when not found", func(t *testing.T) {
		home := t.TempDir()
		setOMPTestHome(t, home)
		t.Setenv("PATH", t.TempDir())

		if _, err := o.findPath(); err == nil {
			t.Fatal("expected error, got nil")
		}
	})
}

func TestOMPConfigureWithModelsWritesModelsYML(t *testing.T) {
	home := t.TempDir()
	setOMPTestHome(t, home)
	t.Setenv("OLLAMA_HOST", "http://0.0.0.0:11434")

	o := &OMP{}
	models := []LaunchModel{
		{
			Name:            "glm-5.1:cloud",
			ContextLength:   202_752,
			MaxOutputTokens: 131_072,
		},
		{
			Name:         "qwen3.6",
			Capabilities: []modelpkg.Capability{modelpkg.CapabilityVision},
		},
	}
	if err := o.ConfigureWithModels("glm-5.1:cloud", models); err != nil {
		t.Fatalf("ConfigureWithModels returned error: %v", err)
	}

	path := filepath.Join(home, ".omp", "agent", "models.yml")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read models.yml: %v", err)
	}

	cfg := parseOMPConfigYAML(t, data)
	provider := ompProviderFromYAML(t, cfg)
	if provider["baseUrl"] != "http://127.0.0.1:11434/v1" {
		t.Fatalf("baseUrl = %v, want connectable OpenAI-compatible host", provider["baseUrl"])
	}
	if provider["api"] != "openai-responses" {
		t.Fatalf("api = %v, want openai-responses", provider["api"])
	}
	if provider["auth"] != "none" {
		t.Fatalf("auth = %v, want none", provider["auth"])
	}
	discovery, _ := provider["discovery"].(map[string]any)
	if discovery["type"] != "ollama" {
		t.Fatalf("discovery = %v, want type ollama", discovery)
	}

	entries := ompModelEntriesFromYAML(t, provider)
	if len(entries) != 2 {
		t.Fatalf("models length = %d, want 2", len(entries))
	}
	if entries[0]["id"] != "glm-5.1:cloud" {
		t.Fatalf("first model id = %v, want primary first", entries[0]["id"])
	}
	if got := numericYAMLValue(entries[0]["contextWindow"]); got != 202_752 {
		t.Fatalf("contextWindow = %d, want 202752", got)
	}
	if got := numericYAMLValue(entries[0]["maxTokens"]); got != 131_072 {
		t.Fatalf("maxTokens = %d, want 131072", got)
	}
	if input := stringSliceYAMLValue(entries[1]["input"]); !slices.Equal(input, []string{"text", "image"}) {
		t.Fatalf("vision input = %v, want [text image]", input)
	}
	if got := o.CurrentModel(); got != "glm-5.1:cloud" {
		t.Fatalf("CurrentModel = %q, want glm-5.1:cloud", got)
	}

	configPath := filepath.Join(home, ".omp", "agent", "config.yml")
	configData, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("failed to read config.yml: %v", err)
	}
	config := parseOMPConfigYAML(t, configData)
	if got := numericYAMLValue(config["setupVersion"]); got != ompSetupVersion {
		t.Fatalf("setupVersion = %d, want %d", got, ompSetupVersion)
	}
	if paths := o.Paths(); !slices.Equal(paths, []string{path, configPath}) {
		t.Fatalf("Paths = %v, want [%s %s]", paths, path, configPath)
	}
}

func TestOMPConfigureWithModelsPreservesExistingConfig(t *testing.T) {
	home := t.TempDir()
	setOMPTestHome(t, home)

	modelsPath := filepath.Join(home, ".omp", "agent", "models.yml")
	if err := os.MkdirAll(filepath.Dir(modelsPath), 0o755); err != nil {
		t.Fatal(err)
	}
	existing := []byte(`
providers:
  anthropic:
    baseUrl: https://example.com/anthropic
  ollama:
    baseUrl: http://old-host:11434
    api: openai-responses
    auth: none
    models:
      - id: old-model
        name: Old Model
        customField: keep-me
`)
	if err := os.WriteFile(modelsPath, existing, 0o644); err != nil {
		t.Fatal(err)
	}

	configPath := filepath.Join(home, ".omp", "agent", "config.yml")
	existingConfig := []byte(`
lastChangelogVersion: 15.7.6
setupVersion: 0
theme: monochrome
`)
	if err := os.WriteFile(configPath, existingConfig, 0o644); err != nil {
		t.Fatal(err)
	}

	o := &OMP{}
	if err := o.ConfigureWithModels("new-model", []LaunchModel{{Name: "new-model"}, {Name: "old-model"}}); err != nil {
		t.Fatalf("ConfigureWithModels returned error: %v", err)
	}

	data, err := os.ReadFile(modelsPath)
	if err != nil {
		t.Fatal(err)
	}
	cfg := parseOMPConfigYAML(t, data)
	providers, _ := cfg["providers"].(map[string]any)
	if _, ok := providers["anthropic"]; !ok {
		t.Fatalf("expected non-Ollama provider to be preserved: %v", providers)
	}

	provider := ompProviderFromYAML(t, cfg)
	if provider["baseUrl"] != "http://127.0.0.1:11434/v1" {
		t.Fatalf("baseUrl = %v, want repaired OpenAI-compatible host", provider["baseUrl"])
	}

	entries := ompModelEntriesFromYAML(t, provider)
	if len(entries) != 2 {
		t.Fatalf("models length = %d, want 2", len(entries))
	}
	if entries[0]["id"] != "new-model" {
		t.Fatalf("first model id = %v, want new-model", entries[0]["id"])
	}
	if entries[1]["id"] != "old-model" {
		t.Fatalf("second model id = %v, want old-model", entries[1]["id"])
	}
	if entries[1]["customField"] != "keep-me" {
		t.Fatalf("custom field was not preserved: %v", entries[1])
	}

	configData, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	config := parseOMPConfigYAML(t, configData)
	if got := numericYAMLValue(config["setupVersion"]); got != ompSetupVersion {
		t.Fatalf("setupVersion = %d, want %d", got, ompSetupVersion)
	}
	if config["theme"] != "monochrome" {
		t.Fatalf("theme was not preserved: %v", config)
	}
	if config["lastChangelogVersion"] != "15.7.6" {
		t.Fatalf("lastChangelogVersion was not preserved: %v", config)
	}
}

func TestOMPConfigureWithModelsAlwaysMarksSetupComplete(t *testing.T) {
	home := t.TempDir()
	setOMPTestHome(t, home)

	configPath := filepath.Join(home, ".omp", "agent", "config.yml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(configPath, []byte("setupVersion: 2\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	o := &OMP{}
	if err := o.ConfigureWithModels("new-model", []LaunchModel{{Name: "new-model"}}); err != nil {
		t.Fatalf("ConfigureWithModels returned error: %v", err)
	}

	configData, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	config := parseOMPConfigYAML(t, configData)
	if got := numericYAMLValue(config["setupVersion"]); got != ompSetupVersion {
		t.Fatalf("setupVersion = %d, want %d", got, ompSetupVersion)
	}
}

func TestOMPConfigureWithModelsRespectsPiConfigDir(t *testing.T) {
	home := t.TempDir()
	setOMPTestHome(t, home)
	t.Setenv("PI_CONFIG_DIR", ".custom-omp")

	o := &OMP{}
	if err := o.ConfigureWithModels("new-model", []LaunchModel{{Name: "new-model"}}); err != nil {
		t.Fatalf("ConfigureWithModels returned error: %v", err)
	}

	modelsPath := filepath.Join(home, ".custom-omp", "agent", "models.yml")
	configPath := filepath.Join(home, ".custom-omp", "agent", "config.yml")
	for _, path := range []string{modelsPath, configPath} {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("expected %s to be written: %v", path, err)
		}
	}
	if _, err := os.Stat(filepath.Join(home, ".omp", "agent", "models.yml")); !os.IsNotExist(err) {
		t.Fatalf("expected default OMP models path to be untouched, got err %v", err)
	}
	if paths := o.Paths(); !slices.Equal(paths, []string{modelsPath, configPath}) {
		t.Fatalf("Paths = %v, want [%s %s]", paths, modelsPath, configPath)
	}
}

func TestOMPConfigureWithModelsRespectsPiCodingAgentDir(t *testing.T) {
	home := t.TempDir()
	setOMPTestHome(t, home)
	agentDir := filepath.Join(home, "agent-override")
	t.Setenv("PI_CONFIG_DIR", ".ignored-omp")
	t.Setenv("PI_CODING_AGENT_DIR", agentDir)

	o := &OMP{}
	if err := o.ConfigureWithModels("new-model", []LaunchModel{{Name: "new-model"}}); err != nil {
		t.Fatalf("ConfigureWithModels returned error: %v", err)
	}

	modelsPath := filepath.Join(agentDir, "models.yml")
	configPath := filepath.Join(agentDir, "config.yml")
	for _, path := range []string{modelsPath, configPath} {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("expected %s to be written: %v", path, err)
		}
	}
	if _, err := os.Stat(filepath.Join(home, ".ignored-omp", "agent", "models.yml")); !os.IsNotExist(err) {
		t.Fatalf("expected PI_CONFIG_DIR path to be ignored when PI_CODING_AGENT_DIR is set, got err %v", err)
	}
	if got := o.CurrentModel(); got != "new-model" {
		t.Fatalf("CurrentModel = %q, want new-model", got)
	}
}

func TestOMPCurrentModelRequiresHealthyProvider(t *testing.T) {
	home := t.TempDir()
	setOMPTestHome(t, home)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

	modelsPath := filepath.Join(home, ".omp", "agent", "models.yml")
	if err := os.MkdirAll(filepath.Dir(modelsPath), 0o755); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name     string
		provider string
	}{
		{
			name: "wrong base url",
			provider: "" +
				"    baseUrl: http://127.0.0.1:9999/v1\n" +
				"    api: openai-responses\n" +
				"    auth: none\n" +
				"    discovery:\n" +
				"      type: ollama\n",
		},
		{
			name: "wrong api",
			provider: "" +
				"    baseUrl: http://127.0.0.1:11434/v1\n" +
				"    api: openai-chat\n" +
				"    auth: none\n" +
				"    discovery:\n" +
				"      type: ollama\n",
		},
		{
			name: "wrong auth",
			provider: "" +
				"    baseUrl: http://127.0.0.1:11434/v1\n" +
				"    api: openai-responses\n" +
				"    auth: api-key\n" +
				"    discovery:\n" +
				"      type: ollama\n",
		},
		{
			name: "wrong discovery",
			provider: "" +
				"    baseUrl: http://127.0.0.1:11434/v1\n" +
				"    api: openai-responses\n" +
				"    auth: none\n" +
				"    discovery:\n" +
				"      type: static\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := "providers:\n" +
				"  ollama:\n" +
				tt.provider +
				"    models:\n" +
				"      - id: gemma4\n"
			if err := os.WriteFile(modelsPath, []byte(cfg), 0o644); err != nil {
				t.Fatal(err)
			}
			if got := (&OMP{}).CurrentModel(); got != "" {
				t.Fatalf("expected stale config to return empty current model, got %q", got)
			}
		})
	}
}

func parseOMPConfigYAML(t *testing.T, data []byte) map[string]any {
	t.Helper()
	var cfg map[string]any
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("generated YAML did not parse: %v\n%s", err, data)
	}
	return cfg
}

func ompProviderFromYAML(t *testing.T, cfg map[string]any) map[string]any {
	t.Helper()
	providers, ok := cfg["providers"].(map[string]any)
	if !ok {
		t.Fatalf("providers missing from config: %v", cfg)
	}
	provider, ok := providers["ollama"].(map[string]any)
	if !ok {
		t.Fatalf("ollama provider missing from config: %v", providers)
	}
	return provider
}

func ompModelEntriesFromYAML(t *testing.T, provider map[string]any) []map[string]any {
	t.Helper()
	rawModels, ok := provider["models"].([]any)
	if !ok {
		t.Fatalf("provider models missing: %v", provider)
	}
	models := make([]map[string]any, 0, len(rawModels))
	for _, raw := range rawModels {
		entry, ok := raw.(map[string]any)
		if !ok {
			t.Fatalf("model entry has unexpected type %T: %v", raw, raw)
		}
		models = append(models, entry)
	}
	return models
}

func numericYAMLValue(value any) int {
	switch v := value.(type) {
	case int:
		return v
	case int64:
		return int(v)
	case float64:
		return int(v)
	default:
		return 0
	}
}

func stringSliceYAMLValue(value any) []string {
	raw, _ := value.([]any)
	out := make([]string, 0, len(raw))
	for _, item := range raw {
		if s, ok := item.(string); ok {
			out = append(out, s)
		}
	}
	return out
}
