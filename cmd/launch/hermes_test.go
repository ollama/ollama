package launch

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/ollama/ollama/cmd/config"
)

func withHermesPlatform(t *testing.T, goos string) {
	t.Helper()
	old := hermesGOOS
	hermesGOOS = goos
	t.Cleanup(func() {
		hermesGOOS = old
	})
}

func withHermesOllamaURL(t *testing.T, rawURL string) {
	t.Helper()
	old := hermesOllamaURL
	hermesOllamaURL = func() *url.URL {
		u, err := url.Parse(rawURL)
		if err != nil {
			t.Fatalf("parse test Ollama URL: %v", err)
		}
		return u
	}
	t.Cleanup(func() {
		hermesOllamaURL = old
	})
}

func withHermesUserHome(t *testing.T, dir string) {
	t.Helper()
	old := hermesUserHome
	hermesUserHome = func() (string, error) { return dir, nil }
	t.Cleanup(func() {
		hermesUserHome = old
	})
}

func clearHermesMessagingEnvVars(t *testing.T) {
	t.Helper()
	for _, group := range hermesMessagingEnvGroups {
		for _, key := range group {
			if value, ok := os.LookupEnv(key); ok {
				t.Setenv(key, value)
			} else {
				t.Setenv(key, "")
			}
			if err := os.Unsetenv(key); err != nil {
				t.Fatalf("unset %s: %v", key, err)
			}
		}
	}
}

func TestHermesIntegration(t *testing.T) {
	h := &Hermes{}

	t.Run("implements Runner", func(t *testing.T) {
		var _ Runner = h
	})

	t.Run("implements managed single model", func(t *testing.T) {
		var _ ManagedSingleModel = h
	})

	t.Run("implements managed runtime refresher", func(t *testing.T) {
		var _ ManagedRuntimeRefresher = h
	})
}

func TestHermesConfigurePreservesExistingConfigAndEnablesWeb(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withHermesPlatform(t, "darwin")

	configPath := filepath.Join(tmpDir, ".hermes", "config.yaml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	existing := "" +
		"memory:\n" +
		"  provider: local\n" +
		"toolsets:\n" +
		"  - terminal\n"
	if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			fmt.Fprint(w, `{"model_info":{"general.context_length":131072}}`)
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"gemma4"},{"name":"qwen3.5"},{"name":"llama3.3"}]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	h := &Hermes{}
	if err := h.Configure("gemma4"); err != nil {
		t.Fatalf("Configure returned error: %v", err)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}

	var cfg map[string]any
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("failed to parse rewritten yaml: %v", err)
	}

	modelCfg, _ := cfg["model"].(map[string]any)
	if got, _ := modelCfg["provider"].(string); got != "ollama-launch" {
		t.Fatalf("expected provider ollama-launch, got %q", got)
	}
	if got, _ := modelCfg["default"].(string); got != "gemma4" {
		t.Fatalf("expected default model gemma4, got %q", got)
	}
	if got, _ := modelCfg["base_url"].(string); got != srv.URL+"/v1" {
		t.Fatalf("expected Ollama base_url %q, got %q", srv.URL+"/v1", got)
	}
	if got, _ := modelCfg["api_key"].(string); got != "ollama" {
		t.Fatalf("expected placeholder api_key ollama, got %q", got)
	}
	if memoryCfg, _ := cfg["memory"].(map[string]any); memoryCfg == nil {
		t.Fatal("expected unrelated config to be preserved")
	}
	if _, ok := cfg["custom_providers"]; ok {
		t.Fatal("expected launcher-managed config to avoid custom_providers duplicates")
	}
	providersCfg, _ := cfg["providers"].(map[string]any)
	ollamaProvider, _ := providersCfg["ollama-launch"].(map[string]any)
	if ollamaProvider == nil {
		t.Fatal("expected ollama-launch provider entry")
	}
	if got, _ := ollamaProvider["name"].(string); got != "Ollama" {
		t.Fatalf("expected providers entry name Ollama, got %q", got)
	}
	if got, _ := ollamaProvider["api"].(string); got != srv.URL+"/v1" {
		t.Fatalf("expected providers entry api %q, got %q", srv.URL+"/v1", got)
	}
	if got, _ := ollamaProvider["default_model"].(string); got != "gemma4" {
		t.Fatalf("expected providers entry default_model gemma4, got %q", got)
	}
	models, _ := ollamaProvider["models"].([]any)
	if len(models) != 3 {
		t.Fatalf("expected providers entry to expose 3 models, got %v", models)
	}

	toolsets, _ := cfg["toolsets"].([]any)
	var gotToolsets []string
	for _, item := range toolsets {
		if s, _ := item.(string); s != "" {
			gotToolsets = append(gotToolsets, s)
		}
	}
	if !strings.Contains(strings.Join(gotToolsets, ","), "terminal") || !strings.Contains(strings.Join(gotToolsets, ","), "web") {
		t.Fatalf("expected toolsets to preserve terminal and add web, got %v", gotToolsets)
	}
}

func TestHermesConfigureUpdatesMatchingCustomProviderWithoutDroppingFields(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withHermesPlatform(t, "darwin")

	configPath := filepath.Join(tmpDir, ".hermes", "config.yaml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	existing := "" +
		"providers:\n" +
		"  ollama:\n" +
		"    name: Ollama\n" +
		"    api: http://127.0.0.1:11434/v1\n" +
		"    default_model: old-model\n" +
		"    models:\n" +
		"      - old-model\n" +
		"      - older-model\n" +
		"    extra_field: keep-me\n" +
		"custom_providers:\n" +
		"  - name: Ollama\n" +
		"    base_url: http://127.0.0.1:11434/v1\n" +
		"    model: old-model\n" +
		"    api_mode: chat_completions\n" +
		"    models:\n" +
		"      old-model:\n" +
		"        context_length: 65536\n" +
		"  - name: Other Endpoint\n" +
		"    base_url: https://example.invalid/v1\n" +
		"    model: untouched\n"
	if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			fmt.Fprint(w, `{"model_info":{"general.context_length":131072}}`)
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"gemma4"},{"name":"qwen3.5"},{"name":"llama3.3"}]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	h := &Hermes{}
	if err := h.Configure("gemma4"); err != nil {
		t.Fatalf("Configure returned error: %v", err)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}

	var cfg map[string]any
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("failed to parse rewritten yaml: %v", err)
	}

	modelCfg, _ := cfg["model"].(map[string]any)
	if got, _ := modelCfg["provider"].(string); got != "ollama-launch" {
		t.Fatalf("expected managed providers entry to migrate to ollama-launch, got %q", got)
	}

	customProviders, _ := cfg["custom_providers"].([]any)
	if len(customProviders) != 1 {
		t.Fatalf("expected only unrelated custom providers to remain, got %d", len(customProviders))
	}

	providersCfg, _ := cfg["providers"].(map[string]any)
	if _, ok := providersCfg["ollama"]; ok {
		t.Fatal("expected legacy providers.ollama entry to be removed")
	}
	ollamaProvider, _ := providersCfg["ollama-launch"].(map[string]any)
	if ollamaProvider == nil {
		t.Fatal("expected ollama-launch providers entry to remain")
	}
	if got, _ := ollamaProvider["api"].(string); got != srv.URL+"/v1" {
		t.Fatalf("expected providers entry api to update to %q, got %q", srv.URL+"/v1", got)
	}
	if got, _ := ollamaProvider["default_model"].(string); got != "gemma4" {
		t.Fatalf("expected providers entry default_model gemma4, got %q", got)
	}
	if got, _ := ollamaProvider["extra_field"].(string); got != "keep-me" {
		t.Fatalf("expected providers entry extra_field to be preserved, got %q", got)
	}
	providerModels, _ := ollamaProvider["models"].([]any)
	if len(providerModels) != 3 {
		t.Fatalf("expected providers entry to refresh full model catalog, got %v", providerModels)
	}

	remaining, _ := customProviders[0].(map[string]any)
	if got, _ := remaining["name"].(string); got != "Other Endpoint" {
		t.Fatalf("expected unrelated custom provider to be preserved, got %q", got)
	}
}

func TestHermesConfigureUsesLaunchResolvedHostForModelDiscovery(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withHermesPlatform(t, "darwin")

	configPath := filepath.Join(tmpDir, ".hermes", "config.yaml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			fmt.Fprint(w, `{"model_info":{"general.context_length":131072}}`)
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"gemma4"},{"name":"qwen3.5"},{"name":"llama3.3"}]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	withHermesOllamaURL(t, srv.URL)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:1")

	h := &Hermes{}
	if err := h.Configure("gemma4"); err != nil {
		t.Fatalf("Configure returned error: %v", err)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}

	var cfg map[string]any
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("failed to parse rewritten yaml: %v", err)
	}

	providersCfg, _ := cfg["providers"].(map[string]any)
	ollamaProvider, _ := providersCfg["ollama-launch"].(map[string]any)
	if ollamaProvider == nil {
		t.Fatal("expected ollama-launch provider entry")
	}
	models, _ := ollamaProvider["models"].([]any)
	if len(models) != 3 {
		t.Fatalf("expected providers entry to expose 3 launch-resolved models, got %v", models)
	}
}

func TestHermesConfigureMigratesLegacyManagedAliases(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withHermesPlatform(t, "darwin")

	configPath := filepath.Join(tmpDir, ".hermes", "config.yaml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	existing := "" +
		"model:\n" +
		"  provider: custom:ollama\n" +
		"  default: old-model\n" +
		"providers:\n" +
		"  ollama:\n" +
		"    name: Ollama\n" +
		"    api: http://127.0.0.1:11434/v1\n" +
		"    default_model: old-model\n" +
		"custom_providers:\n" +
		"  - name: Ollama\n" +
		"    base_url: http://127.0.0.1:11434/v1\n" +
		"    model: old-model\n"
	if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"gemma4"},{"name":"qwen3.5"}]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	h := &Hermes{}
	if err := h.Configure("gemma4"); err != nil {
		t.Fatalf("Configure returned error: %v", err)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}

	var cfg map[string]any
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("failed to parse rewritten yaml: %v", err)
	}

	modelCfg, _ := cfg["model"].(map[string]any)
	if got, _ := modelCfg["provider"].(string); got != "ollama-launch" {
		t.Fatalf("expected migrated provider ollama-launch, got %q", got)
	}

	providersCfg, _ := cfg["providers"].(map[string]any)
	if _, ok := providersCfg["ollama"]; ok {
		t.Fatal("expected legacy providers.ollama entry to be removed")
	}
	if _, ok := providersCfg["ollama-launch"]; !ok {
		t.Fatal("expected providers.ollama-launch entry")
	}
	if _, ok := cfg["custom_providers"]; ok {
		t.Fatal("expected managed custom_providers entry to be removed during migration")
	}
}

func TestHermesPathsUsesLocalConfigPathForNativeWindowsHermes(t *testing.T) {
	tmpDir := t.TempDir()
	winHome := filepath.Join(tmpDir, "winhome")
	setTestHome(t, winHome)
	withHermesPlatform(t, "windows")
	withHermesUserHome(t, winHome)
	t.Setenv("PATH", tmpDir)
	writeFakeBinary(t, tmpDir, "hermes")

	got := (&Hermes{}).Paths()
	want := filepath.Join(winHome, ".hermes", "config.yaml")
	if len(got) != 1 || got[0] != want {
		t.Fatalf("expected local config path %q, got %v", want, got)
	}
}

func TestHermesCurrentModelRequiresHealthyManagedConfig(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withHermesPlatform(t, "darwin")
	withHermesOllamaURL(t, "http://127.0.0.1:11434")

	configPath := filepath.Join(tmpDir, ".hermes", "config.yaml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name string
		cfg  string
	}{
		{
			name: "wrong provider",
			cfg: "" +
				"model:\n" +
				"  provider: openrouter\n" +
				"  default: gemma4\n" +
				"  base_url: http://127.0.0.1:11434/v1\n",
		},
		{
			name: "wrong base url",
			cfg: "" +
				"model:\n" +
				"  provider: ollama-launch\n" +
				"  default: gemma4\n" +
				"  base_url: http://127.0.0.1:9999/v1\n" +
				"providers:\n" +
				"  ollama-launch:\n" +
				"    api: http://127.0.0.1:9999/v1\n" +
				"    default_model: gemma4\n",
		},
		{
			name: "missing managed provider entry",
			cfg: "" +
				"model:\n" +
				"  provider: ollama-launch\n" +
				"  default: gemma4\n" +
				"  base_url: http://127.0.0.1:11434/v1\n",
		},
		{
			name: "inconsistent managed provider entry",
			cfg: "" +
				"model:\n" +
				"  provider: ollama-launch\n" +
				"  default: gemma4\n" +
				"  base_url: http://127.0.0.1:11434/v1\n" +
				"providers:\n" +
				"  ollama-launch:\n" +
				"    api: http://127.0.0.1:11434/v1\n" +
				"    default_model: qwen3.5\n",
		},
		{
			name: "legacy launch managed config",
			cfg: "" +
				"model:\n" +
				"  provider: custom:ollama\n" +
				"  default: gemma4\n" +
				"  base_url: http://127.0.0.1:11434/v1\n" +
				"providers:\n" +
				"  ollama:\n" +
				"    api: http://127.0.0.1:11434/v1\n" +
				"    default_model: gemma4\n",
		},
		{
			name: "duplicate managed custom provider",
			cfg: "" +
				"model:\n" +
				"  provider: ollama-launch\n" +
				"  default: gemma4\n" +
				"  base_url: http://127.0.0.1:11434/v1\n" +
				"providers:\n" +
				"  ollama-launch:\n" +
				"    api: http://127.0.0.1:11434/v1\n" +
				"    default_model: gemma4\n" +
				"custom_providers:\n" +
				"  - name: Ollama\n" +
				"    base_url: http://127.0.0.1:11434/v1\n" +
				"    model: gemma4\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := os.WriteFile(configPath, []byte(tt.cfg), 0o644); err != nil {
				t.Fatal(err)
			}
			if got := (&Hermes{}).CurrentModel(); got != "" {
				t.Fatalf("expected stale config to return empty current model, got %q", got)
			}
		})
	}
}

func TestHermesCurrentModelReturnsEmptyWhenConfigMissing(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withHermesPlatform(t, "darwin")

	if got := (&Hermes{}).CurrentModel(); got != "" {
		t.Fatalf("expected missing config to return empty current model, got %q", got)
	}
}

func TestHermesRunPassthroughArgs(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX shell test binary")
	}

	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withLauncherHooks(t)
	withInteractiveSession(t, true)
	withHermesPlatform(t, runtime.GOOS)
	clearHermesMessagingEnvVars(t)
	t.Setenv("PATH", tmpDir+string(os.PathListSeparator)+os.Getenv("PATH"))

	bin := filepath.Join(tmpDir, "hermes")
	if err := os.WriteFile(bin, []byte("#!/bin/sh\nprintf '[%s]\\n' \"$*\" >> \"$HOME/hermes-invocations.log\"\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		t.Fatalf("did not expect messaging prompt during passthrough launch: %s", prompt)
		return false, nil
	}

	h := &Hermes{}
	if err := h.Run("", []string{"--continue"}); err != nil {
		t.Fatalf("Run returned error: %v", err)
	}

	data, err := os.ReadFile(filepath.Join(tmpDir, "hermes-invocations.log"))
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.TrimSpace(string(data)); got != "[--continue]" {
		t.Fatalf("expected passthrough args to reach hermes, got %q", got)
	}
}

func TestHermesRun_PromptsForMessagingSetupBeforeDefaultLaunch(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX shell test binary")
	}

	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withLauncherHooks(t)
	withInteractiveSession(t, true)
	withHermesPlatform(t, runtime.GOOS)
	clearHermesMessagingEnvVars(t)
	t.Setenv("PATH", tmpDir+string(os.PathListSeparator)+os.Getenv("PATH"))

	bin := filepath.Join(tmpDir, "hermes")
	script := `#!/bin/sh
printf '[%s]\n' "$*" >> "$HOME/hermes-invocations.log"
if [ "$1" = "gateway" ] && [ "$2" = "setup" ]; then
  /bin/mkdir -p "$HOME/.hermes"
  printf 'TELEGRAM_BOT_TOKEN=configured\n' > "$HOME/.hermes/.env"
fi
`
	if err := os.WriteFile(bin, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}

	promptCount := 0
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		promptCount++
		if prompt != hermesGatewaySetupTitle {
			t.Fatalf("unexpected prompt %q", prompt)
		}
		if options.YesLabel != "Yes" || options.NoLabel != "Set up later" {
			t.Fatalf("unexpected prompt labels: %+v", options)
		}
		return true, nil
	}

	h := &Hermes{}
	if err := h.Run("", nil); err != nil {
		t.Fatalf("Run returned error: %v", err)
	}

	if promptCount != 1 {
		t.Fatalf("expected one messaging prompt, got %d", promptCount)
	}

	data, err := os.ReadFile(filepath.Join(tmpDir, "hermes-invocations.log"))
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 2 {
		t.Fatalf("expected setup then launch invocations, got %v", lines)
	}
	if lines[0] != "[gateway setup]" {
		t.Fatalf("expected gateway setup first, got %q", lines[0])
	}
	if lines[1] != "[]" {
		t.Fatalf("expected default hermes launch after setup, got %q", lines[1])
	}
}

func TestHermesRun_SetUpLaterRepromptsOnLaterLaunches(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX shell test binary")
	}

	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withLauncherHooks(t)
	withInteractiveSession(t, true)
	withHermesPlatform(t, runtime.GOOS)
	clearHermesMessagingEnvVars(t)
	t.Setenv("PATH", tmpDir+string(os.PathListSeparator)+os.Getenv("PATH"))

	bin := filepath.Join(tmpDir, "hermes")
	if err := os.WriteFile(bin, []byte("#!/bin/sh\nprintf '[%s]\\n' \"$*\" >> \"$HOME/hermes-invocations.log\"\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	promptCount := 0
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		promptCount++
		if prompt != hermesGatewaySetupTitle {
			t.Fatalf("unexpected prompt %q", prompt)
		}
		return false, nil
	}

	h := &Hermes{}
	if err := h.Run("", nil); err != nil {
		t.Fatalf("first Run returned error: %v", err)
	}
	if err := h.Run("", nil); err != nil {
		t.Fatalf("second Run returned error: %v", err)
	}

	if promptCount != 2 {
		t.Fatalf("expected two prompts across two launches, got %d", promptCount)
	}

	data, err := os.ReadFile(filepath.Join(tmpDir, "hermes-invocations.log"))
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 2 {
		t.Fatalf("expected one default launch per run, got %v", lines)
	}
	for _, line := range lines {
		if line != "[]" {
			t.Fatalf("expected only default launches after choosing later, got %v", lines)
		}
	}
}

func TestHermesRun_SkipsMessagingPromptWhenConfigured(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX shell test binary")
	}

	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withLauncherHooks(t)
	withInteractiveSession(t, true)
	withHermesPlatform(t, runtime.GOOS)
	clearHermesMessagingEnvVars(t)
	t.Setenv("PATH", tmpDir+string(os.PathListSeparator)+os.Getenv("PATH"))

	envPath := filepath.Join(tmpDir, ".hermes", ".env")
	if err := os.MkdirAll(filepath.Dir(envPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(envPath, []byte("DISCORD_BOT_TOKEN=configured\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	bin := filepath.Join(tmpDir, "hermes")
	if err := os.WriteFile(bin, []byte("#!/bin/sh\nprintf '[%s]\\n' \"$*\" >> \"$HOME/hermes-invocations.log\"\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		t.Fatalf("did not expect messaging prompt when Hermes gateway is configured: %s", prompt)
		return false, nil
	}

	h := &Hermes{}
	if err := h.Run("", nil); err != nil {
		t.Fatalf("Run returned error: %v", err)
	}

	data, err := os.ReadFile(filepath.Join(tmpDir, "hermes-invocations.log"))
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.TrimSpace(string(data)); got != "[]" {
		t.Fatalf("expected only default launch invocation, got %q", got)
	}
}

func TestHermesRun_SkipsMessagingPromptWithYesPolicy(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX shell test binary")
	}

	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withLauncherHooks(t)
	withInteractiveSession(t, true)
	withHermesPlatform(t, runtime.GOOS)
	clearHermesMessagingEnvVars(t)
	t.Setenv("PATH", tmpDir+string(os.PathListSeparator)+os.Getenv("PATH"))

	bin := filepath.Join(tmpDir, "hermes")
	if err := os.WriteFile(bin, []byte("#!/bin/sh\nprintf '[%s]\\n' \"$*\" >> \"$HOME/hermes-invocations.log\"\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	restoreConfirm := withLaunchConfirmPolicy(launchConfirmPolicy{yes: true})
	defer restoreConfirm()

	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		t.Fatalf("did not expect messaging prompt in --yes mode: %s", prompt)
		return false, nil
	}

	h := &Hermes{}
	if err := h.Run("", nil); err != nil {
		t.Fatalf("Run returned error: %v", err)
	}

	data, err := os.ReadFile(filepath.Join(tmpDir, "hermes-invocations.log"))
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.TrimSpace(string(data)); got != "[]" {
		t.Fatalf("expected only default launch invocation, got %q", got)
	}
}

func TestHermesRun_MessagingSetupFailureStopsLaunch(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX shell test binary")
	}

	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withLauncherHooks(t)
	withInteractiveSession(t, true)
	withHermesPlatform(t, runtime.GOOS)
	clearHermesMessagingEnvVars(t)
	t.Setenv("PATH", tmpDir+string(os.PathListSeparator)+os.Getenv("PATH"))

	bin := filepath.Join(tmpDir, "hermes")
	script := `#!/bin/sh
printf '[%s]\n' "$*" >> "$HOME/hermes-invocations.log"
if [ "$1" = "gateway" ] && [ "$2" = "setup" ]; then
  exit 23
fi
`
	if err := os.WriteFile(bin, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}

	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		if prompt != hermesGatewaySetupTitle {
			t.Fatalf("unexpected prompt %q", prompt)
		}
		return true, nil
	}

	h := &Hermes{}
	err := h.Run("", nil)
	if err == nil {
		t.Fatal("expected messaging setup failure")
	}
	if !strings.Contains(err.Error(), "hermes messaging setup failed") {
		t.Fatalf("expected helpful messaging setup error, got %v", err)
	}
	if !strings.Contains(err.Error(), hermesGatewaySetupHint) {
		t.Fatalf("expected recovery hint, got %v", err)
	}

	data, readErr := os.ReadFile(filepath.Join(tmpDir, "hermes-invocations.log"))
	if readErr != nil {
		t.Fatal(readErr)
	}
	if got := strings.TrimSpace(string(data)); got != "[gateway setup]" {
		t.Fatalf("expected launch to stop after failed setup, got %q", got)
	}
}

func TestHermesRefreshRuntimeAfterConfigure_RestartsRunningGateway(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX shell test binary")
	}

	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withHermesPlatform(t, runtime.GOOS)
	t.Setenv("PATH", tmpDir+string(os.PathListSeparator)+os.Getenv("PATH"))

	bin := filepath.Join(tmpDir, "hermes")
	script := `#!/bin/sh
printf '[%s]\n' "$*" >> "$HOME/hermes-invocations.log"
if [ "$1" = "gateway" ] && [ "$2" = "status" ]; then
  printf '✓ Gateway is running (PID: 123)\n'
fi
`
	if err := os.WriteFile(bin, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}

	h := &Hermes{}
	if err := h.RefreshRuntimeAfterConfigure(); err != nil {
		t.Fatalf("RefreshRuntimeAfterConfigure returned error: %v", err)
	}

	data, err := os.ReadFile(filepath.Join(tmpDir, "hermes-invocations.log"))
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 2 {
		t.Fatalf("expected status then restart invocations, got %v", lines)
	}
	if lines[0] != "[gateway status]" {
		t.Fatalf("expected gateway status first, got %q", lines[0])
	}
	if lines[1] != "[gateway restart]" {
		t.Fatalf("expected gateway restart second, got %q", lines[1])
	}
}

func TestHermesRefreshRuntimeAfterConfigure_SkipsRestartWhenGatewayStopped(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX shell test binary")
	}

	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withHermesPlatform(t, runtime.GOOS)
	t.Setenv("PATH", tmpDir+string(os.PathListSeparator)+os.Getenv("PATH"))

	bin := filepath.Join(tmpDir, "hermes")
	script := `#!/bin/sh
printf '[%s]\n' "$*" >> "$HOME/hermes-invocations.log"
if [ "$1" = "gateway" ] && [ "$2" = "status" ]; then
  printf '✗ Gateway is not running\n'
fi
`
	if err := os.WriteFile(bin, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}

	h := &Hermes{}
	if err := h.RefreshRuntimeAfterConfigure(); err != nil {
		t.Fatalf("RefreshRuntimeAfterConfigure returned error: %v", err)
	}

	data, err := os.ReadFile(filepath.Join(tmpDir, "hermes-invocations.log"))
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.TrimSpace(string(data)); got != "[gateway status]" {
		t.Fatalf("expected only gateway status invocation, got %q", got)
	}
}

func TestHermesMessagingConfiguredRecognizesSupportedGatewayVars(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withHermesPlatform(t, "darwin")
	clearHermesMessagingEnvVars(t)

	envPath := filepath.Join(tmpDir, ".hermes", ".env")
	if err := os.MkdirAll(filepath.Dir(envPath), 0o755); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name string
		env  string
		want bool
	}{
		{name: "none", env: "", want: false},
		{name: "telegram", env: "TELEGRAM_BOT_TOKEN=token\n", want: true},
		{name: "discord", env: "DISCORD_BOT_TOKEN=token\n", want: true},
		{name: "slack", env: "SLACK_BOT_TOKEN=token\n", want: true},
		{name: "signal", env: "SIGNAL_ACCOUNT=account\n", want: true},
		{name: "email", env: "EMAIL_ADDRESS=user@example.com\n", want: true},
		{name: "sms", env: "TWILIO_ACCOUNT_SID=sid\n", want: true},
		{name: "matrix token", env: "MATRIX_ACCESS_TOKEN=token\n", want: true},
		{name: "matrix password", env: "MATRIX_PASSWORD=secret\n", want: true},
		{name: "mattermost", env: "MATTERMOST_TOKEN=token\n", want: true},
		{name: "whatsapp", env: "WHATSAPP_PHONE_NUMBER_ID=phone\n", want: true},
		{name: "dingtalk", env: "DINGTALK_CLIENT_ID=client\n", want: true},
		{name: "feishu", env: "FEISHU_APP_ID=app\n", want: true},
		{name: "wecom", env: "WECOM_BOT_ID=bot\n", want: true},
		{name: "weixin", env: "WEIXIN_ACCOUNT_ID=account\n", want: true},
		{name: "bluebubbles", env: "BLUEBUBBLES_SERVER_URL=https://example.invalid\n", want: true},
		{name: "webhooks", env: "WEBHOOK_ENABLED=true\n", want: true},
	}

	h := &Hermes{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := os.WriteFile(envPath, []byte(tt.env), 0o644); err != nil {
				t.Fatal(err)
			}
			if got := h.messagingConfigured(); got != tt.want {
				t.Fatalf("messagingConfigured() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestHermesEnsureInstalledWindowsShowsWSLGuidance(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withHermesPlatform(t, "windows")
	t.Setenv("PATH", tmpDir)

	h := &Hermes{}
	err := h.ensureInstalled()
	if err == nil {
		t.Fatal("expected WSL guidance error")
	}
	msg := err.Error()
	if !strings.Contains(msg, "wsl --install") {
		t.Fatalf("expected install command in guidance, got %v", err)
	}
	if !strings.Contains(msg, "hermes-agent.nousresearch.com") {
		t.Fatalf("expected docs link in guidance, got %v", err)
	}
	if strings.Contains(msg, "hermes is not installed") {
		t.Fatalf("guidance should not lead with 'hermes is not installed', got %v", err)
	}
}

func TestHermesEnsureInstalledUnixPromptsBeforeInstall(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell test binaries")
	}

	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withHermesPlatform(t, "darwin")
	withLauncherHooks(t)
	t.Setenv("PATH", tmpDir)

	writeScript := func(name, content string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(tmpDir, name), []byte(content), 0o755); err != nil {
			t.Fatal(err)
		}
	}

	writeScript("curl", "#!/bin/sh\nexit 0\n")
	writeScript("git", "#!/bin/sh\nexit 0\n")
	writeScript("bash", fmt.Sprintf(`#!/bin/sh
printf '%%s\n' "$*" >> %q
/bin/cat > %q <<'EOS'
#!/bin/sh
exit 0
EOS
/bin/chmod +x %q
exit 0
`, filepath.Join(tmpDir, "bash.log"), filepath.Join(tmpDir, "hermes"), filepath.Join(tmpDir, "hermes")))

	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		if prompt != "Hermes is not installed. Install now?" {
			t.Fatalf("unexpected install prompt %q", prompt)
		}
		return true, nil
	}

	h := &Hermes{}
	if err := h.ensureInstalled(); err != nil {
		t.Fatalf("ensureInstalled returned error: %v", err)
	}

	data, err := os.ReadFile(filepath.Join(tmpDir, "bash.log"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "--skip-setup") {
		t.Fatalf("expected install script to skip upstream setup, got logs:\n%s", data)
	}
	if !strings.Contains(string(data), "-lc "+hermesInstallScript) {
		t.Fatalf("expected official install script invocation, got logs:\n%s", data)
	}
}

func TestHermesEnsureInstalledUnixCanBeDeclined(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell test binaries")
	}

	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withHermesPlatform(t, "darwin")
	withLauncherHooks(t)
	t.Setenv("PATH", tmpDir)

	for _, name := range []string{"bash", "curl", "git"} {
		if err := os.WriteFile(filepath.Join(tmpDir, name), []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
			t.Fatal(err)
		}
	}

	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		if prompt != "Hermes is not installed. Install now?" {
			t.Fatalf("unexpected install prompt %q", prompt)
		}
		return false, nil
	}

	h := &Hermes{}
	err := h.ensureInstalled()
	if err == nil || !strings.Contains(err.Error(), "hermes installation cancelled") {
		t.Fatalf("expected install cancellation error, got %v", err)
	}
}

func TestHermesOnboardSkipsWhenLaunchConfigAlreadyMarked(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withHermesPlatform(t, runtime.GOOS)

	if err := config.MarkIntegrationOnboarded("hermes"); err != nil {
		t.Fatalf("failed to mark Hermes onboarded: %v", err)
	}

	h := &Hermes{}
	if err := h.Onboard(); err != nil {
		t.Fatalf("expected Onboard to no-op when already marked, got %v", err)
	}
}

func TestHermesOnboardMarksLaunchConfig(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withHermesPlatform(t, runtime.GOOS)

	h := &Hermes{}
	if err := h.Onboard(); err != nil {
		t.Fatalf("Onboard returned error: %v", err)
	}

	saved, err := config.LoadIntegration("hermes")
	if err != nil {
		t.Fatalf("failed to load Hermes integration config: %v", err)
	}
	if !saved.Onboarded {
		t.Fatal("expected Hermes to be marked onboarded")
	}
}

func TestHermesGatewayStatusRunningRecognizesRunningStates(t *testing.T) {
	tests := []struct {
		name   string
		output string
		want   bool
	}{
		{name: "manual", output: "✓ Gateway is running (PID: 123)", want: true},
		{name: "systemd", output: "✓ User gateway service is running", want: true},
		{name: "launchd", output: "✓ Gateway service is loaded", want: true},
		{name: "manual stopped", output: "✗ Gateway is not running", want: false},
		{name: "systemd stopped", output: "✗ User gateway service is stopped", want: false},
		{name: "launchd unloaded", output: "✗ Gateway service is not loaded", want: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := hermesGatewayStatusRunning(tt.output); got != tt.want {
				t.Fatalf("hermesGatewayStatusRunning(%q) = %v, want %v", tt.output, got, tt.want)
			}
		})
	}
}
