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

func writeFakeWSLExe(t *testing.T, dir string) {
	t.Helper()
	script := `#!/bin/sh
if [ "$1" != "bash" ] || [ "$2" != "-lc" ]; then
  exit 2
fi
case "$3" in
  'printf %s "$HOME"')
    printf "%s" "$HOME"
    ;;
  *)
    exec /bin/sh -lc "$3"
    ;;
esac
`
	if err := os.WriteFile(filepath.Join(dir, "wsl.exe"), []byte(script), 0o755); err != nil {
		t.Fatal(err)
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
}

func TestHermesConfigurePreservesExistingConfigAndEnablesWeb(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withHermesPlatform(t, runtime.GOOS)

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
	customProviders, _ := cfg["custom_providers"].([]any)
	if len(customProviders) != 1 {
		t.Fatalf("expected one managed custom provider entry, got %d", len(customProviders))
	}
	customProvider, _ := customProviders[0].(map[string]any)
	if got, _ := customProvider["name"].(string); got != "Ollama" {
		t.Fatalf("expected managed provider name Ollama, got %q", got)
	}
	if got, _ := customProvider["base_url"].(string); got != srv.URL+"/v1" {
		t.Fatalf("expected managed provider base_url %q, got %q", srv.URL+"/v1", got)
	}
	if got, _ := customProvider["model"].(string); got != "gemma4" {
		t.Fatalf("expected managed provider model gemma4, got %q", got)
	}
	if got, _ := customProvider["api_key"].(string); got != "ollama" {
		t.Fatalf("expected managed provider api_key ollama, got %q", got)
	}
	if got, _ := customProvider["api_mode"].(string); got != "chat_completions" {
		t.Fatalf("expected managed provider api_mode chat_completions, got %q", got)
	}
	customModels, _ := customProvider["models"].(map[string]any)
	if len(customModels) != 3 {
		t.Fatalf("expected managed custom provider to expose 3 models, got %v", customModels)
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
	withHermesPlatform(t, runtime.GOOS)

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
	if len(customProviders) != 2 {
		t.Fatalf("expected both custom providers to remain, got %d", len(customProviders))
	}

	first, _ := customProviders[0].(map[string]any)
	if got, _ := first["name"].(string); got != "Ollama" {
		t.Fatalf("expected managed provider name to stay Ollama, got %q", got)
	}
	if got, _ := first["base_url"].(string); got != srv.URL+"/v1" {
		t.Fatalf("expected matching provider base_url to update to %q, got %q", srv.URL+"/v1", got)
	}
	if got, _ := first["model"].(string); got != "gemma4" {
		t.Fatalf("expected matching provider model gemma4, got %q", got)
	}
	if got, _ := first["api_mode"].(string); got != "chat_completions" {
		t.Fatalf("expected matching provider api_mode chat_completions, got %q", got)
	}
	modelsCfg, _ := first["models"].(map[string]any)
	if modelsCfg == nil {
		t.Fatal("expected matching provider model metadata to be preserved")
	}
	if len(modelsCfg) != 3 {
		t.Fatalf("expected managed provider models map to refresh full catalog, got %v", modelsCfg)
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

	second, _ := customProviders[1].(map[string]any)
	if got, _ := second["name"].(string); got != "Other Endpoint" {
		t.Fatalf("expected unrelated custom provider to be preserved, got %q", got)
	}
}

func TestHermesConfigureUsesLaunchResolvedHostForModelDiscovery(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withHermesPlatform(t, runtime.GOOS)

	configPath := filepath.Join(tmpDir, ".hermes", "config.yaml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			fmt.Fprint(w, `{"model_info":{"general.context_length":131072}}`)
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
	withHermesPlatform(t, runtime.GOOS)

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
}

func TestHermesPathsUsesWSLConfigPathOnWindows(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("simulates WSL through POSIX shell scripts")
	}

	tmpDir := t.TempDir()
	winHome := filepath.Join(tmpDir, "winhome")
	wslHome := filepath.Join(tmpDir, "wslhome")
	setTestHome(t, winHome)
	withHermesPlatform(t, "windows")
	withHermesUserHome(t, winHome)
	t.Setenv("HOME", wslHome)
	t.Setenv("PATH", tmpDir)
	writeFakeWSLExe(t, tmpDir)

	got := (&Hermes{}).Paths()
	want := filepath.Join(wslHome, ".hermes", "config.yaml")
	if len(got) != 1 || got[0] != want {
		t.Fatalf("expected WSL config path %q, got %v", want, got)
	}
}

func TestHermesPathsUsesLocalConfigPathForNativeWindowsHermes(t *testing.T) {
	tmpDir := t.TempDir()
	winHome := filepath.Join(tmpDir, "winhome")
	setTestHome(t, winHome)
	withHermesPlatform(t, "windows")
	withHermesUserHome(t, winHome)
	t.Setenv("PATH", tmpDir)
	if err := os.WriteFile(filepath.Join(tmpDir, "hermes"), []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	writeFakeWSLExe(t, tmpDir)

	got := (&Hermes{}).Paths()
	want := filepath.Join(winHome, ".hermes", "config.yaml")
	if len(got) != 1 || got[0] != want {
		t.Fatalf("expected local config path %q, got %v", want, got)
	}
}

func TestHermesConfigureAndCurrentModelUseWSLConfigOnWindows(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("simulates WSL through POSIX shell scripts")
	}

	tmpDir := t.TempDir()
	winHome := filepath.Join(tmpDir, "winhome")
	wslHome := filepath.Join(tmpDir, "wslhome")
	setTestHome(t, winHome)
	withHermesPlatform(t, "windows")
	withHermesUserHome(t, winHome)
	t.Setenv("HOME", wslHome)
	t.Setenv("PATH", tmpDir)
	writeFakeWSLExe(t, tmpDir)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
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

	if _, err := os.Stat(filepath.Join(winHome, ".hermes", "config.yaml")); !os.IsNotExist(err) {
		t.Fatalf("expected Windows-side config to remain untouched, got err=%v", err)
	}

	wslConfigPath := filepath.Join(wslHome, ".hermes", "config.yaml")
	data, err := os.ReadFile(wslConfigPath)
	if err != nil {
		t.Fatalf("expected WSL-side config to be written: %v", err)
	}

	var cfg map[string]any
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("failed to parse WSL config: %v", err)
	}
	modelCfg, _ := cfg["model"].(map[string]any)
	if got, _ := modelCfg["provider"].(string); got != "ollama-launch" {
		t.Fatalf("expected WSL config provider ollama-launch, got %q", got)
	}
	if got := h.CurrentModel(); got != "gemma4" {
		t.Fatalf("expected CurrentModel to read WSL config, got %q", got)
	}
}

func TestHermesCurrentModelRequiresHealthyManagedConfig(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withHermesPlatform(t, runtime.GOOS)
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
	withHermesPlatform(t, runtime.GOOS)

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
	withHermesPlatform(t, runtime.GOOS)
	t.Setenv("PATH", tmpDir+string(os.PathListSeparator)+os.Getenv("PATH"))

	bin := filepath.Join(tmpDir, "hermes")
	if err := os.WriteFile(bin, []byte("#!/bin/sh\nprintf '[%s]\\n' \"$*\" >> \"$HOME/hermes-invocations.log\"\n"), 0o755); err != nil {
		t.Fatal(err)
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

func TestHermesEnsureInstalledWindowsWithoutWSLGivesGuidance(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withHermesPlatform(t, "windows")
	t.Setenv("PATH", tmpDir)

	h := &Hermes{}
	err := h.ensureInstalled()
	if err == nil {
		t.Fatal("expected missing WSL guidance error")
	}
	if !strings.Contains(err.Error(), "wsl --install") {
		t.Fatalf("expected WSL guidance, got %v", err)
	}
}

func TestHermesEnsureInstalledWindowsUsesWSLHandoff(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("simulates WSL through POSIX shell scripts")
	}

	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withHermesPlatform(t, "windows")
	withLauncherHooks(t)
	t.Setenv("PATH", tmpDir)

	logPath := filepath.Join(tmpDir, "wsl.log")
	markerPath := filepath.Join(tmpDir, "installed.marker")
	script := fmt.Sprintf(`#!/bin/sh
printf '%%s\n' "$*" >> %q
case "$*" in
  *"command -v hermes >/dev/null 2>&1"*)
    if [ -f %q ]; then exit 0; fi
    exit 1
    ;;
  *)
    /usr/bin/touch %q
    exit 0
    ;;
esac
`, logPath, markerPath, markerPath)
	if err := os.WriteFile(filepath.Join(tmpDir, "wsl.exe"), []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}

	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		return true, nil
	}

	h := &Hermes{}
	if err := h.ensureInstalled(); err != nil {
		t.Fatalf("ensureInstalled returned error: %v", err)
	}

	data, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	logs := string(data)
	if !strings.Contains(logs, "curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash") {
		t.Fatalf("expected WSL install handoff to run official installer, got logs:\n%s", logs)
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
