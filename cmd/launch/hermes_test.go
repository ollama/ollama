package launch

import (
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

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

func withHermesGatewayAddr(t *testing.T, addr string) {
	t.Helper()
	oldAddr := hermesGatewayAddr
	oldStartWait := hermesGatewayStartWait
	oldServiceWait := hermesGatewayServiceWait
	hermesGatewayAddr = addr
	hermesGatewayStartWait = 3 * time.Second
	hermesGatewayServiceWait = 250 * time.Millisecond
	t.Cleanup(func() {
		hermesGatewayAddr = oldAddr
		hermesGatewayStartWait = oldStartWait
		hermesGatewayServiceWait = oldServiceWait
	})
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

func TestHermesRunStartsGatewayWhenPortClosed(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX shell test binary")
	}

	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withHermesPlatform(t, runtime.GOOS)
	t.Setenv("PATH", tmpDir+string(os.PathListSeparator)+os.Getenv("PATH"))

	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := l.Addr().String()
	l.Close()
	withHermesGatewayAddr(t, addr)
	port := strings.TrimPrefix(addr, "127.0.0.1:")

	bin := filepath.Join(tmpDir, "hermes")
	script := "#!/bin/sh\n" +
		"printf '[%s]\\n' \"$*\" >> \"$HOME/hermes-invocations.log\"\n" +
		"if [ \"$1 $2\" = \"gateway start\" ]; then\n" +
		"  exit 1\n" +
		"fi\n" +
		fmt.Sprintf("if [ \"$1 $2\" = \"gateway run\" ]; then\n  exec python3 -c 'import socket,time; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind((\"127.0.0.1\", %s)); s.listen(); time.sleep(5)'\n", port) +
		"fi\n"
	if err := os.WriteFile(bin, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}

	h := &Hermes{}
	stderr := captureStderr(t, func() {
		if err := h.Run("", nil); err != nil {
			t.Fatalf("Run returned error: %v", err)
		}
	})

	if !strings.Contains(stderr, "Starting Hermes gateway") || !strings.Contains(stderr, "Hermes gateway is running") {
		t.Fatalf("expected gateway startup messaging, got %q", stderr)
	}

	data, err := os.ReadFile(filepath.Join(tmpDir, "hermes-invocations.log"))
	if err != nil {
		t.Fatal(err)
	}
	got := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(got) != 3 || got[0] != "[gateway start]" || got[1] != "[gateway run]" || got[2] != "[]" {
		t.Fatalf("expected service-start attempt, manual gateway run, then hermes launch, got %q", got)
	}
}

func TestHermesRunGatewayFailureIncludesManualGuidance(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX shell test binary")
	}

	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withHermesPlatform(t, runtime.GOOS)
	t.Setenv("PATH", tmpDir+string(os.PathListSeparator)+os.Getenv("PATH"))

	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := l.Addr().String()
	l.Close()
	withHermesGatewayAddr(t, addr)

	bin := filepath.Join(tmpDir, "hermes")
	if err := os.WriteFile(bin, []byte("#!/bin/sh\nexit 1\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	h := &Hermes{}
	err = h.Run("", nil)
	if err == nil {
		t.Fatal("expected gateway startup failure")
	}
	if !strings.Contains(err.Error(), "hermes setup gateway") {
		t.Fatalf("expected manual gateway guidance, got %v", err)
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
