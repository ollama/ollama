package launch

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/cmd/internal/fileutil"
)

func withCodexAppPlatform(t *testing.T, goos string) {
	t.Helper()
	old := codexAppGOOS
	codexAppGOOS = goos
	t.Cleanup(func() {
		codexAppGOOS = old
	})
}

func withCodexAppProcessHooks(t *testing.T, isRunning func() bool, quit func() error, open func() error) {
	t.Helper()
	oldIsRunning := codexAppIsRunning
	oldQuit := codexAppQuitApp
	oldOpen := codexAppOpenApp
	oldOpenPath := codexAppOpenPath
	oldOpenStart := codexAppOpenStart
	oldForceQuit := codexAppForceQuit
	oldHasWindow := codexAppHasWindow
	oldRunPath := codexAppRunPath
	oldStartID := codexAppStartID
	oldCanOpenID := codexAppCanOpenID
	oldExitTimeout := codexAppExitTimeout
	oldForceExitTimeout := codexAppForceExitTimeout
	codexAppIsRunning = isRunning
	codexAppHasWindow = isRunning
	codexAppQuitApp = quit
	codexAppOpenApp = open
	t.Cleanup(func() {
		codexAppIsRunning = oldIsRunning
		codexAppQuitApp = oldQuit
		codexAppOpenApp = oldOpen
		codexAppOpenPath = oldOpenPath
		codexAppOpenStart = oldOpenStart
		codexAppForceQuit = oldForceQuit
		codexAppHasWindow = oldHasWindow
		codexAppRunPath = oldRunPath
		codexAppStartID = oldStartID
		codexAppCanOpenID = oldCanOpenID
		codexAppExitTimeout = oldExitTimeout
		codexAppForceExitTimeout = oldForceExitTimeout
	})
}

func TestCodexAppIntegration(t *testing.T) {
	c := &CodexApp{}

	t.Run("implements runner", func(t *testing.T) {
		var _ Runner = c
	})
	t.Run("implements supported integration", func(t *testing.T) {
		var _ SupportedIntegration = c
	})
	t.Run("implements managed single model", func(t *testing.T) {
		var _ ManagedSingleModel = c
	})
	t.Run("receives model list", func(t *testing.T) {
		var _ ManagedModelListConfigurer = c
	})
	t.Run("onboarding is noninteractive", func(t *testing.T) {
		var _ ManagedInteractiveOnboarding = c
		if c.RequiresInteractiveOnboarding() {
			t.Fatal("Codex App onboarding should only mark launch config")
		}
	})
	t.Run("implements restore", func(t *testing.T) {
		var _ RestorableIntegration = c
		var _ RestoreHintIntegration = c
		var _ ConfigurationSuccessIntegration = c
		var _ RestoreSuccessIntegration = c
	})
}

func TestCodexAppSupportedPlatforms(t *testing.T) {
	for _, goos := range []string{"darwin", "windows"} {
		t.Run(goos, func(t *testing.T) {
			withCodexAppPlatform(t, goos)
			if err := codexAppSupported(); err != nil {
				t.Fatalf("codexAppSupported returned error: %v", err)
			}
		})
	}

	t.Run("linux unsupported", func(t *testing.T) {
		withCodexAppPlatform(t, "linux")
		err := codexAppSupported()
		if err == nil || !strings.Contains(err.Error(), "macOS and Windows") {
			t.Fatalf("codexAppSupported error = %v, want platform message", err)
		}
	})
}

func TestCodexAppWindowsAppPathCandidates(t *testing.T) {
	withCodexAppPlatform(t, "windows")
	local := filepath.Join(t.TempDir(), "LocalAppData")
	t.Setenv("LOCALAPPDATA", local)

	exe := filepath.Join(local, "Codex", "app-26.429.30905", "Codex.exe")
	if err := os.MkdirAll(filepath.Dir(exe), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(exe, []byte{}, 0o644); err != nil {
		t.Fatal(err)
	}

	if got := codexAppAppPath(); got != exe {
		t.Fatalf("codexAppAppPath = %q, want %q", got, exe)
	}
}

func TestCodexAppInstalledUsesWindowsStartMenuFallback(t *testing.T) {
	withCodexAppPlatform(t, "windows")
	t.Setenv("LOCALAPPDATA", filepath.Join(t.TempDir(), "LocalAppData"))

	oldStartID := codexAppStartID
	oldIsRunning := codexAppIsRunning
	codexAppStartID = func() string { return "OpenAI.Codex_12345!App" }
	codexAppIsRunning = func() bool { return false }
	t.Cleanup(func() {
		codexAppStartID = oldStartID
		codexAppIsRunning = oldIsRunning
	})

	if !codexAppInstalled() {
		t.Fatal("expected Windows Start menu app id to count as installed")
	}
}

func TestCodexAppInstalledUsesMacBundleIDFallback(t *testing.T) {
	withCodexAppPlatform(t, "darwin")

	oldCanOpenID := codexAppCanOpenID
	oldStat := codexAppStat
	codexAppCanOpenID = func() bool { return true }
	codexAppStat = func(string) (os.FileInfo, error) { return nil, os.ErrNotExist }
	t.Cleanup(func() {
		codexAppCanOpenID = oldCanOpenID
		codexAppStat = oldStat
	})

	if !codexAppInstalled() {
		t.Fatal("expected macOS LaunchServices bundle id fallback to count as installed")
	}
}

func TestCodexAppConfigureActivatesOllamaProfile(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:9999")

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	existing := "" +
		"profile = \"default\"\n" +
		"model = \"gpt-5.5\"\n\n" +
		"[profiles.default]\n" +
		"model = \"gpt-5.5\"\n"
	if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}

	c := &CodexApp{}
	if err := c.ConfigureWithModels("llama3.2", []string{"llama3.2", "qwen3:8b"}); err != nil {
		t.Fatalf("ConfigureWithModels returned error: %v", err)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	content := string(data)
	catalogPath, err := codexAppModelCatalogPath()
	if err != nil {
		t.Fatal(err)
	}

	for _, want := range []string{
		fmt.Sprintf(`profile = %q`, codexAppProfileName),
		`model = "llama3.2"`,
		fmt.Sprintf(`model_provider = %q`, codexAppProfileName),
		fmt.Sprintf(`model_catalog_json = %q`, catalogPath),
		codexProfileHeaderFor(codexAppProfileName),
		`model = "llama3.2"`,
		`openai_base_url = "http://127.0.0.1:9999/v1/"`,
		fmt.Sprintf(`model_provider = %q`, codexAppProfileName),
		`model_catalog_json = "`,
		codexProviderHeaderFor(codexAppProfileName),
		`name = "Ollama"`,
		`base_url = "http://127.0.0.1:9999/v1/"`,
		`wire_api = "responses"`,
		`[profiles.default]`,
	} {
		if !strings.Contains(content, want) {
			t.Fatalf("expected config to contain %q, got:\n%s", want, content)
		}
	}
	if got := c.CurrentModel(); got != "llama3.2" {
		t.Fatalf("CurrentModel = %q, want llama3.2", got)
	}

	restoreData, err := os.ReadFile(codexAppRestoreStatePath())
	if err != nil {
		t.Fatalf("expected restore state: %v", err)
	}
	if !strings.Contains(string(restoreData), `"profile": "default"`) {
		t.Fatalf("expected restore state to remember default profile, got %s", restoreData)
	}
	catalogData, err := os.ReadFile(catalogPath)
	if err != nil {
		t.Fatalf("expected model catalog: %v", err)
	}
	var catalog struct {
		Models []map[string]any `json:"models"`
	}
	if err := json.Unmarshal(catalogData, &catalog); err != nil {
		t.Fatalf("catalog should be valid JSON: %v", err)
	}
	if got := catalogSlugs(catalog.Models); strings.Join(got, ",") != "llama3.2,qwen3:8b" {
		t.Fatalf("catalog slugs = %v, want fallback models", got)
	}
}

func TestCodexAppConfigureUsesAppSpecificProfileWithoutTouchingCLIProfile(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:9999")

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	existing := "" +
		`profile = "default"` + "\n\n" +
		"[profiles.ollama-launch]\n" +
		`model = "cli-model"` + "\n" +
		`openai_base_url = "http://cli.invalid/v1/"` + "\n" +
		`model_provider = "ollama-launch"` + "\n\n" +
		"[model_providers.ollama-launch]\n" +
		`name = "CLI Ollama"` + "\n" +
		`base_url = "http://cli.invalid/v1/"` + "\n" +
		`wire_api = "responses"` + "\n\n" +
		"[profiles.default]\n" +
		`model = "gpt-5.5"` + "\n"
	if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := (&CodexApp{}).ConfigureWithModels("llama3.2", []string{"llama3.2"}); err != nil {
		t.Fatalf("ConfigureWithModels returned error: %v", err)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	content := string(data)
	if got := codexRootStringValue(content, "profile"); got != codexAppProfileName {
		t.Fatalf("root profile = %q, want %q", got, codexAppProfileName)
	}
	if got := codexSectionStringValue(content, codexProfileHeader(), "openai_base_url"); got != "http://cli.invalid/v1/" {
		t.Fatalf("CLI profile base URL = %q, want preserved CLI URL in:\n%s", got, content)
	}
	if got := codexSectionStringValue(content, codexProviderHeader(), "name"); got != "CLI Ollama" {
		t.Fatalf("CLI provider name = %q, want preserved CLI provider in:\n%s", got, content)
	}
	if got := codexSectionStringValue(content, codexProfileHeaderFor(codexAppProfileName), "model"); got != "llama3.2" {
		t.Fatalf("app profile model = %q, want llama3.2", got)
	}
	if got := codexSectionStringValue(content, codexProviderHeaderFor(codexAppProfileName), "base_url"); got != "http://127.0.0.1:9999/v1/" {
		t.Fatalf("app provider base URL = %q", got)
	}
	assertBackupContains(t, filepath.Join(fileutil.BackupDir(), codexAppIntegrationName, "config.toml.*"), `profile = "default"`)
}

func TestCodexAppConfigureUsesConnectableHostForUnspecifiedBindAddress(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	t.Setenv("OLLAMA_HOST", "http://0.0.0.0:11434")

	if err := (&CodexApp{}).ConfigureWithModels("llama3.2", []string{"llama3.2"}); err != nil {
		t.Fatalf("ConfigureWithModels returned error: %v", err)
	}

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	content := string(data)
	if strings.Contains(content, "0.0.0.0") {
		t.Fatalf("config should not write bind-only host, got:\n%s", content)
	}
	if got := codexSectionStringValue(content, codexProfileHeaderFor(codexAppProfileName), "openai_base_url"); got != "http://127.0.0.1:11434/v1/" {
		t.Fatalf("app profile openai_base_url = %q, want connectable loopback URL", got)
	}
	if got := codexSectionStringValue(content, codexProviderHeaderFor(codexAppProfileName), "base_url"); got != "http://127.0.0.1:11434/v1/" {
		t.Fatalf("app provider base_url = %q, want connectable loopback URL", got)
	}
	if got := codexRootStringValue(content, "model_provider"); got != codexAppProfileName {
		t.Fatalf("root model_provider = %q, want %q", got, codexAppProfileName)
	}
}

func TestCodexAppConfigureRejectsMalformedTomlBeforeSideEffects(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	existing := "profile = \n"
	if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}

	err := (&CodexApp{}).ConfigureWithModels("llama3.2", []string{"llama3.2"})
	if err == nil || !strings.Contains(err.Error(), "invalid Codex config TOML") {
		t.Fatalf("ConfigureWithModels error = %v, want invalid TOML", err)
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != existing {
		t.Fatalf("malformed config should be left untouched, got:\n%s", data)
	}
	if _, err := os.Stat(codexAppRestoreStatePath()); !os.IsNotExist(err) {
		t.Fatalf("restore state should not be written before config validation, err=%v", err)
	}
	catalogPath, err := codexAppModelCatalogPath()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(catalogPath); !os.IsNotExist(err) {
		t.Fatalf("model catalog should not be written before config validation, err=%v", err)
	}
}

func TestCodexAppConfigureRejectsMalformedTomlEvenWithExistingRestoreState(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	existing := "[profiles.ollama-launch\nmodel = \"llama3.2\"\n"
	if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(codexAppRestoreStatePath()), 0o755); err != nil {
		t.Fatal(err)
	}
	restoreState := `{"had_profile":true,"profile":"default","had_model":true,"model":"gpt-5.5","had_model_provider":true,"model_provider":"openai","had_model_catalog_json":false}`
	if err := os.WriteFile(codexAppRestoreStatePath(), []byte(restoreState), 0o644); err != nil {
		t.Fatal(err)
	}

	err := (&CodexApp{}).ConfigureWithModels("llama3.2", []string{"llama3.2"})
	if err == nil || !strings.Contains(err.Error(), "invalid Codex config TOML") {
		t.Fatalf("ConfigureWithModels error = %v, want invalid TOML", err)
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != existing {
		t.Fatalf("malformed config should be left untouched, got:\n%s", data)
	}
	stateData, err := os.ReadFile(codexAppRestoreStatePath())
	if err != nil {
		t.Fatal(err)
	}
	if string(stateData) != restoreState {
		t.Fatalf("restore state should be left untouched, got:\n%s", stateData)
	}
}

func TestCodexAppCurrentModelRequiresManagedActiveProfile(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	content := "" +
		"profile = \"default\"\n\n" +
		codexProfileHeaderFor(codexAppProfileName) + "\n" +
		"model = \"llama3.2\"\n" +
		fmt.Sprintf("model_provider = %q\n\n", codexAppProfileName) +
		codexProviderHeaderFor(codexAppProfileName) + "\n" +
		"base_url = \"http://127.0.0.1:11434/v1/\"\n"
	if err := os.WriteFile(configPath, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	if got := (&CodexApp{}).CurrentModel(); got != "" {
		t.Fatalf("CurrentModel = %q, want empty when active profile is not managed", got)
	}
}

func TestCodexAppCurrentModelReadsManagedRootConfig(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	content := "" +
		`model = "qwen3:8b"` + "\n" +
		fmt.Sprintf(`model_provider = %q`, codexAppProfileName) + "\n\n" +
		fmt.Sprintf(`model_catalog_json = %q`, mustWriteCodexAppTestCatalog(t, "qwen3:8b")) + "\n\n" +
		codexProfileHeaderFor(codexAppProfileName) + "\n" +
		fmt.Sprintf(`model_provider = %q`, codexAppProfileName) + "\n" +
		fmt.Sprintf(`model_catalog_json = %q`, mustCodexAppModelCatalogPath(t)) + "\n\n" +
		codexProviderHeaderFor(codexAppProfileName) + "\n" +
		`base_url = "http://127.0.0.1:11434/v1/"` + "\n"
	if err := os.WriteFile(configPath, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	if got := (&CodexApp{}).CurrentModel(); got != "qwen3:8b" {
		t.Fatalf("CurrentModel = %q, want qwen3:8b", got)
	}
}

func TestCodexAppCurrentModelRequiresHealthyCatalog(t *testing.T) {
	for _, tt := range []struct {
		name           string
		rootCatalog    bool
		profileCatalog bool
		writeCatalog   bool
		catalogData    string
	}{
		{
			name:           "missing catalog reference",
			rootCatalog:    false,
			profileCatalog: true,
			writeCatalog:   true,
			catalogData:    `{"models":[{"slug":"llama3.2"}]}`,
		},
		{
			name:           "deleted catalog file",
			rootCatalog:    true,
			profileCatalog: true,
			writeCatalog:   false,
			catalogData:    `{"models":[{"slug":"llama3.2"}]}`,
		},
		{
			name:           "missing profile catalog reference",
			rootCatalog:    true,
			profileCatalog: false,
			writeCatalog:   true,
			catalogData:    `{"models":[{"slug":"llama3.2"}]}`,
		},
		{
			name:           "corrupt catalog file",
			rootCatalog:    true,
			profileCatalog: true,
			writeCatalog:   true,
			catalogData:    `{"models":`,
		},
		{
			name:           "empty catalog",
			rootCatalog:    true,
			profileCatalog: true,
			writeCatalog:   true,
			catalogData:    `{"models":[]}`,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			tmpDir := t.TempDir()
			setTestHome(t, tmpDir)
			t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")

			configPath := filepath.Join(tmpDir, ".codex", "config.toml")
			if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
				t.Fatal(err)
			}
			catalogPath := mustCodexAppModelCatalogPath(t)
			if tt.writeCatalog {
				if err := os.WriteFile(catalogPath, []byte(tt.catalogData), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			var rootCatalogLine, profileCatalogLine string
			if tt.rootCatalog {
				rootCatalogLine = fmt.Sprintf(`model_catalog_json = %q`, catalogPath) + "\n"
			}
			if tt.profileCatalog {
				profileCatalogLine = fmt.Sprintf(`model_catalog_json = %q`, catalogPath) + "\n"
			}
			content := "" +
				`model = "llama3.2"` + "\n" +
				fmt.Sprintf(`model_provider = %q`, codexAppProfileName) + "\n" +
				rootCatalogLine + "\n" +
				codexProfileHeaderFor(codexAppProfileName) + "\n" +
				fmt.Sprintf(`model_provider = %q`, codexAppProfileName) + "\n" +
				profileCatalogLine + "\n" +
				codexProviderHeaderFor(codexAppProfileName) + "\n" +
				`base_url = "http://127.0.0.1:11434/v1/"` + "\n"
			if err := os.WriteFile(configPath, []byte(content), 0o644); err != nil {
				t.Fatal(err)
			}

			if got := (&CodexApp{}).CurrentModel(); got != "" {
				t.Fatalf("CurrentModel = %q, want empty when catalog is unhealthy", got)
			}
		})
	}
}

func TestCodexAppConfigurePopulatesCatalogFromTagsAndShow(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	showCalls := make(map[string]int)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"gemma4"},{"name":"qwen3:8b"},{"name":"llama3.2"}]}`)
		case "/api/show":
			var req struct {
				Model string `json:"model"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Fatalf("decode show request: %v", err)
			}
			showCalls[req.Model]++
			fmt.Fprintf(w, `{"model_info":{"general.context_length":%d},"capabilities":["vision"]}`, 65536+len(req.Model))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	if err := (&CodexApp{}).ConfigureWithModels("gemma4", []string{"fallback"}); err != nil {
		t.Fatalf("ConfigureWithModels returned error: %v", err)
	}

	catalogPath, err := codexAppModelCatalogPath()
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(catalogPath)
	if err != nil {
		t.Fatal(err)
	}
	var catalog struct {
		Models []map[string]any `json:"models"`
	}
	if err := json.Unmarshal(data, &catalog); err != nil {
		t.Fatalf("catalog should be valid JSON: %v", err)
	}

	if got := catalogSlugs(catalog.Models); strings.Join(got, ",") != "gemma4,qwen3:8b,llama3.2" {
		t.Fatalf("catalog slugs = %v, want /api/tags models", got)
	}
	for _, model := range catalog.Models {
		slug, _ := model["slug"].(string)
		if model["display_name"] != slug {
			t.Fatalf("display_name should match slug for %q: %v", slug, model["display_name"])
		}
		if model["visibility"] != "list" {
			t.Fatalf("visibility for %q = %v, want list", slug, model["visibility"])
		}
		if model["default_reasoning_level"] != nil {
			t.Fatalf("default_reasoning_level for %q = %v, want nil", slug, model["default_reasoning_level"])
		}
		levels, ok := model["supported_reasoning_levels"].([]any)
		if !ok || len(levels) != 0 {
			t.Fatalf("supported_reasoning_levels for %q = %v, want empty list", slug, model["supported_reasoning_levels"])
		}
		wantContext := float64(128000)
		wantModalities := []string{"text"}
		wantShowCalls := 0
		if slug == "gemma4" {
			wantContext = float64(65536 + len(slug))
			wantModalities = []string{"text", "image"}
			wantShowCalls = 1
		}
		if model["context_window"] != wantContext {
			t.Fatalf("context_window for %q = %v, want %v", slug, model["context_window"], wantContext)
		}
		if got := catalogInputModalities(model); strings.Join(got, ",") != strings.Join(wantModalities, ",") {
			t.Fatalf("input_modalities for %q = %v, want %v", slug, got, wantModalities)
		}
		if showCalls[slug] != wantShowCalls {
			t.Fatalf("show calls for %q = %d, want %d", slug, showCalls[slug], wantShowCalls)
		}
	}
}

func TestCodexAppConfigureUpgradesLegacyRestoreState(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:9999")

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	existing := "" +
		`model = "gpt-5.5"` + "\n" +
		`model_provider = "odc-resp-dev"` + "\n\n" +
		"[model_providers.odc-resp-dev]\n" +
		`base_url = "https://example.invalid/v1/"` + "\n"
	if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(codexAppRestoreStatePath()), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(codexAppRestoreStatePath(), []byte(`{"had_profile":false}`), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := (&CodexApp{}).ConfigureWithModels("llama3.2", []string{"llama3.2"}); err != nil {
		t.Fatalf("ConfigureWithModels returned error: %v", err)
	}

	state, err := loadCodexAppRestoreState()
	if err != nil {
		t.Fatal(err)
	}
	if state.HadProfile {
		t.Fatalf("HadProfile = true, want legacy false")
	}
	if !state.HadModel || state.Model != "gpt-5.5" {
		t.Fatalf("model restore state = (%v, %q), want previous root model", state.HadModel, state.Model)
	}
	if !state.HadModelProvider || state.ModelProvider != "odc-resp-dev" {
		t.Fatalf("model provider restore state = (%v, %q), want previous root provider", state.HadModelProvider, state.ModelProvider)
	}
}

func TestCodexAppRestoreRestoresPreviousProfile(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withCodexAppPlatform(t, "darwin")

	var openCalls int
	withCodexAppProcessHooks(t,
		func() bool { return false },
		func() error { return nil },
		func() error {
			openCalls++
			return nil
		},
	)

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	existing := "" +
		"profile = \"default\"\n" +
		"model = \"gpt-5.5\"\n" +
		"model_provider = \"openai\"\n" +
		"model_catalog_json = \"/tmp/original-catalog.json\"\n\n" +
		"[profiles.default]\n" +
		"model = \"gpt-5.5\"\n"
	if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}

	c := &CodexApp{}
	if err := c.ConfigureWithModels("llama3.2", []string{"llama3.2"}); err != nil {
		t.Fatalf("ConfigureWithModels returned error: %v", err)
	}
	if err := c.Restore(); err != nil {
		t.Fatalf("Restore returned error: %v", err)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `profile = "default"`) || strings.Contains(string(data), fmt.Sprintf(`profile = %q`, codexAppProfileName)) {
		t.Fatalf("restore should restore previous active profile, got:\n%s", data)
	}
	restored := string(data)
	if strings.Contains(restored, codexProfileHeaderFor(codexAppProfileName)) || strings.Contains(restored, codexProviderHeaderFor(codexAppProfileName)) {
		t.Fatalf("restore should remove owned app sections, got:\n%s", restored)
	}
	for key, want := range map[string]string{
		"profile":            "default",
		"model":              "gpt-5.5",
		"model_provider":     "openai",
		"model_catalog_json": "/tmp/original-catalog.json",
	} {
		if got := codexRootStringValue(restored, key); got != want {
			t.Fatalf("root %s = %q, want %q in:\n%s", key, got, want, restored)
		}
	}
	if openCalls != 1 {
		t.Fatalf("open calls = %d, want 1", openCalls)
	}
	if _, err := os.Stat(codexAppRestoreStatePath()); !os.IsNotExist(err) {
		t.Fatalf("restore state should be removed, got err=%v", err)
	}
}

func TestCodexAppRestoreMissingConfigRemovesRestoreState(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withCodexAppPlatform(t, "darwin")

	var openCalls int
	withCodexAppProcessHooks(t,
		func() bool { return false },
		func() error { return nil },
		func() error {
			openCalls++
			return nil
		},
	)

	if err := os.MkdirAll(filepath.Dir(codexAppRestoreStatePath()), 0o755); err != nil {
		t.Fatal(err)
	}
	restoreState := `{"had_profile":true,"profile":"stale","had_model":true,"model":"old","had_model_provider":true,"model_provider":"openai","had_model_catalog_json":false}`
	if err := os.WriteFile(codexAppRestoreStatePath(), []byte(restoreState), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := (&CodexApp{}).Restore(); err != nil {
		t.Fatalf("Restore returned error: %v", err)
	}

	if _, err := os.Stat(codexAppRestoreStatePath()); !os.IsNotExist(err) {
		t.Fatalf("restore state should be removed when config is missing, got err=%v", err)
	}
	if openCalls != 1 {
		t.Fatalf("open calls = %d, want 1", openCalls)
	}
}

func TestCodexAppConfigureMissingConfigReplacesStaleRestoreState(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:9999")

	if err := os.MkdirAll(filepath.Dir(codexAppRestoreStatePath()), 0o755); err != nil {
		t.Fatal(err)
	}
	restoreState := `{"had_profile":true,"profile":"stale","had_model":true,"model":"old","had_model_provider":true,"model_provider":"openai","had_model_catalog_json":false}`
	if err := os.WriteFile(codexAppRestoreStatePath(), []byte(restoreState), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := (&CodexApp{}).ConfigureWithModels("llama3.2", []string{"llama3.2"}); err != nil {
		t.Fatalf("ConfigureWithModels returned error: %v", err)
	}

	state, err := loadCodexAppRestoreState()
	if err != nil {
		t.Fatal(err)
	}
	if state.HadProfile || state.HadModel || state.HadModelProvider || state.HadModelCatalogJSON {
		t.Fatalf("restore state = %+v, want empty snapshot when config was missing", state)
	}
}

func TestCodexAppConfigureRefreshesRestoreStateAfterManualProfileSwitch(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:9999")
	withCodexAppPlatform(t, "darwin")

	var openCalls int
	withCodexAppProcessHooks(t,
		func() bool { return false },
		func() error { return nil },
		func() error {
			openCalls++
			return nil
		},
	)

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	initial := "" +
		`profile = "default"` + "\n" +
		`model = "gpt-5.5"` + "\n" +
		`model_provider = "openai"` + "\n\n" +
		"[profiles.default]\n" +
		`model = "gpt-5.5"` + "\n"
	if err := os.WriteFile(configPath, []byte(initial), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := (&CodexApp{}).ConfigureWithModels("llama3.2", []string{"llama3.2"}); err != nil {
		t.Fatalf("first ConfigureWithModels returned error: %v", err)
	}

	manual := "" +
		`profile = "manual"` + "\n" +
		`model = "manual-model"` + "\n" +
		`model_provider = "openai"` + "\n\n" +
		"[profiles.manual]\n" +
		`model = "manual-model"` + "\n"
	if err := os.WriteFile(configPath, []byte(manual), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := (&CodexApp{}).ConfigureWithModels("qwen3:8b", []string{"qwen3:8b"}); err != nil {
		t.Fatalf("second ConfigureWithModels returned error: %v", err)
	}
	if err := (&CodexApp{}).Restore(); err != nil {
		t.Fatalf("Restore returned error: %v", err)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	restored := string(data)
	for key, want := range map[string]string{
		"profile":        "manual",
		"model":          "manual-model",
		"model_provider": "openai",
	} {
		if got := codexRootStringValue(restored, key); got != want {
			t.Fatalf("root %s = %q, want %q in:\n%s", key, got, want, restored)
		}
	}
	if openCalls != 1 {
		t.Fatalf("open calls = %d, want 1", openCalls)
	}
}

func TestCodexAppRestoreRejectsMalformedTomlWithoutWriting(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withCodexAppPlatform(t, "darwin")

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	existing := "model = \"unterminated\n"
	if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(codexAppRestoreStatePath()), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(codexAppRestoreStatePath(), []byte(`{"had_profile":false,"had_model":false,"had_model_provider":false,"had_model_catalog_json":false}`), 0o644); err != nil {
		t.Fatal(err)
	}

	err := (&CodexApp{}).Restore()
	if err == nil || !strings.Contains(err.Error(), "invalid Codex config TOML") {
		t.Fatalf("Restore error = %v, want invalid TOML", err)
	}
	catalogPath, pathErr := codexAppModelCatalogPath()
	if pathErr != nil {
		t.Fatal(pathErr)
	}
	for _, want := range []string{
		"Restore did not complete",
		"Codex config: " + configPath,
		"Restore state: " + codexAppRestoreStatePath(),
		"Model catalog: " + catalogPath,
		"Backups: " + filepath.Join(fileutil.BackupDir(), codexAppIntegrationName),
	} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("Restore error missing %q:\n%v", want, err)
		}
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != existing {
		t.Fatalf("malformed config should be left untouched, got:\n%s", data)
	}
	if _, err := os.Stat(codexAppRestoreStatePath()); err != nil {
		t.Fatalf("restore state should remain after failed restore: %v", err)
	}
}

func TestCodexAppRestoreWithoutStateRemovesManagedRootModel(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withCodexAppPlatform(t, "darwin")

	var openCalls int
	withCodexAppProcessHooks(t,
		func() bool { return false },
		func() error { return nil },
		func() error {
			openCalls++
			return nil
		},
	)

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	catalogPath, err := codexAppModelCatalogPath()
	if err != nil {
		t.Fatal(err)
	}
	existing := "" +
		fmt.Sprintf(`profile = %q`, codexAppProfileName) + "\n" +
		`model = "llama3.2"` + "\n" +
		fmt.Sprintf(`model_provider = %q`, codexAppProfileName) + "\n" +
		fmt.Sprintf(`model_catalog_json = %q`, catalogPath) + "\n\n" +
		codexProfileHeaderFor(codexAppProfileName) + "\n" +
		`model = "llama3.2"` + "\n" +
		fmt.Sprintf(`model_provider = %q`, codexAppProfileName) + "\n\n" +
		codexProviderHeaderFor(codexAppProfileName) + "\n" +
		`base_url = "http://127.0.0.1:11434/v1/"` + "\n"
	if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(catalogPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(catalogPath, []byte(`{"models":[]}`), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := (&CodexApp{}).Restore(); err != nil {
		t.Fatalf("Restore returned error: %v", err)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	content := string(data)
	for _, key := range []string{"profile", "model", "model_provider", "model_catalog_json"} {
		if got, ok := codexRootStringValueOK(content, key); ok {
			t.Fatalf("root %s should be removed, got %q in:\n%s", key, got, content)
		}
	}
	if strings.Contains(content, codexProfileHeaderFor(codexAppProfileName)) || strings.Contains(content, codexProviderHeaderFor(codexAppProfileName)) {
		t.Fatalf("owned app sections should be removed, got:\n%s", content)
	}
	if _, err := os.Stat(catalogPath); !os.IsNotExist(err) {
		t.Fatalf("owned catalog should be removed when unused, err=%v", err)
	}
	if openCalls != 1 {
		t.Fatalf("open calls = %d, want 1", openCalls)
	}
}

func TestCodexAppRestoreDoesNotStompUserChangedRootConfig(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withCodexAppPlatform(t, "darwin")

	var openCalls int
	withCodexAppProcessHooks(t,
		func() bool { return false },
		func() error { return nil },
		func() error {
			openCalls++
			return nil
		},
	)

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	catalogPath, err := codexAppModelCatalogPath()
	if err != nil {
		t.Fatal(err)
	}
	existing := "" +
		`profile = "manual"` + "\n" +
		`model = "gpt-5.5"` + "\n" +
		`model_provider = "openai"` + "\n\n" +
		codexProfileHeaderFor(codexAppProfileName) + "\n" +
		`model = "llama3.2"` + "\n" +
		fmt.Sprintf(`model_catalog_json = %q`, catalogPath) + "\n\n" +
		codexProviderHeaderFor(codexAppProfileName) + "\n" +
		`base_url = "http://127.0.0.1:11434/v1/"` + "\n\n" +
		"[profiles.manual]\n" +
		`model = "gpt-5.5"` + "\n"
	if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(catalogPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(catalogPath, []byte(`{"models":[]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(codexAppRestoreStatePath()), 0o755); err != nil {
		t.Fatal(err)
	}
	restoreState := `{"had_profile":true,"profile":"default","had_model":true,"model":"old","had_model_provider":true,"model_provider":"old-provider","had_model_catalog_json":false}`
	if err := os.WriteFile(codexAppRestoreStatePath(), []byte(restoreState), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := (&CodexApp{}).Restore(); err != nil {
		t.Fatalf("Restore returned error: %v", err)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	content := string(data)
	for key, want := range map[string]string{
		"profile":        "manual",
		"model":          "gpt-5.5",
		"model_provider": "openai",
	} {
		if got := codexRootStringValue(content, key); got != want {
			t.Fatalf("root %s = %q, want %q in:\n%s", key, got, want, content)
		}
	}
	if strings.Contains(content, codexProfileHeaderFor(codexAppProfileName)) || strings.Contains(content, codexProviderHeaderFor(codexAppProfileName)) {
		t.Fatalf("owned app sections should be removed when no longer active, got:\n%s", content)
	}
	if _, err := os.Stat(catalogPath); !os.IsNotExist(err) {
		t.Fatalf("owned catalog should be removed when unused, err=%v", err)
	}
	if openCalls != 1 {
		t.Fatalf("open calls = %d, want 1", openCalls)
	}
}

func TestCodexAppRestoreDoesNotTreatCLIProfileAsOwned(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withCodexAppPlatform(t, "darwin")

	withCodexAppProcessHooks(t,
		func() bool { return false },
		func() error { return nil },
		func() error { return nil },
	)

	configPath := filepath.Join(tmpDir, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		t.Fatal(err)
	}
	existing := "" +
		`profile = "ollama-launch"` + "\n" +
		`model = "cli-model"` + "\n" +
		`model_provider = "ollama-launch"` + "\n\n" +
		"[profiles.ollama-launch]\n" +
		`model = "cli-model"` + "\n" +
		`openai_base_url = "http://cli.invalid/v1/"` + "\n" +
		`model_provider = "ollama-launch"` + "\n\n" +
		"[model_providers.ollama-launch]\n" +
		`name = "CLI Ollama"` + "\n" +
		`base_url = "http://cli.invalid/v1/"` + "\n" +
		`wire_api = "responses"` + "\n"
	if err := os.WriteFile(configPath, []byte(existing), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(codexAppRestoreStatePath()), 0o755); err != nil {
		t.Fatal(err)
	}
	restoreState := `{"had_profile":true,"profile":"default","had_model":true,"model":"gpt-5.5","had_model_provider":true,"model_provider":"openai","had_model_catalog_json":false}`
	if err := os.WriteFile(codexAppRestoreStatePath(), []byte(restoreState), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := (&CodexApp{}).Restore(); err != nil {
		t.Fatalf("Restore returned error: %v", err)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != existing {
		t.Fatalf("CLI Codex profile should be left untouched, got:\n%s", data)
	}
}

func TestCodexAppRunRestartsRunningAppWhenConfirmed(t *testing.T) {
	withCodexAppPlatform(t, "darwin")
	restoreConfirm := withLaunchConfirmPolicy(launchConfirmPolicy{yes: true})
	defer restoreConfirm()

	running := true
	var quitCalls, openCalls int
	withCodexAppProcessHooks(t,
		func() bool { return running },
		func() error {
			quitCalls++
			running = false
			return nil
		},
		func() error {
			openCalls++
			return nil
		},
	)

	if err := (&CodexApp{}).Run("qwen3.5", nil); err != nil {
		t.Fatalf("Run returned error: %v", err)
	}
	if quitCalls != 1 || openCalls != 1 {
		t.Fatalf("quit/open calls = %d/%d, want 1/1", quitCalls, openCalls)
	}
}

func TestCodexAppRunWaitsForGracefulExitBeforeReopening(t *testing.T) {
	withCodexAppPlatform(t, "darwin")
	restoreConfirm := withLaunchConfirmPolicy(launchConfirmPolicy{yes: true})
	defer restoreConfirm()

	oldSleep := codexAppSleep
	t.Cleanup(func() {
		codexAppSleep = oldSleep
	})

	running := true
	var quitCalls, openCalls, sleepCalls int
	codexAppSleep = func(time.Duration) {
		sleepCalls++
		if sleepCalls == 2 {
			running = false
		}
	}
	withCodexAppProcessHooks(t,
		func() bool { return running },
		func() error {
			quitCalls++
			return nil
		},
		func() error {
			openCalls++
			return nil
		},
	)

	if err := (&CodexApp{}).Run("qwen3.5", nil); err != nil {
		t.Fatalf("Run returned error: %v", err)
	}
	if quitCalls != 1 || openCalls != 1 {
		t.Fatalf("quit/open calls = %d/%d, want 1/1", quitCalls, openCalls)
	}
	if sleepCalls == 0 {
		t.Fatal("expected restart to wait for Codex to exit before reopening")
	}
}

func TestCodexAppRunForceStopsMacAfterGracefulTimeout(t *testing.T) {
	withCodexAppPlatform(t, "darwin")
	restoreConfirm := withLaunchConfirmPolicy(launchConfirmPolicy{yes: true})
	defer restoreConfirm()

	running := true
	calls := make([]string, 0)
	withCodexAppProcessHooks(t,
		func() bool { return running },
		func() error {
			calls = append(calls, "quit")
			return nil
		},
		func() error {
			calls = append(calls, "open")
			return nil
		},
	)
	codexAppExitTimeout = 0
	codexAppForceQuit = func() error {
		calls = append(calls, "force")
		running = false
		return nil
	}

	if err := (&CodexApp{}).Run("qwen3.5", nil); err != nil {
		t.Fatalf("Run returned error: %v", err)
	}
	want := []string{"quit", "force", "open"}
	if strings.Join(calls, ",") != strings.Join(want, ",") {
		t.Fatalf("calls = %v, want %v", calls, want)
	}
}

func TestCodexAppRunReturnsMacForceStopError(t *testing.T) {
	withCodexAppPlatform(t, "darwin")
	restoreConfirm := withLaunchConfirmPolicy(launchConfirmPolicy{yes: true})
	defer restoreConfirm()

	withCodexAppProcessHooks(t,
		func() bool { return true },
		func() error { return nil },
		func() error {
			t.Fatal("app should not reopen when force stop fails")
			return nil
		},
	)
	codexAppExitTimeout = 0
	codexAppForceQuit = func() error {
		return fmt.Errorf("operation not permitted")
	}

	err := (&CodexApp{}).Run("qwen3.5", nil)
	if err == nil || !strings.Contains(err.Error(), "force stop Codex") || !strings.Contains(err.Error(), "operation not permitted") {
		t.Fatalf("Run error = %v, want force stop failure", err)
	}
}

func TestCodexAppRunOpensOnWindowsWhenNotRunning(t *testing.T) {
	withCodexAppPlatform(t, "windows")

	var openCalls int
	withCodexAppProcessHooks(t,
		func() bool { return false },
		func() error { return nil },
		func() error {
			openCalls++
			return nil
		},
	)

	if err := (&CodexApp{}).Run("qwen3.5", nil); err != nil {
		t.Fatalf("Run returned error: %v", err)
	}
	if openCalls != 1 {
		t.Fatalf("open calls = %d, want 1", openCalls)
	}
}

func TestCodexAppRunRestartsWindowsStartAppID(t *testing.T) {
	withCodexAppPlatform(t, "windows")
	restoreConfirm := withLaunchConfirmPolicy(launchConfirmPolicy{yes: true})
	defer restoreConfirm()

	running := true
	var quitCalls int
	withCodexAppProcessHooks(t,
		func() bool { return running },
		func() error {
			quitCalls++
			running = false
			return nil
		},
		func() error {
			t.Fatal("open app fallback should not be used")
			return nil
		},
	)

	codexAppStartID = func() string { return "OpenAI.Codex_2p2nqsd0c76g0!App" }
	codexAppRunPath = func() string {
		return `C:\Program Files\WindowsApps\OpenAI.Codex_26.429.8261.0_x64__2p2nqsd0c76g0\app\Codex.exe`
	}
	var openedStartID, openedPath string
	codexAppOpenStart = func(appID string) error {
		openedStartID = appID
		return nil
	}
	codexAppOpenPath = func(path string) error {
		openedPath = path
		return nil
	}

	if err := (&CodexApp{}).Run("qwen3.5", nil); err != nil {
		t.Fatalf("Run returned error: %v", err)
	}
	if quitCalls != 1 {
		t.Fatalf("quit calls = %d, want 1", quitCalls)
	}
	if openedStartID != "OpenAI.Codex_2p2nqsd0c76g0!App" {
		t.Fatalf("opened Start AppID = %q", openedStartID)
	}
	if openedPath != "" {
		t.Fatalf("opened path = %q, want Start AppID path only", openedPath)
	}
}

func TestCodexAppRunForceStopsWindowsBackgroundProcessesBeforeReopening(t *testing.T) {
	withCodexAppPlatform(t, "windows")
	restoreConfirm := withLaunchConfirmPolicy(launchConfirmPolicy{yes: true})
	defer restoreConfirm()

	windowOpen := true
	running := true
	calls := make([]string, 0)
	withCodexAppProcessHooks(t,
		func() bool { return running },
		func() error {
			calls = append(calls, "quit")
			windowOpen = false
			return nil
		},
		func() error {
			t.Fatal("open app fallback should not be used")
			return nil
		},
	)
	codexAppHasWindow = func() bool { return windowOpen }
	codexAppForceQuit = func() error {
		calls = append(calls, "force")
		running = false
		return nil
	}
	codexAppStartID = func() string { return "OpenAI.Codex_2p2nqsd0c76g0!App" }
	codexAppOpenStart = func(appID string) error {
		calls = append(calls, "open:"+appID)
		return nil
	}

	if err := (&CodexApp{}).Run("qwen3.5", nil); err != nil {
		t.Fatalf("Run returned error: %v", err)
	}
	want := []string{"quit", "force", "open:OpenAI.Codex_2p2nqsd0c76g0!App"}
	if strings.Join(calls, ",") != strings.Join(want, ",") {
		t.Fatalf("calls = %v, want %v", calls, want)
	}
}

func TestCodexAppRunReturnsWindowsForceStopError(t *testing.T) {
	withCodexAppPlatform(t, "windows")
	restoreConfirm := withLaunchConfirmPolicy(launchConfirmPolicy{yes: true})
	defer restoreConfirm()

	windowOpen := true
	withCodexAppProcessHooks(t,
		func() bool { return true },
		func() error {
			windowOpen = false
			return nil
		},
		func() error {
			t.Fatal("open app fallback should not be used")
			return nil
		},
	)
	codexAppHasWindow = func() bool { return windowOpen }
	codexAppForceQuit = func() error {
		return fmt.Errorf("access denied")
	}
	codexAppOpenStart = func(string) error {
		t.Fatal("app should not reopen when force stop fails")
		return nil
	}

	err := (&CodexApp{}).Run("qwen3.5", nil)
	if err == nil || !strings.Contains(err.Error(), "force stop Codex") || !strings.Contains(err.Error(), "access denied") {
		t.Fatalf("Run error = %v, want force stop failure", err)
	}
}

func TestCodexAppRunRejectsExtraArgs(t *testing.T) {
	withCodexAppPlatform(t, "darwin")
	err := (&CodexApp{}).Run("qwen3.5", []string{"--foo"})
	if err == nil || !strings.Contains(err.Error(), "does not accept extra arguments") {
		t.Fatalf("Run error = %v, want extra args rejection", err)
	}
}

func TestCodexAppProcessMatchesMainAndAppServer(t *testing.T) {
	for _, command := range []string{
		"/Applications/Codex.app/Contents/MacOS/Codex",
		"/Applications/Codex.app/Contents/Resources/codex app-server --analytics-default-enabled",
		`C:\Users\parth\AppData\Local\Programs\Codex\Codex.exe`,
		`"C:\Users\parth\AppData\Local\Codex\app-26.429.30905\resources\codex.exe" app-server --analytics-default-enabled`,
		`"C:\Users\parth\AppData\Local\openai-codex-electron\resources\codex.exe" "app-server"`,
	} {
		if !codexAppProcessMatches(command) {
			t.Fatalf("expected command to match Codex App process: %s", command)
		}
	}

	for _, command := range []string{
		"/Applications/Codex.app/Contents/Frameworks/Codex Helper.app/Contents/MacOS/Codex Helper",
		"/Applications/Codex.app/Contents/Frameworks/Electron Framework.framework/Helpers/chrome_crashpad_handler",
		`"C:\Program Files\WindowsApps\OpenAI.Codex_26.429.8261.0_x64__2p2nqsd0c76g0\app\Codex.exe" --type=renderer --user-data-dir="C:\Users\parth\AppData\Roaming\Codex"`,
		`"C:\Program Files\WindowsApps\OpenAI.Codex_26.429.8261.0_x64__2p2nqsd0c76g0\app\Codex.exe" --type=crashpad-handler`,
	} {
		if codexAppProcessMatches(command) {
			t.Fatalf("expected helper command not to match Codex App process: %s", command)
		}
	}
}

func catalogSlugs(models []map[string]any) []string {
	slugs := make([]string, 0, len(models))
	for _, model := range models {
		if slug, _ := model["slug"].(string); slug != "" {
			slugs = append(slugs, slug)
		}
	}
	return slugs
}

func catalogInputModalities(entry map[string]any) []string {
	raw, _ := entry["input_modalities"].([]any)
	modalities := make([]string, 0, len(raw))
	for _, item := range raw {
		if modality, _ := item.(string); modality != "" {
			modalities = append(modalities, modality)
		}
	}
	return modalities
}

func mustCodexAppModelCatalogPath(t *testing.T) string {
	t.Helper()
	catalogPath, err := codexAppModelCatalogPath()
	if err != nil {
		t.Fatal(err)
	}
	return catalogPath
}

func mustWriteCodexAppTestCatalog(t *testing.T, slugs ...string) string {
	t.Helper()
	catalogPath := mustCodexAppModelCatalogPath(t)
	if err := os.MkdirAll(filepath.Dir(catalogPath), 0o755); err != nil {
		t.Fatal(err)
	}
	models := make([]map[string]string, 0, len(slugs))
	for _, slug := range slugs {
		models = append(models, map[string]string{"slug": slug})
	}
	data, err := json.Marshal(map[string]any{"models": models})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(catalogPath, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return catalogPath
}
