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
	oldRunPath := codexAppRunPath
	oldStartID := codexAppStartID
	codexAppIsRunning = isRunning
	codexAppQuitApp = quit
	codexAppOpenApp = open
	t.Cleanup(func() {
		codexAppIsRunning = oldIsRunning
		codexAppQuitApp = oldQuit
		codexAppOpenApp = oldOpen
		codexAppOpenPath = oldOpenPath
		codexAppOpenStart = oldOpenStart
		codexAppRunPath = oldRunPath
		codexAppStartID = oldStartID
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
		`profile = "ollama-launch"`,
		`model = "llama3.2"`,
		`model_provider = "ollama-launch"`,
		fmt.Sprintf(`model_catalog_json = %q`, catalogPath),
		`[profiles.ollama-launch]`,
		`model = "llama3.2"`,
		`openai_base_url = "http://127.0.0.1:9999/v1/"`,
		`model_provider = "ollama-launch"`,
		`model_catalog_json = "`,
		`[model_providers.ollama-launch]`,
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
		"[profiles.ollama-launch]\n" +
		"model = \"llama3.2\"\n" +
		"model_provider = \"ollama-launch\"\n\n" +
		"[model_providers.ollama-launch]\n" +
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
		`model_provider = "ollama-launch"` + "\n\n" +
		"[model_providers.ollama-launch]\n" +
		`base_url = "http://127.0.0.1:11434/v1/"` + "\n"
	if err := os.WriteFile(configPath, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	if got := (&CodexApp{}).CurrentModel(); got != "qwen3:8b" {
		t.Fatalf("CurrentModel = %q, want qwen3:8b", got)
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
			fmt.Fprintf(w, `{"model_info":{"general.context_length":%d}}`, 65536+len(req.Model))
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
		if model["context_window"] != float64(65536+len(slug)) {
			t.Fatalf("context_window for %q = %v", slug, model["context_window"])
		}
		if showCalls[slug] != 1 {
			t.Fatalf("show calls for %q = %d, want 1", slug, showCalls[slug])
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
	if !strings.Contains(string(data), `profile = "default"`) || strings.Contains(string(data), `profile = "ollama-launch"`) {
		t.Fatalf("restore should restore previous active profile, got:\n%s", data)
	}
	restored := string(data)
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
