package launch

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func withClaudeDesktopPlatform(t *testing.T, goos string) {
	t.Helper()
	oldGOOS := claudeDesktopGOOS
	oldProbe := claudeDesktopProbeGateway
	claudeDesktopGOOS = goos
	claudeDesktopProbeGateway = func(context.Context, string) error { return nil }
	t.Cleanup(func() {
		claudeDesktopGOOS = oldGOOS
		claudeDesktopProbeGateway = oldProbe
	})
}

func withClaudeDesktopProcessHooks(t *testing.T, running func() bool, quit func() error, open func() error) {
	t.Helper()
	oldRunning := claudeDesktopIsRunning
	oldQuit := claudeDesktopQuitApp
	oldOpen := claudeDesktopOpenApp
	oldOpenPath := claudeDesktopOpenAppPath
	oldRunningPath := claudeDesktopRunningAppPath
	claudeDesktopIsRunning = func(context.Context) (bool, error) { return running(), nil }
	claudeDesktopQuitApp = func(context.Context) error { return quit() }
	claudeDesktopOpenApp = open
	claudeDesktopOpenAppPath = oldOpenPath
	claudeDesktopRunningAppPath = oldRunningPath
	t.Cleanup(func() {
		claudeDesktopIsRunning = oldRunning
		claudeDesktopQuitApp = oldQuit
		claudeDesktopOpenApp = oldOpen
		claudeDesktopOpenAppPath = oldOpenPath
		claudeDesktopRunningAppPath = oldRunningPath
	})
}

func claudeDesktopReadJSON(t *testing.T, path string) map[string]any {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var cfg map[string]any
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
	return cfg
}

func claudeDesktopReadStringSlice(t *testing.T, value any) []string {
	t.Helper()
	items, ok := value.([]any)
	if !ok {
		t.Fatalf("value = %T, want JSON array", value)
	}
	result := make([]string, len(items))
	for i, item := range items {
		result[i], ok = item.(string)
		if !ok {
			t.Fatalf("value[%d] = %T, want string", i, item)
		}
	}
	return result
}

func assertClaudeDesktopEgressHosts(t *testing.T, profile map[string]any) {
	t.Helper()
	got := claudeDesktopReadStringSlice(t, profile["coworkEgressAllowedHosts"])
	if len(got) != len(claudeDesktopEgressHosts) {
		t.Fatalf("coworkEgressAllowedHosts = %v, want %v", got, claudeDesktopEgressHosts)
	}
	for i := range got {
		if got[i] != claudeDesktopEgressHosts[i] {
			t.Fatalf("coworkEgressAllowedHosts = %v, want %v", got, claudeDesktopEgressHosts)
		}
	}
}

func TestClaudeDesktopIntegration(t *testing.T) {
	c := &ClaudeDesktop{}

	t.Run("implements Runner", func(t *testing.T) {
		var _ Runner = c
	})
	t.Run("implements managed autodiscovery integration", func(t *testing.T) {
		var _ ManagedAutodiscoveryIntegration = c
	})
	t.Run("does not use Ollama Cloud auth gate", func(t *testing.T) {
		if _, ok := any(c).(ManagedAutodiscoveryCloudIntegration); ok {
			t.Fatal("Claude Desktop's loopback gateway should not require Ollama Cloud sign-in")
		}
	})
	t.Run("implements restore", func(t *testing.T) {
		var _ RestorableIntegration = c
	})
	t.Run("has restore hint", func(t *testing.T) {
		var _ RestoreHintIntegration = c
		if !strings.Contains(c.RestoreHint(), "--restore") {
			t.Fatalf("expected restore hint to mention --restore, got %q", c.RestoreHint())
		}
		if strings.Contains(c.RestoreHint(), "Tip:") {
			t.Fatalf("restore hint should not use Tip wording, got %q", c.RestoreHint())
		}
	})
	t.Run("has success messages", func(t *testing.T) {
		var _ ConfigurationSuccessIntegration = c
		var _ RestoreSuccessIntegration = c
		if got := c.ConfigurationSuccessMessage(); got != "Claude Desktop profile changed to Ollama.\nTo restore the usual Claude profile, run: ollama launch claude-desktop --restore" {
			t.Fatalf("configuration success message = %q", got)
		}
		if got := c.RestoreSuccessMessage(); got != "Claude Desktop restored to the usual Claude profile." {
			t.Fatalf("restore success message = %q", got)
		}
	})
	t.Run("skips local model readiness", func(t *testing.T) {
		var _ ManagedModelReadinessSkipper = c
		if !c.SkipModelReadiness() {
			t.Fatal("expected Claude Desktop to skip local model readiness")
		}
	})
}

func TestClaudeDesktopSupportedOnlyOnDarwin(t *testing.T) {
	withClaudeDesktopPlatform(t, "windows")
	if err := (&ClaudeDesktop{}).Supported(); err == nil || !strings.Contains(err.Error(), "only supported on macOS") {
		t.Fatalf("Supported error = %v, want macOS-only error", err)
	}
}

func TestClaudeDesktopConfigureRequiresOllamaGateway(t *testing.T) {
	withClaudeDesktopPlatform(t, "darwin")
	claudeDesktopProbeGateway = func(context.Context, string) error {
		return errors.New("another service is using 127.0.0.1:11435")
	}

	err := (&ClaudeDesktop{}).ConfigureAutodiscovery()
	if err == nil || !strings.Contains(err.Error(), "restart Ollama and try again") {
		t.Fatalf("ConfigureAutodiscovery error = %v, want gateway recovery guidance", err)
	}
}

func TestLaunchIntegration_ClaudeDesktopDoesNotRequireLocalCloudSignIn(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withClaudeDesktopPlatform(t, "darwin")
	withInteractiveSession(t, true)
	withLauncherHooks(t)
	t.Setenv("OLLAMA_API_KEY", "test-api-key")

	if err := os.MkdirAll(filepath.Join(tmpDir, "Applications", "Claude.app"), 0o755); err != nil {
		t.Fatal(err)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/status":
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprint(w, `{"error":"not found"}`)
		case "/api/me":
			w.WriteHeader(http.StatusUnauthorized)
			fmt.Fprint(w, `{"error":"unauthorized","signin_url":"https://example.com/signin"}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	DefaultSignIn = func(modelName, signInURL string) (string, error) {
		t.Fatalf("Claude Desktop launch should not require Ollama Cloud sign-in, got %s at %s", modelName, signInURL)
		return "", nil
	}

	var openCalls int
	withClaudeDesktopProcessHooks(t,
		func() bool { return false },
		func() error { return nil },
		func() error {
			openCalls++
			return nil
		},
	)

	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{Name: "claude-desktop"}); err != nil {
		t.Fatalf("LaunchIntegration returned error: %v", err)
	}
	if openCalls != 1 {
		t.Fatalf("open calls = %d, want 1", openCalls)
	}
}

func TestClaudeDesktopConfigureWritesOllamaCloudProfile(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withClaudeDesktopPlatform(t, "darwin")
	t.Setenv("OLLAMA_API_KEY", "test-api-key")

	paths, err := claudeDesktopConfigPaths()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(paths.desktopConfig), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(paths.meta), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(paths.desktopConfig, []byte(`{"existing":true}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(paths.meta, []byte(`{"entries":[{"id":"custom","name":"Custom"}]}`), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := (&ClaudeDesktop{}).ConfigureAutodiscovery(); err != nil {
		t.Fatalf("Configure returned error: %v", err)
	}

	desktopConfig := claudeDesktopReadJSON(t, paths.desktopConfig)
	if desktopConfig["existing"] != true {
		t.Fatalf("existing desktop config key was not preserved: %v", desktopConfig)
	}
	if desktopConfig["deploymentMode"] != "3p" {
		t.Fatalf("deploymentMode = %v, want 3p", desktopConfig["deploymentMode"])
	}
	normalConfig := claudeDesktopReadJSON(t, paths.normalConfig)
	if normalConfig["deploymentMode"] != "3p" {
		t.Fatalf("normal deploymentMode = %v, want 3p", normalConfig["deploymentMode"])
	}

	meta := claudeDesktopReadJSON(t, paths.meta)
	if meta["appliedId"] != claudeDesktopProfileID {
		t.Fatalf("appliedId = %v, want %s", meta["appliedId"], claudeDesktopProfileID)
	}
	entries, _ := meta["entries"].([]any)
	if len(entries) != 2 {
		t.Fatalf("entries len = %d, want 2: %v", len(entries), entries)
	}

	profile := claudeDesktopReadJSON(t, paths.profile)
	if profile["inferenceProvider"] != "gateway" {
		t.Fatalf("inferenceProvider = %v, want gateway", profile["inferenceProvider"])
	}
	if profile["inferenceGatewayBaseUrl"] != claudeDesktopGatewayBaseURL {
		t.Fatalf("base URL = %v, want %s", profile["inferenceGatewayBaseUrl"], claudeDesktopGatewayBaseURL)
	}
	if profile["inferenceGatewayApiKey"] != "ollama" {
		t.Fatal("expected local placeholder API key to be written")
	}
	if profile["inferenceGatewayAuthScheme"] != "bearer" {
		t.Fatalf("auth scheme = %v, want bearer", profile["inferenceGatewayAuthScheme"])
	}
	if profile["disableDeploymentModeChooser"] != true {
		t.Fatalf("disableDeploymentModeChooser = %v, want true", profile["disableDeploymentModeChooser"])
	}
	if profile["chatTabEnabled"] != true {
		t.Fatalf("chatTabEnabled = %v, want true", profile["chatTabEnabled"])
	}
	if profile["autoModeEnabled"] != false {
		t.Fatalf("autoModeEnabled = %v, want false", profile["autoModeEnabled"])
	}
	assertClaudeDesktopEgressHosts(t, profile)
	if profile["disableEssentialTelemetry"] != true {
		t.Fatalf("disableEssentialTelemetry = %v, want true", profile["disableEssentialTelemetry"])
	}
	if profile["disableNonessentialTelemetry"] != true {
		t.Fatalf("disableNonessentialTelemetry = %v, want true", profile["disableNonessentialTelemetry"])
	}
	if _, ok := profile["inferenceModels"]; ok {
		t.Fatalf("inferenceModels should be omitted so Claude can discover models, got %v", profile["inferenceModels"])
	}
}

func TestClaudeDesktopConfigureActivatesNormalProfileLast(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withClaudeDesktopPlatform(t, "darwin")

	paths, err := claudeDesktopConfigPaths()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(paths.normalConfig), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(paths.desktopConfig), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(paths.normalConfig, []byte(`{"deploymentMode":"1p","existing":true}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(paths.desktopConfig, []byte(`{"deploymentMode":`), 0o644); err != nil {
		t.Fatal(err)
	}

	err = (&ClaudeDesktop{}).ConfigureAutodiscovery()
	if err == nil {
		t.Fatal("ConfigureAutodiscovery succeeded with malformed third-party config")
	}
	normal := claudeDesktopReadJSON(t, paths.normalConfig)
	if normal["deploymentMode"] != "1p" || normal["existing"] != true {
		t.Fatalf("normal profile changed before third-party assets were ready: %v", normal)
	}
	if (&ClaudeDesktop{}).UsesOllamaGateway() {
		t.Fatal("failed configuration left Claude routed through Ollama")
	}
}

func TestClaudeDesktopConfigureAutodiscoveryRemovesExistingModelCatalog(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withClaudeDesktopPlatform(t, "darwin")
	t.Setenv("OLLAMA_API_KEY", "test-api-key")

	paths, err := claudeDesktopConfigPaths()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(paths.profile), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(paths.profile, []byte(`{"autoModeEnabled":true,"chatTabEnabled":false,"coworkEgressAllowedHosts":["old.example.com"],"disableEssentialTelemetry":false,"disableNonessentialTelemetry":false,"inferenceModels":["qwen3.5"],"inferenceGatewayApiKey":"old"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := (&ClaudeDesktop{}).ConfigureAutodiscovery(); err != nil {
		t.Fatalf("ConfigureAutodiscovery returned error: %v", err)
	}

	profile := claudeDesktopReadJSON(t, paths.profile)
	if _, ok := profile["inferenceModels"]; ok {
		t.Fatalf("inferenceModels should be removed, got %v", profile["inferenceModels"])
	}
	if profile["inferenceGatewayApiKey"] != "ollama" {
		t.Fatal("expected local placeholder API key to replace the old key")
	}
	if profile["chatTabEnabled"] != true {
		t.Fatalf("chatTabEnabled = %v, want true", profile["chatTabEnabled"])
	}
	if profile["autoModeEnabled"] != false {
		t.Fatalf("autoModeEnabled = %v, want false", profile["autoModeEnabled"])
	}
	assertClaudeDesktopEgressHosts(t, profile)
	if profile["disableEssentialTelemetry"] != true || profile["disableNonessentialTelemetry"] != true {
		t.Fatalf("telemetry flags = %v/%v, want true/true", profile["disableEssentialTelemetry"], profile["disableNonessentialTelemetry"])
	}
}

func TestClaudeDesktopWindowsConfigPathsUseLocalAppData(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withClaudeDesktopPlatform(t, "windows")
	t.Setenv("LOCALAPPDATA", filepath.Join(tmpDir, "LocalAppData"))

	paths, err := claudeDesktopConfigPaths()
	if err != nil {
		t.Fatal(err)
	}
	if want := filepath.Join(tmpDir, "LocalAppData", "Claude-3p", "claude_desktop_config.json"); paths.desktopConfig != want {
		t.Fatalf("desktop config = %q, want %q", paths.desktopConfig, want)
	}
	if want := filepath.Join(tmpDir, "LocalAppData", "Claude", "claude_desktop_config.json"); paths.normalConfig != want {
		t.Fatalf("normal config = %q, want %q", paths.normalConfig, want)
	}
}

func TestClaudeDesktopWindowsConfigPathsFallbackToNestProfile(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withClaudeDesktopPlatform(t, "windows")
	local := filepath.Join(tmpDir, "LocalAppData")
	t.Setenv("LOCALAPPDATA", local)
	if err := os.MkdirAll(filepath.Join(local, "Claude Nest-3p"), 0o755); err != nil {
		t.Fatal(err)
	}

	paths, err := claudeDesktopConfigPaths()
	if err != nil {
		t.Fatal(err)
	}
	if want := filepath.Join(local, "Claude Nest-3p", "claude_desktop_config.json"); paths.desktopConfig != want {
		t.Fatalf("desktop config = %q, want %q", paths.desktopConfig, want)
	}
}

func TestClaudeDesktopInstalledOnWindowsRecognizesLocalProfileDir(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withClaudeDesktopPlatform(t, "windows")
	local := filepath.Join(tmpDir, "LocalAppData")
	t.Setenv("LOCALAPPDATA", local)
	withClaudeDesktopProcessHooks(t, func() bool { return false }, func() error { return nil }, func() error { return nil })
	if err := os.MkdirAll(filepath.Join(local, "Claude-3p"), 0o755); err != nil {
		t.Fatal(err)
	}

	if !ClaudeDesktopInstalled() {
		t.Fatal("expected Claude Desktop to be installed when the Windows profile directory exists")
	}
}

func TestClaudeDesktopInstalledOnDarwinRequiresApp(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withClaudeDesktopPlatform(t, "darwin")
	previousStat := claudeDesktopStat
	claudeDesktopStat = func(path string) (os.FileInfo, error) {
		if strings.HasPrefix(path, "/Applications/") {
			return nil, os.ErrNotExist
		}
		return os.Stat(path)
	}
	t.Cleanup(func() { claudeDesktopStat = previousStat })
	if err := os.MkdirAll(filepath.Join(tmpDir, "Library", "Application Support", "Claude"), 0o755); err != nil {
		t.Fatal(err)
	}

	if ClaudeDesktopInstalled() {
		t.Fatal("profile files alone should not make Claude Desktop appear installed on macOS")
	}
}

func TestClaudeDesktopWindowsAppPathFindsAnthropicClaudeInstall(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withClaudeDesktopPlatform(t, "windows")
	local := filepath.Join(tmpDir, "LocalAppData")
	t.Setenv("LOCALAPPDATA", local)
	want := filepath.Join(local, "AnthropicClaude", "app-1.2.3", "Claude.exe")
	if err := os.MkdirAll(filepath.Dir(want), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(want, []byte(""), 0o755); err != nil {
		t.Fatal(err)
	}

	if got := claudeDesktopAppPath(); got != want {
		t.Fatalf("claudeDesktopAppPath() = %q, want %q", got, want)
	}
}

func TestWaitForClaudeDesktopExitUsesRunningHook(t *testing.T) {
	withClaudeDesktopPlatform(t, "windows")
	runningChecks := 0
	withClaudeDesktopProcessHooks(t,
		func() bool {
			runningChecks++
			return runningChecks == 1
		},
		func() error { return nil },
		func() error { return nil },
	)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := waitForClaudeDesktopExit(ctx); err != nil {
		t.Fatalf("waitForClaudeDesktopExit returned error: %v", err)
	}
	if runningChecks < 2 {
		t.Fatalf("expected running hook to be checked until the visible window exits, got %d checks", runningChecks)
	}
}

func TestClaudeDesktopWindowsOpenDoesNotFallBackToClaudeCommand(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withClaudeDesktopPlatform(t, "windows")
	t.Setenv("LOCALAPPDATA", filepath.Join(tmpDir, "LocalAppData"))

	oldRunningPath := claudeDesktopRunningAppPath
	claudeDesktopRunningAppPath = func() string { return "" }
	t.Cleanup(func() { claudeDesktopRunningAppPath = oldRunningPath })

	err := defaultClaudeDesktopOpenApp()
	if err == nil || !strings.Contains(err.Error(), "Claude Desktop executable was not found") {
		t.Fatalf("defaultClaudeDesktopOpenApp error = %v, want executable-not-found error", err)
	}
}

func TestClaudeDesktopConfigureIgnoresCloudAPIKey(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withClaudeDesktopPlatform(t, "darwin")
	t.Setenv("OLLAMA_API_KEY", "bad-key")

	if err := (&ClaudeDesktop{}).ConfigureAutodiscovery(); err != nil {
		t.Fatalf("Configure error = %v", err)
	}

	paths, err := claudeDesktopConfigPaths()
	if err != nil {
		t.Fatal(err)
	}
	profile := claudeDesktopReadJSON(t, paths.profile)
	if profile["inferenceGatewayApiKey"] != "ollama" {
		t.Fatalf("gateway API key = %v, want local placeholder", profile["inferenceGatewayApiKey"])
	}
}

func TestClaudeDesktopConfigureDoesNotRequireAPIKey(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withClaudeDesktopPlatform(t, "darwin")
	t.Setenv("OLLAMA_API_KEY", "")
	if err := (&ClaudeDesktop{}).ConfigureAutodiscovery(); err != nil {
		t.Fatalf("Configure error = %v", err)
	}
}

func TestClaudeDesktopConfigureReplacesExistingAPIKeyWithPlaceholder(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withClaudeDesktopPlatform(t, "darwin")
	t.Setenv("OLLAMA_API_KEY", "")

	paths, err := claudeDesktopConfigPaths()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(paths.profile), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(paths.profile, []byte(`{"inferenceGatewayApiKey":"existing-key"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := (&ClaudeDesktop{}).ConfigureAutodiscovery(); err != nil {
		t.Fatalf("ConfigureAutodiscovery returned error: %v", err)
	}
	profile := claudeDesktopReadJSON(t, paths.profile)
	if profile["inferenceGatewayApiKey"] != "ollama" {
		t.Fatalf("gateway API key = %v, want local placeholder", profile["inferenceGatewayApiKey"])
	}
}

func TestClaudeDesktopConfigureDoesNotPromptForExistingAPIKey(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withClaudeDesktopPlatform(t, "darwin")
	withInteractiveSession(t, true)
	t.Setenv("OLLAMA_API_KEY", "")

	paths, err := claudeDesktopConfigPaths()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(paths.profile), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(paths.profile, []byte(`{"inferenceGatewayApiKey":"stale-key"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := (&ClaudeDesktop{}).ConfigureAutodiscovery(); err != nil {
		t.Fatalf("ConfigureAutodiscovery returned error: %v", err)
	}
	profile := claudeDesktopReadJSON(t, paths.profile)
	if profile["inferenceGatewayApiKey"] != "ollama" {
		t.Fatalf("configured key = %v, want local placeholder", profile["inferenceGatewayApiKey"])
	}
}

func TestClaudeDesktopAutodiscoveryConfiguredRequiresAppliedOllamaProfile(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withClaudeDesktopPlatform(t, "darwin")
	t.Setenv("OLLAMA_API_KEY", "test-api-key")

	c := &ClaudeDesktop{}
	if err := c.ConfigureAutodiscovery(); err != nil {
		t.Fatalf("Configure returned error: %v", err)
	}
	if !c.AutodiscoveryConfigured() {
		t.Fatal("expected Claude Desktop autodiscovery config to be detected")
	}

	paths, err := claudeDesktopConfigPaths()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(paths.meta, []byte(`{"appliedId":"custom"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if c.AutodiscoveryConfigured() {
		t.Fatal("expected another applied profile to hide Claude Desktop autodiscovery config")
	}
	if c.UsesOllamaGateway() {
		t.Fatal("expected another applied profile to stop routing through Ollama")
	}
}

func TestClaudeDesktopAutodiscoveryConfiguredRequiresAPIKey(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withClaudeDesktopPlatform(t, "darwin")
	t.Setenv("OLLAMA_API_KEY", "test-api-key")

	c := &ClaudeDesktop{}
	if err := c.ConfigureAutodiscovery(); err != nil {
		t.Fatalf("Configure returned error: %v", err)
	}

	paths, err := claudeDesktopConfigPaths()
	if err != nil {
		t.Fatal(err)
	}
	profile := claudeDesktopReadJSON(t, paths.profile)
	delete(profile, "inferenceGatewayApiKey")
	data, err := json.Marshal(profile)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(paths.profile, data, 0o644); err != nil {
		t.Fatal(err)
	}

	if c.AutodiscoveryConfigured() {
		t.Fatal("expected missing gateway API key to force Claude Desktop reconfiguration")
	}
	if !c.UsesOllamaGateway() {
		t.Fatal("expected missing gateway API key to leave Ollama routing active")
	}
}

func TestClaudeDesktopAutodiscoveryConfiguredRequiresTelemetryDisabled(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withClaudeDesktopPlatform(t, "darwin")

	c := &ClaudeDesktop{}
	if err := c.ConfigureAutodiscovery(); err != nil {
		t.Fatalf("Configure returned error: %v", err)
	}

	paths, err := claudeDesktopConfigPaths()
	if err != nil {
		t.Fatal(err)
	}
	profile := claudeDesktopReadJSON(t, paths.profile)
	profile["disableEssentialTelemetry"] = false
	if err := writeClaudeDesktopJSON(paths.profile, profile); err != nil {
		t.Fatal(err)
	}
	if c.AutodiscoveryConfigured() {
		t.Fatal("expected essential telemetry to force Claude Desktop profile reconfiguration")
	}
	if !c.UsesOllamaGateway() {
		t.Fatal("expected essential telemetry drift to leave Ollama routing active")
	}

	profile["disableEssentialTelemetry"] = true
	profile["disableNonessentialTelemetry"] = false
	if err := writeClaudeDesktopJSON(paths.profile, profile); err != nil {
		t.Fatal(err)
	}

	if c.AutodiscoveryConfigured() {
		t.Fatal("expected nonessential telemetry to force Claude Desktop profile reconfiguration")
	}
	if !c.UsesOllamaGateway() {
		t.Fatal("expected nonessential telemetry drift to leave Ollama routing active")
	}
}

func TestClaudeDesktopAutodiscoveryConfiguredRequiresEgressHosts(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withClaudeDesktopPlatform(t, "darwin")

	c := &ClaudeDesktop{}
	if err := c.ConfigureAutodiscovery(); err != nil {
		t.Fatalf("Configure returned error: %v", err)
	}
	paths, err := claudeDesktopConfigPaths()
	if err != nil {
		t.Fatal(err)
	}
	profile := claudeDesktopReadJSON(t, paths.profile)

	for _, value := range []any{nil, []string{"github.com"}} {
		if value == nil {
			delete(profile, "coworkEgressAllowedHosts")
		} else {
			profile["coworkEgressAllowedHosts"] = value
		}
		if err := writeClaudeDesktopJSON(paths.profile, profile); err != nil {
			t.Fatal(err)
		}
		if c.AutodiscoveryConfigured() {
			t.Fatalf("expected egress hosts %v to force Claude Desktop profile reconfiguration", value)
		}
		if !c.UsesOllamaGateway() {
			t.Fatalf("expected egress hosts %v to leave Ollama routing active", value)
		}
	}
}

func TestClaudeDesktopUsesOllamaGatewayRequiresCoreRoutingSettings(t *testing.T) {
	setTestHome(t, t.TempDir())
	withClaudeDesktopPlatform(t, "darwin")

	c := &ClaudeDesktop{}
	paths, err := claudeDesktopConfigPaths()
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name   string
		mutate func() error
	}{
		{
			name: "normal deployment mode",
			mutate: func() error {
				return writeClaudeDesktopDeploymentMode(paths.normalConfig, "1p")
			},
		},
		{
			name: "third-party deployment mode",
			mutate: func() error {
				return writeClaudeDesktopDeploymentMode(paths.desktopConfig, "1p")
			},
		},
		{
			name: "applied profile",
			mutate: func() error {
				return writeClaudeDesktopMeta(paths.meta, "custom", "Custom")
			},
		},
		{
			name: "inference provider",
			mutate: func() error {
				profile := claudeDesktopReadJSON(t, paths.profile)
				profile["inferenceProvider"] = "bedrock"
				return writeClaudeDesktopJSON(paths.profile, profile)
			},
		},
		{
			name: "gateway base URL",
			mutate: func() error {
				profile := claudeDesktopReadJSON(t, paths.profile)
				profile["inferenceGatewayBaseUrl"] = "http://127.0.0.1:11434"
				return writeClaudeDesktopJSON(paths.profile, profile)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := c.ConfigureAutodiscovery(); err != nil {
				t.Fatal(err)
			}
			if !c.UsesOllamaGateway() {
				t.Fatal("expected configured Claude Desktop to route through Ollama")
			}
			if err := tt.mutate(); err != nil {
				t.Fatal(err)
			}
			if c.UsesOllamaGateway() {
				t.Fatal("expected core routing drift to stop Ollama routing")
			}
		})
	}
}

func TestClaudeDesktopRestoreSwitchesBackToFirstPartyMode(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withClaudeDesktopPlatform(t, "darwin")
	withClaudeDesktopProcessHooks(t, func() bool { return false }, func() error { return nil }, func() error { return nil })

	paths, err := claudeDesktopConfigPaths()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(paths.profile), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(paths.meta, []byte(`{"appliedId":"`+claudeDesktopProfileID+`","entries":[{"id":"`+claudeDesktopProfileID+`","name":"Ollama"}]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(paths.profile, []byte(`{"autoModeEnabled":true,"coworkEgressAllowedHosts":["github.com"],"disableDeploymentModeChooser":true,"disableEssentialTelemetry":true,"disableNonessentialTelemetry":true,"inferenceGatewayApiKey":"keep","inferenceProvider":"gateway","inferenceGatewayBaseUrl":"https://ollama.com","inferenceGatewayAuthScheme":"bearer","inferenceModels":["legacy"]}`), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := (&ClaudeDesktop{}).Restore(); err != nil {
		t.Fatalf("Restore returned error: %v", err)
	}

	desktopConfig := claudeDesktopReadJSON(t, paths.desktopConfig)
	if desktopConfig["deploymentMode"] != "1p" {
		t.Fatalf("deploymentMode = %v, want 1p", desktopConfig["deploymentMode"])
	}
	normalConfig := claudeDesktopReadJSON(t, paths.normalConfig)
	if normalConfig["deploymentMode"] != "1p" {
		t.Fatalf("normal deploymentMode = %v, want 1p", normalConfig["deploymentMode"])
	}
	profile := claudeDesktopReadJSON(t, paths.profile)
	if profile["disableDeploymentModeChooser"] != false {
		t.Fatalf("disableDeploymentModeChooser = %v, want false", profile["disableDeploymentModeChooser"])
	}
	if profile["inferenceGatewayApiKey"] != "keep" {
		t.Fatal("restore should leave existing Ollama profile credentials in place")
	}
	for _, key := range []string{"inferenceProvider", "inferenceGatewayBaseUrl", "inferenceGatewayAuthScheme", "inferenceModels", "coworkEgressAllowedHosts", "autoModeEnabled", "disableEssentialTelemetry", "disableNonessentialTelemetry"} {
		if _, ok := profile[key]; ok {
			t.Fatalf("restore should clear stale %s from the Ollama profile: %v", key, profile)
		}
	}
	meta := claudeDesktopReadJSON(t, paths.meta)
	if _, ok := meta["appliedId"]; ok {
		t.Fatalf("restore should clear the applied Ollama third-party profile: %v", meta)
	}
	if (&ClaudeDesktop{}).AutodiscoveryConfigured() {
		t.Fatal("restore should leave Claude Desktop autodiscovery unconfigured")
	}
}

func TestClaudeDesktopRestoreRemainsAvailableOnWindows(t *testing.T) {
	localAppData := t.TempDir()
	t.Setenv("LOCALAPPDATA", localAppData)
	withClaudeDesktopPlatform(t, "windows")
	withClaudeDesktopProcessHooks(t, func() bool { return false }, func() error { return nil }, func() error { return nil })

	targets, err := claudeDesktopTargetPaths()
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range targets.normalConfigs {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(`{"deploymentMode":"3p"}`), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for _, target := range targets.thirdPartyProfiles {
		if err := os.MkdirAll(filepath.Dir(target.profile), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(target.desktopConfig, []byte(`{"deploymentMode":"3p"}`), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(target.meta, []byte(`{"appliedId":"`+claudeDesktopProfileID+`"}`), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(target.profile, []byte(`{"inferenceProvider":"gateway","inferenceGatewayBaseUrl":"http://127.0.0.1:11435"}`), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	if err := (&ClaudeDesktop{}).Restore(); err != nil {
		t.Fatalf("Restore returned error: %v", err)
	}
	for _, path := range targets.normalConfigs {
		if got := claudeDesktopReadJSON(t, path)["deploymentMode"]; got != "1p" {
			t.Fatalf("%s deploymentMode = %v, want 1p", path, got)
		}
	}
	for _, target := range targets.thirdPartyProfiles {
		if got := claudeDesktopReadJSON(t, target.desktopConfig)["deploymentMode"]; got != "1p" {
			t.Fatalf("%s deploymentMode = %v, want 1p", target.desktopConfig, got)
		}
		if _, ok := claudeDesktopReadJSON(t, target.profile)["inferenceProvider"]; ok {
			t.Fatalf("%s still contains inferenceProvider", target.profile)
		}
	}
}

func TestClaudeDesktopRunRestartsRunningAppWhenConfirmed(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withClaudeDesktopPlatform(t, "darwin")
	restoreConfirm := withLaunchConfirmPolicy(launchConfirmPolicy{yes: true})
	defer restoreConfirm()
	paths, err := claudeDesktopConfigPaths()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(paths.profile), 0o755); err != nil {
		t.Fatal(err)
	}

	running := true
	var quitCalls, openCalls int
	withClaudeDesktopProcessHooks(t,
		func() bool { return running },
		func() error {
			quitCalls++
			running = false
			return os.WriteFile(paths.profile, []byte(`{"chatTabEnabled":false}`), 0o644)
		},
		func() error {
			openCalls++
			return nil
		},
	)

	if err := (&ClaudeDesktop{}).Run("qwen3.5", nil, nil); err != nil {
		t.Fatalf("Run returned error: %v", err)
	}
	if quitCalls != 1 || openCalls != 1 {
		t.Fatalf("quit/open calls = %d/%d, want 1/1", quitCalls, openCalls)
	}
	profile := claudeDesktopReadJSON(t, paths.profile)
	if profile["chatTabEnabled"] != true {
		t.Fatalf("chatTabEnabled = %v, want true after Claude shutdown write", profile["chatTabEnabled"])
	}
}

func TestClaudeDesktopSetInstalledFromDesktopRequiresRestartConfirmation(t *testing.T) {
	setTestHome(t, t.TempDir())
	withClaudeDesktopPlatform(t, "darwin")
	withClaudeDesktopProcessHooks(t,
		func() bool { return true },
		func() error { t.Fatal("Claude should not quit without confirmation"); return nil },
		func() error { t.Fatal("Claude should not open without confirmation"); return nil },
	)

	err := (&ClaudeDesktop{}).SetInstalledFromDesktop(true, false)
	if err == nil || !strings.Contains(err.Error(), "restart confirmation is required") {
		t.Fatalf("SetInstalledFromDesktop error = %v, want restart confirmation error", err)
	}
}

func TestClaudeDesktopSetInstalledFromDesktopOpensStoppedAppWhenEnabled(t *testing.T) {
	setTestHome(t, t.TempDir())
	withClaudeDesktopPlatform(t, "darwin")
	openCalls := 0
	withClaudeDesktopProcessHooks(t,
		func() bool { return false },
		func() error { t.Fatal("stopped Claude should not be quit"); return nil },
		func() error { openCalls++; return nil },
	)

	c := &ClaudeDesktop{}
	if err := c.SetInstalledFromDesktop(true, false); err != nil {
		t.Fatalf("SetInstalledFromDesktop returned error: %v", err)
	}
	if openCalls != 1 || !c.AutodiscoveryConfigured() {
		t.Fatalf("open calls/configured = %d/%v, want 1/true", openCalls, c.AutodiscoveryConfigured())
	}
}

func TestClaudeDesktopSetInstalledFromDesktopDoesNotOpenStoppedAppWhenDisabled(t *testing.T) {
	setTestHome(t, t.TempDir())
	withClaudeDesktopPlatform(t, "darwin")
	c := &ClaudeDesktop{}
	if err := c.ConfigureAutodiscovery(); err != nil {
		t.Fatal(err)
	}
	withClaudeDesktopProcessHooks(t,
		func() bool { return false },
		func() error { t.Fatal("stopped Claude should not be quit"); return nil },
		func() error { t.Fatal("disabling should not open stopped Claude"); return nil },
	)

	if err := c.SetInstalledFromDesktop(false, false); err != nil {
		t.Fatalf("SetInstalledFromDesktop returned error: %v", err)
	}
	if c.AutodiscoveryConfigured() {
		t.Fatal("Claude gateway remains configured after disabling")
	}
}

func TestClaudeDesktopRestoreForShutdownDoesNotReopenApp(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withClaudeDesktopPlatform(t, "darwin")
	c := &ClaudeDesktop{}
	if err := c.ConfigureAutodiscovery(); err != nil {
		t.Fatal(err)
	}
	paths, err := claudeDesktopConfigPaths()
	if err != nil {
		t.Fatal(err)
	}

	running := true
	quitCalls := 0
	withClaudeDesktopProcessHooks(t,
		func() bool { return running },
		func() error {
			quitCalls++
			running = false
			return os.WriteFile(paths.profile, []byte(`{"provider":"ollama"}`), 0o644)
		},
		func() error {
			t.Fatal("Claude reopened during shutdown")
			return nil
		},
	)

	if err := c.RestoreForShutdown(context.Background()); err != nil {
		t.Fatal(err)
	}
	if quitCalls != 1 {
		t.Fatalf("quit calls = %d, want 1", quitCalls)
	}
	if c.UsesOllamaGateway() {
		t.Fatal("Claude gateway remains configured after shutdown restore")
	}
}

func TestClaudeDesktopRestoreForShutdownBoundsHungQuit(t *testing.T) {
	withClaudeDesktopPlatform(t, "darwin")
	oldRunning := claudeDesktopIsRunning
	oldQuit := claudeDesktopQuitApp
	claudeDesktopIsRunning = func(context.Context) (bool, error) { return true, nil }
	claudeDesktopQuitApp = func(ctx context.Context) error {
		<-ctx.Done()
		return ctx.Err()
	}
	t.Cleanup(func() {
		claudeDesktopIsRunning = oldRunning
		claudeDesktopQuitApp = oldQuit
	})

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	err := (&ClaudeDesktop{}).RestoreForShutdown(ctx)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("RestoreForShutdown error = %v, want deadline exceeded", err)
	}
}

func TestClaudeDesktopRestoreForShutdownBoundsExitWait(t *testing.T) {
	withClaudeDesktopPlatform(t, "darwin")
	oldRunning := claudeDesktopIsRunning
	oldQuit := claudeDesktopQuitApp
	claudeDesktopIsRunning = func(context.Context) (bool, error) { return true, nil }
	claudeDesktopQuitApp = func(context.Context) error { return nil }
	t.Cleanup(func() {
		claudeDesktopIsRunning = oldRunning
		claudeDesktopQuitApp = oldQuit
	})

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	err := (&ClaudeDesktop{}).RestoreForShutdown(ctx)
	if err == nil || !strings.Contains(err.Error(), "did not quit") {
		t.Fatalf("RestoreForShutdown error = %v, want bounded exit error", err)
	}
}

func TestClaudeDesktopRestoreForShutdownPreservesProfileWhenInitialCheckExpires(t *testing.T) {
	setTestHome(t, t.TempDir())
	withClaudeDesktopPlatform(t, "darwin")
	c := &ClaudeDesktop{}
	if err := c.ConfigureAutodiscovery(); err != nil {
		t.Fatal(err)
	}

	oldRunning := claudeDesktopIsRunning
	oldQuit := claudeDesktopQuitApp
	claudeDesktopIsRunning = func(ctx context.Context) (bool, error) {
		<-ctx.Done()
		return false, ctx.Err()
	}
	claudeDesktopQuitApp = func(context.Context) error {
		t.Fatal("quit called after an indeterminate process check")
		return nil
	}
	t.Cleanup(func() {
		claudeDesktopIsRunning = oldRunning
		claudeDesktopQuitApp = oldQuit
	})

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	err := c.RestoreForShutdown(ctx)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("RestoreForShutdown error = %v, want deadline exceeded", err)
	}
	if !c.UsesOllamaGateway() {
		t.Fatal("indeterminate initial process check restored the profile")
	}
}

func TestClaudeDesktopRestoreForShutdownPreservesProfileWhenExitCheckExpires(t *testing.T) {
	setTestHome(t, t.TempDir())
	withClaudeDesktopPlatform(t, "darwin")
	c := &ClaudeDesktop{}
	if err := c.ConfigureAutodiscovery(); err != nil {
		t.Fatal(err)
	}

	oldRunning := claudeDesktopIsRunning
	oldQuit := claudeDesktopQuitApp
	checks := 0
	claudeDesktopIsRunning = func(ctx context.Context) (bool, error) {
		checks++
		if checks == 1 {
			return true, nil
		}
		<-ctx.Done()
		return false, ctx.Err()
	}
	claudeDesktopQuitApp = func(context.Context) error { return nil }
	t.Cleanup(func() {
		claudeDesktopIsRunning = oldRunning
		claudeDesktopQuitApp = oldQuit
	})

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	err := c.RestoreForShutdown(ctx)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("RestoreForShutdown error = %v, want deadline exceeded", err)
	}
	if !c.UsesOllamaGateway() {
		t.Fatal("indeterminate exit process check restored the profile")
	}
}

func TestClaudeDesktopRunRejectsExtraArgs(t *testing.T) {
	withClaudeDesktopPlatform(t, "darwin")
	err := (&ClaudeDesktop{}).Run("qwen3.5", nil, []string{"--foo"})
	if err == nil || !strings.Contains(err.Error(), "does not accept extra arguments") {
		t.Fatalf("Run error = %v, want extra args rejection", err)
	}
}
