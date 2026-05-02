package launch

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func withClaudeDesktopPlatform(t *testing.T, goos string) {
	t.Helper()
	old := claudeDesktopGOOS
	claudeDesktopGOOS = goos
	t.Cleanup(func() {
		claudeDesktopGOOS = old
	})
}

func withClaudeDesktopValidation(t *testing.T, fn func(context.Context, string) error) {
	t.Helper()
	old := claudeDesktopValidateAPIKey
	claudeDesktopValidateAPIKey = fn
	t.Cleanup(func() {
		claudeDesktopValidateAPIKey = old
	})
}

func withClaudeDesktopPrompt(t *testing.T, fn func() (string, error)) {
	t.Helper()
	old := claudeDesktopPromptAPIKey
	claudeDesktopPromptAPIKey = fn
	t.Cleanup(func() {
		claudeDesktopPromptAPIKey = old
	})
}

func withClaudeDesktopProcessHooks(t *testing.T, running func() bool, quit func() error, open func() error) {
	t.Helper()
	oldRunning := claudeDesktopIsRunning
	oldQuit := claudeDesktopQuitApp
	oldOpen := claudeDesktopOpenApp
	oldOpenPath := claudeDesktopOpenAppPath
	oldRunningPath := claudeDesktopRunningAppPath
	oldSleep := claudeDesktopSleep
	claudeDesktopIsRunning = running
	claudeDesktopQuitApp = quit
	claudeDesktopOpenApp = open
	claudeDesktopOpenAppPath = oldOpenPath
	claudeDesktopRunningAppPath = oldRunningPath
	claudeDesktopSleep = func(time.Duration) {}
	t.Cleanup(func() {
		claudeDesktopIsRunning = oldRunning
		claudeDesktopQuitApp = oldQuit
		claudeDesktopOpenApp = oldOpen
		claudeDesktopOpenAppPath = oldOpenPath
		claudeDesktopRunningAppPath = oldRunningPath
		claudeDesktopSleep = oldSleep
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

func TestClaudeDesktopIntegration(t *testing.T) {
	c := &ClaudeDesktop{}

	t.Run("implements Runner", func(t *testing.T) {
		var _ Runner = c
	})
	t.Run("implements managed autodiscovery integration", func(t *testing.T) {
		var _ ManagedAutodiscoveryIntegration = c
	})
	t.Run("uses Ollama Cloud", func(t *testing.T) {
		var _ ManagedAutodiscoveryCloudIntegration = c
		if !c.UsesOllamaCloud() {
			t.Fatal("expected Claude Desktop autodiscovery to require Ollama Cloud")
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
	t.Run("skips local model readiness", func(t *testing.T) {
		var _ ManagedModelReadinessSkipper = c
		if !c.SkipModelReadiness() {
			t.Fatal("expected Claude Desktop to skip local model readiness")
		}
	})
}

func TestClaudeDesktopConfigureWritesOllamaCloudProfile(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withClaudeDesktopPlatform(t, "darwin")
	t.Setenv("OLLAMA_API_KEY", "test-api-key")

	var validatedKey string
	withClaudeDesktopValidation(t, func(_ context.Context, key string) error {
		validatedKey = key
		return nil
	})

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
	if validatedKey != "test-api-key" {
		t.Fatalf("validated key = %q, want test API key", validatedKey)
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
	if profile["inferenceGatewayApiKey"] != "test-api-key" {
		t.Fatal("expected configured API key to be written")
	}
	if profile["inferenceGatewayAuthScheme"] != "bearer" {
		t.Fatalf("auth scheme = %v, want bearer", profile["inferenceGatewayAuthScheme"])
	}
	if profile["disableDeploymentModeChooser"] != true {
		t.Fatalf("disableDeploymentModeChooser = %v, want true", profile["disableDeploymentModeChooser"])
	}
	if _, ok := profile["inferenceModels"]; ok {
		t.Fatalf("inferenceModels should be omitted so Claude can discover models, got %v", profile["inferenceModels"])
	}
}

func TestClaudeDesktopConfigureAutodiscoveryRemovesExistingModelCatalog(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withClaudeDesktopPlatform(t, "darwin")
	t.Setenv("OLLAMA_API_KEY", "test-api-key")
	withClaudeDesktopValidation(t, func(context.Context, string) error { return nil })

	paths, err := claudeDesktopConfigPaths()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(paths.profile), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(paths.profile, []byte(`{"inferenceModels":["qwen3.5"],"inferenceGatewayApiKey":"old"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := (&ClaudeDesktop{}).ConfigureAutodiscovery(); err != nil {
		t.Fatalf("ConfigureAutodiscovery returned error: %v", err)
	}

	profile := claudeDesktopReadJSON(t, paths.profile)
	if _, ok := profile["inferenceModels"]; ok {
		t.Fatalf("inferenceModels should be removed, got %v", profile["inferenceModels"])
	}
	if profile["inferenceGatewayApiKey"] != "test-api-key" {
		t.Fatal("expected env API key to replace the old key")
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

func TestClaudeDesktopAutodiscoveryConfiguredOnWindows(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withClaudeDesktopPlatform(t, "windows")
	t.Setenv("LOCALAPPDATA", filepath.Join(tmpDir, "LocalAppData"))
	t.Setenv("OLLAMA_API_KEY", "test-api-key")
	withClaudeDesktopValidation(t, func(context.Context, string) error { return nil })

	c := &ClaudeDesktop{}
	if err := c.ConfigureAutodiscovery(); err != nil {
		t.Fatalf("Configure returned error: %v", err)
	}
	if !c.AutodiscoveryConfigured() {
		t.Fatal("expected Claude Desktop autodiscovery config to be detected on Windows")
	}
}

func TestClaudeDesktopConfigureAutodiscoveryTouchesAllWindowsProfileCandidates(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withClaudeDesktopPlatform(t, "windows")
	local := filepath.Join(tmpDir, "LocalAppData")
	t.Setenv("LOCALAPPDATA", local)
	t.Setenv("OLLAMA_API_KEY", "test-api-key")
	withClaudeDesktopValidation(t, func(context.Context, string) error { return nil })

	targets, err := claudeDesktopTargetPaths()
	if err != nil {
		t.Fatal(err)
	}
	if len(targets.normalConfigs) != 2 {
		t.Fatalf("normal config target count = %d, want 2", len(targets.normalConfigs))
	}
	if len(targets.thirdPartyProfiles) != 2 {
		t.Fatalf("third-party target count = %d, want 2", len(targets.thirdPartyProfiles))
	}

	c := &ClaudeDesktop{}
	if err := c.ConfigureAutodiscovery(); err != nil {
		t.Fatalf("ConfigureAutodiscovery returned error: %v", err)
	}

	for _, path := range targets.normalConfigs {
		cfg := claudeDesktopReadJSON(t, path)
		if cfg["deploymentMode"] != "3p" {
			t.Fatalf("%s deploymentMode = %v, want 3p", path, cfg["deploymentMode"])
		}
	}
	for _, target := range targets.thirdPartyProfiles {
		cfg := claudeDesktopReadJSON(t, target.desktopConfig)
		if cfg["deploymentMode"] != "3p" {
			t.Fatalf("%s deploymentMode = %v, want 3p", target.desktopConfig, cfg["deploymentMode"])
		}
		meta := claudeDesktopReadJSON(t, target.meta)
		if meta["appliedId"] != claudeDesktopProfileID {
			t.Fatalf("%s appliedId = %v, want %s", target.meta, meta["appliedId"], claudeDesktopProfileID)
		}
		profile := claudeDesktopReadJSON(t, target.profile)
		if profile["inferenceProvider"] != "gateway" {
			t.Fatalf("%s inferenceProvider = %v, want gateway", target.profile, profile["inferenceProvider"])
		}
		if profile["inferenceGatewayBaseUrl"] != claudeDesktopGatewayBaseURL {
			t.Fatalf("%s base URL = %v, want %s", target.profile, profile["inferenceGatewayBaseUrl"], claudeDesktopGatewayBaseURL)
		}
		if profile["inferenceGatewayApiKey"] != "test-api-key" {
			t.Fatalf("%s should contain the configured API key", target.profile)
		}
		if _, ok := profile["inferenceModels"]; ok {
			t.Fatalf("%s inferenceModels should be omitted, got %v", target.profile, profile["inferenceModels"])
		}
	}
	if !c.AutodiscoveryConfigured() {
		t.Fatal("expected all Windows profile candidates to be considered configured")
	}

	if err := writeClaudeDesktopDeploymentMode(targets.thirdPartyProfiles[1].desktopConfig, "1p"); err != nil {
		t.Fatal(err)
	}
	if c.AutodiscoveryConfigured() {
		t.Fatal("expected a stale Windows candidate to force reconfiguration")
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

	if !claudeDesktopInstalled() {
		t.Fatal("expected Claude Desktop to be installed when the Windows profile directory exists")
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

	if err := waitForClaudeDesktopExit(time.Second); err != nil {
		t.Fatalf("waitForClaudeDesktopExit returned error: %v", err)
	}
	if runningChecks < 2 {
		t.Fatalf("expected running hook to be checked until the visible window exits, got %d checks", runningChecks)
	}
}

func TestClaudeDesktopWindowsRestartUsesCapturedDesktopPath(t *testing.T) {
	withClaudeDesktopPlatform(t, "windows")
	restoreConfirm := withLaunchConfirmPolicy(launchConfirmPolicy{yes: true})
	defer restoreConfirm()

	desktopPath := `C:\Users\parth\AppData\Local\AnthropicClaude\app-1.2.3\Claude.exe`
	running := true
	var openedPath string
	withClaudeDesktopProcessHooks(t,
		func() bool { return running },
		func() error {
			running = false
			return nil
		},
		func() error {
			t.Fatal("expected restart to open the captured Desktop executable path, not the generic launcher")
			return nil
		},
	)
	claudeDesktopRunningAppPath = func() string { return desktopPath }
	claudeDesktopOpenAppPath = func(path string) error {
		openedPath = path
		return nil
	}

	if err := (&ClaudeDesktop{}).Run("qwen3.5", nil); err != nil {
		t.Fatalf("Run returned error: %v", err)
	}
	if openedPath != desktopPath {
		t.Fatalf("opened path = %q, want %q", openedPath, desktopPath)
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
	if err == nil || !strings.Contains(err.Error(), "Claude App executable was not found") {
		t.Fatalf("defaultClaudeDesktopOpenApp error = %v, want executable-not-found error", err)
	}
}

func TestClaudeDesktopConfigureStopsBeforeWriteWhenKeyValidationFails(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withClaudeDesktopPlatform(t, "darwin")
	t.Setenv("OLLAMA_API_KEY", "bad-key")
	withClaudeDesktopValidation(t, func(context.Context, string) error {
		return errors.New("invalid key")
	})

	err := (&ClaudeDesktop{}).ConfigureAutodiscovery()
	if err == nil || !strings.Contains(err.Error(), "invalid key") {
		t.Fatalf("Configure error = %v, want invalid key", err)
	}

	paths, err := claudeDesktopConfigPaths()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(paths.desktopConfig); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("desktop config should not be written after validation failure, stat err = %v", err)
	}
}

func TestValidateClaudeDesktopAPIKeyUsesClaudeModelsRoute(t *testing.T) {
	oldClient := claudeDesktopHTTPClient
	var gotPath, gotAuth string
	claudeDesktopHTTPClient = &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		gotPath = req.URL.Path
		gotAuth = req.Header.Get("Authorization")
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader(`{"data":[]}`)),
			Header:     make(http.Header),
		}, nil
	})}
	t.Cleanup(func() {
		claudeDesktopHTTPClient = oldClient
	})

	if err := validateClaudeDesktopAPIKey(context.Background(), "test-key"); err != nil {
		t.Fatalf("validateClaudeDesktopAPIKey returned error: %v", err)
	}
	if gotPath != "/v1/models" {
		t.Fatalf("validation path = %q, want /v1/models", gotPath)
	}
	if gotAuth != "Bearer test-key" {
		t.Fatalf("Authorization header = %q, want bearer key", gotAuth)
	}
}

func TestClaudeDesktopConfigureRequiresAPIKey(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withClaudeDesktopPlatform(t, "darwin")
	t.Setenv("OLLAMA_API_KEY", "")
	withClaudeDesktopValidation(t, func(context.Context, string) error {
		t.Fatal("validation should not run without an API key")
		return nil
	})

	err := (&ClaudeDesktop{}).ConfigureAutodiscovery()
	if err == nil || !strings.Contains(err.Error(), "OLLAMA_API_KEY is required") {
		t.Fatalf("Configure error = %v, want missing key guidance", err)
	}
}

func TestClaudeDesktopConfigureReusesExistingAPIKey(t *testing.T) {
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

	var validatedKey string
	withClaudeDesktopValidation(t, func(_ context.Context, key string) error {
		validatedKey = key
		return nil
	})

	if err := (&ClaudeDesktop{}).ConfigureAutodiscovery(); err != nil {
		t.Fatalf("ConfigureAutodiscovery returned error: %v", err)
	}
	if validatedKey != "existing-key" {
		t.Fatalf("validated key = %q, want existing-key", validatedKey)
	}
}

func TestClaudeDesktopConfigureReplacesInvalidExistingAPIKey(t *testing.T) {
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

	var validated []string
	withClaudeDesktopValidation(t, func(_ context.Context, key string) error {
		validated = append(validated, key)
		if key == "stale-key" {
			return errors.New("invalid key")
		}
		return nil
	})
	withClaudeDesktopPrompt(t, func() (string, error) {
		return "replacement-key", nil
	})

	if err := (&ClaudeDesktop{}).ConfigureAutodiscovery(); err != nil {
		t.Fatalf("ConfigureAutodiscovery returned error: %v", err)
	}
	if diff := compareStrings(validated, []string{"stale-key", "replacement-key"}); diff != "" {
		t.Fatalf("validated keys mismatch: %s", diff)
	}
	profile := claudeDesktopReadJSON(t, paths.profile)
	if profile["inferenceGatewayApiKey"] != "replacement-key" {
		t.Fatalf("configured key = %v, want replacement-key", profile["inferenceGatewayApiKey"])
	}
}

func TestClaudeDesktopConfigureReusesExistingAPIKeyFromAnyWindowsProfile(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withClaudeDesktopPlatform(t, "windows")
	local := filepath.Join(tmpDir, "LocalAppData")
	t.Setenv("LOCALAPPDATA", local)
	t.Setenv("OLLAMA_API_KEY", "")

	targets, err := claudeDesktopTargetPaths()
	if err != nil {
		t.Fatal(err)
	}
	fallbackProfile := targets.thirdPartyProfiles[1].profile
	if err := os.MkdirAll(filepath.Dir(fallbackProfile), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(fallbackProfile, []byte(`{"inferenceGatewayApiKey":"fallback-key"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	var validatedKey string
	withClaudeDesktopValidation(t, func(_ context.Context, key string) error {
		validatedKey = key
		return nil
	})

	if err := (&ClaudeDesktop{}).ConfigureAutodiscovery(); err != nil {
		t.Fatalf("ConfigureAutodiscovery returned error: %v", err)
	}
	if validatedKey != "fallback-key" {
		t.Fatalf("validated key = %q, want fallback-key", validatedKey)
	}
	for _, target := range targets.thirdPartyProfiles {
		profile := claudeDesktopReadJSON(t, target.profile)
		if profile["inferenceGatewayApiKey"] != "fallback-key" {
			t.Fatalf("%s should reuse fallback key, got %v", target.profile, profile["inferenceGatewayApiKey"])
		}
	}
}

func TestClaudeDesktopAutodiscoveryConfiguredRequiresAppliedOllamaProfile(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withClaudeDesktopPlatform(t, "darwin")
	t.Setenv("OLLAMA_API_KEY", "test-api-key")
	withClaudeDesktopValidation(t, func(context.Context, string) error { return nil })

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
}

func TestClaudeDesktopAutodiscoveryConfiguredRequiresAPIKey(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withClaudeDesktopPlatform(t, "darwin")
	t.Setenv("OLLAMA_API_KEY", "test-api-key")
	withClaudeDesktopValidation(t, func(context.Context, string) error { return nil })

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
	if err := os.WriteFile(paths.profile, []byte(`{"disableDeploymentModeChooser":true,"inferenceGatewayApiKey":"keep","inferenceProvider":"gateway","inferenceGatewayBaseUrl":"https://ollama.com","inferenceGatewayAuthScheme":"bearer","inferenceModels":["legacy"]}`), 0o644); err != nil {
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
	for _, key := range []string{"inferenceProvider", "inferenceGatewayBaseUrl", "inferenceGatewayAuthScheme", "inferenceModels"} {
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

func TestClaudeDesktopRestoreTouchesAllWindowsProfileCandidates(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withClaudeDesktopPlatform(t, "windows")
	local := filepath.Join(tmpDir, "LocalAppData")
	t.Setenv("LOCALAPPDATA", local)
	withClaudeDesktopProcessHooks(t, func() bool { return false }, func() error { return nil }, func() error { return nil })

	targets, err := claudeDesktopTargetPaths()
	if err != nil {
		t.Fatal(err)
	}
	if len(targets.normalConfigs) != 2 {
		t.Fatalf("normal config target count = %d, want 2", len(targets.normalConfigs))
	}
	if len(targets.thirdPartyProfiles) != 2 {
		t.Fatalf("third-party target count = %d, want 2", len(targets.thirdPartyProfiles))
	}
	for _, target := range targets.thirdPartyProfiles {
		if err := os.MkdirAll(filepath.Dir(target.profile), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(target.meta, []byte(`{"appliedId":"`+claudeDesktopProfileID+`","entries":[{"id":"`+claudeDesktopProfileID+`","name":"Ollama"}]}`), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(target.profile, []byte(`{"disableDeploymentModeChooser":true,"inferenceGatewayApiKey":"keep","inferenceProvider":"gateway","inferenceGatewayBaseUrl":"https://ollama.com","inferenceGatewayAuthScheme":"bearer","inferenceModels":["legacy"]}`), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	if err := (&ClaudeDesktop{}).Restore(); err != nil {
		t.Fatalf("Restore returned error: %v", err)
	}

	for _, path := range targets.normalConfigs {
		cfg := claudeDesktopReadJSON(t, path)
		if cfg["deploymentMode"] != "1p" {
			t.Fatalf("%s deploymentMode = %v, want 1p", path, cfg["deploymentMode"])
		}
	}
	for _, target := range targets.thirdPartyProfiles {
		cfg := claudeDesktopReadJSON(t, target.desktopConfig)
		if cfg["deploymentMode"] != "1p" {
			t.Fatalf("%s deploymentMode = %v, want 1p", target.desktopConfig, cfg["deploymentMode"])
		}
		meta := claudeDesktopReadJSON(t, target.meta)
		if _, ok := meta["appliedId"]; ok {
			t.Fatalf("%s should not keep the Ollama applied profile: %v", target.meta, meta)
		}
		profile := claudeDesktopReadJSON(t, target.profile)
		if profile["disableDeploymentModeChooser"] != false {
			t.Fatalf("%s disableDeploymentModeChooser = %v, want false", target.profile, profile["disableDeploymentModeChooser"])
		}
		if profile["inferenceGatewayApiKey"] != "keep" {
			t.Fatalf("%s should preserve gateway API key", target.profile)
		}
		for _, key := range []string{"inferenceProvider", "inferenceGatewayBaseUrl", "inferenceGatewayAuthScheme", "inferenceModels"} {
			if _, ok := profile[key]; ok {
				t.Fatalf("%s should clear stale %s: %v", target.profile, key, profile)
			}
		}
	}
}

func TestClaudeDesktopRunRestartsRunningAppWhenConfirmed(t *testing.T) {
	withClaudeDesktopPlatform(t, "darwin")
	restoreConfirm := withLaunchConfirmPolicy(launchConfirmPolicy{yes: true})
	defer restoreConfirm()

	running := true
	var quitCalls, openCalls int
	withClaudeDesktopProcessHooks(t,
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

	if err := (&ClaudeDesktop{}).Run("qwen3.5", nil); err != nil {
		t.Fatalf("Run returned error: %v", err)
	}
	if quitCalls != 1 || openCalls != 1 {
		t.Fatalf("quit/open calls = %d/%d, want 1/1", quitCalls, openCalls)
	}
}

func TestClaudeDesktopRunRejectsExtraArgs(t *testing.T) {
	withClaudeDesktopPlatform(t, "darwin")
	err := (&ClaudeDesktop{}).Run("qwen3.5", []string{"--foo"})
	if err == nil || !strings.Contains(err.Error(), "does not accept extra arguments") {
		t.Fatalf("Run error = %v, want extra args rejection", err)
	}
}
