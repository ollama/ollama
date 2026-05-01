package launch

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

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

func withClaudeDesktopProcessHooks(t *testing.T, running func() bool, quit func() error, open func() error) {
	t.Helper()
	oldRunning := claudeDesktopIsRunning
	oldQuit := claudeDesktopQuitApp
	oldOpen := claudeDesktopOpenApp
	oldSleep := claudeDesktopSleep
	claudeDesktopIsRunning = running
	claudeDesktopQuitApp = quit
	claudeDesktopOpenApp = open
	claudeDesktopSleep = func(time.Duration) {}
	t.Cleanup(func() {
		claudeDesktopIsRunning = oldRunning
		claudeDesktopQuitApp = oldQuit
		claudeDesktopOpenApp = oldOpen
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
	t.Run("implements managed single model", func(t *testing.T) {
		var _ ManagedSingleModel = c
	})
	t.Run("implements managed model list configurer", func(t *testing.T) {
		var _ ManagedModelListConfigurer = c
	})
	t.Run("implements restore", func(t *testing.T) {
		var _ RestorableIntegration = c
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

	if err := (&ClaudeDesktop{}).Configure("gpt-oss:120b-cloud"); err != nil {
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
	models := claudeDesktopStringSlice(profile["inferenceModels"])
	if len(models) != 1 || models[0] != "gpt-oss:120b" {
		t.Fatalf("models = %v, want [gpt-oss:120b]", models)
	}
}

func TestClaudeDesktopConfigureWithModelsWritesOllamaCloudModelCatalog(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withClaudeDesktopPlatform(t, "darwin")
	t.Setenv("OLLAMA_API_KEY", "test-api-key")
	withClaudeDesktopValidation(t, func(context.Context, string) error { return nil })

	if err := (&ClaudeDesktop{}).ConfigureWithModels("qwen3.5:cloud", []string{
		"glm-5.1:cloud",
		"qwen3.5:cloud",
		"gemma4",
		"minimax-m2.7:cloud",
	}); err != nil {
		t.Fatalf("ConfigureWithModels returned error: %v", err)
	}

	paths, err := claudeDesktopConfigPaths()
	if err != nil {
		t.Fatal(err)
	}
	profile := claudeDesktopReadJSON(t, paths.profile)
	models := claudeDesktopStringSlice(profile["inferenceModels"])
	want := []string{"qwen3.5", "glm-5.1", "minimax-m2.7"}
	if diff := compareStrings(models, want); diff != "" {
		t.Fatalf("models mismatch (-want +got):\n%s", diff)
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

	err := (&ClaudeDesktop{}).Configure("qwen3.5:cloud")
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

func TestClaudeDesktopConfigureRequiresAPIKey(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withClaudeDesktopPlatform(t, "darwin")
	t.Setenv("OLLAMA_API_KEY", "")
	withClaudeDesktopValidation(t, func(context.Context, string) error {
		t.Fatal("validation should not run without an API key")
		return nil
	})

	err := (&ClaudeDesktop{}).Configure("qwen3.5:cloud")
	if err == nil || !strings.Contains(err.Error(), "OLLAMA_API_KEY is required") {
		t.Fatalf("Configure error = %v, want missing key guidance", err)
	}
}

func TestClaudeDesktopCurrentModelReadsAppliedOllamaProfile(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)
	withClaudeDesktopPlatform(t, "darwin")
	t.Setenv("OLLAMA_API_KEY", "test-api-key")
	withClaudeDesktopValidation(t, func(context.Context, string) error { return nil })

	c := &ClaudeDesktop{}
	if err := c.Configure("qwen3.5:cloud"); err != nil {
		t.Fatalf("Configure returned error: %v", err)
	}
	if got := c.CurrentModel(); got != "qwen3.5" {
		t.Fatalf("CurrentModel() = %q, want qwen3.5", got)
	}

	paths, err := claudeDesktopConfigPaths()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(paths.meta, []byte(`{"appliedId":"custom"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := c.CurrentModel(); got != "" {
		t.Fatalf("CurrentModel() with another applied profile = %q, want empty", got)
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
	if err := os.WriteFile(paths.profile, []byte(`{"disableDeploymentModeChooser":true,"inferenceGatewayApiKey":"keep"}`), 0o644); err != nil {
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
