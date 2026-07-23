package launch

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/cmd/internal/fileutil"
)

type launcherEditorRunner struct {
	paths    []string
	models   []string
	edited   [][]string
	ranModel string
}

func (r *launcherEditorRunner) Run(model string, _ []LaunchModel, args []string) error {
	r.ranModel = model
	return nil
}

func (r *launcherEditorRunner) String() string { return "LauncherEditor" }

func (r *launcherEditorRunner) Paths() []string { return r.paths }

func (r *launcherEditorRunner) Edit(models []LaunchModel) error {
	names := launchModelNames(models)
	r.edited = append(r.edited, names)
	return nil
}

func (r *launcherEditorRunner) Models() []string {
	return append([]string(nil), r.models...)
}

type launcherSingleRunner struct {
	ranModel string
}

func (r *launcherSingleRunner) Run(model string, _ []LaunchModel, args []string) error {
	r.ranModel = model
	return nil
}

func (r *launcherSingleRunner) String() string { return "StubSingle" }

func selectionItemNames(items []SelectionItem) []string {
	names := make([]string, 0, len(items))
	for _, item := range items {
		names = append(names, item.Name)
	}
	return names
}

type launcherRestorableRunner struct {
	launcherSingleRunner
	restored              bool
	restoreErr            error
	restoreSuccessMessage string
}

func (r *launcherRestorableRunner) Restore() error {
	r.restored = true
	return r.restoreErr
}

func (r *launcherRestorableRunner) RestoreSuccessMessage() string {
	return r.restoreSuccessMessage
}

type launcherManagedRunner struct {
	paths                []string
	currentModel         string
	configured           []string
	ranModel             string
	onboarded            bool
	onboardCalls         int
	onboardingComplete   bool
	refreshCalls         int
	refreshErr           error
	restoreHint          string
	configSuccessMessage string
	skipModelReadiness   bool
}

func (r *launcherManagedRunner) Run(model string, _ []LaunchModel, args []string) error {
	r.ranModel = model
	return nil
}

func (r *launcherManagedRunner) String() string { return "StubManaged" }

func (r *launcherManagedRunner) Paths() []string { return r.paths }

func (r *launcherManagedRunner) Configure(model string) error {
	r.configured = append(r.configured, model)
	r.currentModel = model
	return nil
}

func (r *launcherManagedRunner) CurrentModel() string { return r.currentModel }

func (r *launcherManagedRunner) Onboard() error {
	r.onboardCalls++
	r.onboarded = true
	r.onboardingComplete = true
	return nil
}

func (r *launcherManagedRunner) OnboardingComplete() bool { return r.onboardingComplete }

func (r *launcherManagedRunner) RefreshRuntimeAfterConfigure() error {
	r.refreshCalls++
	return r.refreshErr
}

func (r *launcherManagedRunner) RestoreHint() string { return r.restoreHint }

func (r *launcherManagedRunner) ConfigurationSuccessMessage() string {
	return r.configSuccessMessage
}

func (r *launcherManagedRunner) SkipModelReadiness() bool { return r.skipModelReadiness }

type launcherHeadlessManagedRunner struct {
	launcherManagedRunner
}

func (r *launcherHeadlessManagedRunner) RequiresInteractiveOnboarding() bool { return false }

type launcherManagedListRunner struct {
	launcherManagedRunner
	configuredModelLists [][]string
}

func (r *launcherManagedListRunner) ConfigureWithModels(primary string, models []LaunchModel) error {
	r.configuredModelLists = append(r.configuredModelLists, launchModelNames(models))
	return r.Configure(primary)
}

type launcherManagedAutodiscoveryRunner struct {
	launcherManagedRunner
	autodiscoveryConfigures int
	autodiscoveryConfigured bool
	usesCloud               bool
	restoreHint             string
	configSuccessMessage    string
}

func (r *launcherManagedAutodiscoveryRunner) AutodiscoveredModel() string { return "Ollama Cloud" }

func (r *launcherManagedAutodiscoveryRunner) UsesOllamaCloud() bool { return r.usesCloud }

func (r *launcherManagedAutodiscoveryRunner) RestoreHint() string { return r.restoreHint }

func (r *launcherManagedAutodiscoveryRunner) ConfigurationSuccessMessage() string {
	return r.configSuccessMessage
}

func (r *launcherManagedAutodiscoveryRunner) AutodiscoveryConfigured() bool {
	return r.autodiscoveryConfigured
}

func (r *launcherManagedAutodiscoveryRunner) ConfigureAutodiscovery() error {
	r.autodiscoveryConfigures++
	r.autodiscoveryConfigured = true
	return nil
}

func setLaunchTestHome(t *testing.T, dir string) {
	t.Helper()
	t.Setenv("HOME", dir)
	t.Setenv("TMPDIR", dir)
	t.Setenv("USERPROFILE", dir)
}

func writeFakeBinary(t *testing.T, dir, name string) {
	t.Helper()
	path := filepath.Join(dir, name)
	data := []byte("#!/bin/sh\nexit 0\n")
	if runtime.GOOS == "windows" {
		path += ".cmd"
		data = []byte("@echo off\r\nexit /b 0\r\n")
	}
	if err := os.WriteFile(path, data, 0o755); err != nil {
		t.Fatalf("failed to write fake binary: %v", err)
	}
}

func withIntegrationOverride(t *testing.T, name string, runner Runner) {
	t.Helper()
	restore := OverrideIntegration(name, runner)
	t.Cleanup(restore)
}

func withInteractiveSession(t *testing.T, interactive bool) {
	t.Helper()
	old := isInteractiveSession
	isInteractiveSession = func() bool { return interactive }
	t.Cleanup(func() {
		isInteractiveSession = old
	})
}

func withLauncherHooks(t *testing.T) {
	t.Helper()
	oldSingle := DefaultSingleSelector
	oldSingleWithUpdates := DefaultSingleSelectorWithUpdates
	oldMulti := DefaultMultiSelector
	oldMultiWithUpdates := DefaultMultiSelectorWithUpdates
	oldConfirm := DefaultConfirmPrompt
	oldSignIn := DefaultSignIn
	oldUpgrade := DefaultUpgrade
	t.Cleanup(func() {
		DefaultSingleSelector = oldSingle
		DefaultSingleSelectorWithUpdates = oldSingleWithUpdates
		DefaultMultiSelector = oldMulti
		DefaultMultiSelectorWithUpdates = oldMultiWithUpdates
		DefaultConfirmPrompt = oldConfirm
		DefaultSignIn = oldSignIn
		DefaultUpgrade = oldUpgrade
	})
}

func TestDefaultLaunchPolicy(t *testing.T) {
	tests := []struct {
		name        string
		interactive bool
		yes         bool
		want        LaunchPolicy
	}{
		{
			name:        "interactive default prompts and prompt-pull",
			interactive: true,
			yes:         false,
			want:        LaunchPolicy{Confirm: LaunchConfirmPrompt, MissingModel: LaunchMissingModelPromptToPull},
		},
		{
			name:        "headless without yes requires yes and fail-missing",
			interactive: false,
			yes:         false,
			want:        LaunchPolicy{Confirm: LaunchConfirmRequireYes, MissingModel: LaunchMissingModelFail},
		},
		{
			name:        "interactive yes auto-approves and auto-pulls",
			interactive: true,
			yes:         true,
			want:        LaunchPolicy{Confirm: LaunchConfirmAutoApprove, MissingModel: LaunchMissingModelAutoPull},
		},
		{
			name:        "headless yes auto-approves and auto-pulls",
			interactive: false,
			yes:         true,
			want:        LaunchPolicy{Confirm: LaunchConfirmAutoApprove, MissingModel: LaunchMissingModelAutoPull},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := defaultLaunchPolicy(tt.interactive, tt.yes)
			if got != tt.want {
				t.Fatalf("defaultLaunchPolicy(%v, %v) = %+v, want %+v", tt.interactive, tt.yes, got, tt.want)
			}
		})
	}
}

func TestBuildLauncherState_ManagedSingleIntegrationUsesCurrentModel(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"gemma4"}]}`)
		case "/api/show":
			fmt.Fprint(w, `{"model_info":{"general.context_length":131072}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	runner := &launcherManagedRunner{currentModel: "gemma4"}
	withIntegrationOverride(t, "pi", runner)

	state, err := BuildLauncherState(context.Background())
	if err != nil {
		t.Fatalf("BuildLauncherState returned error: %v", err)
	}

	if state.Integrations["pi"].CurrentModel != "gemma4" {
		t.Fatalf("expected managed current model from integration config, got %q", state.Integrations["pi"].CurrentModel)
	}
	if !state.Integrations["pi"].ModelUsable {
		t.Fatal("expected managed current model to be usable")
	}
}

func TestBuildLauncherState_DeprecatedSavedModelIsUsable(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)

	if err := config.SetLastModel("qwen2.5:14b"); err != nil {
		t.Fatalf("failed to seed last model: %v", err)
	}
	if err := config.SaveIntegration("codex", []string{"llama3.2:latest"}); err != nil {
		t.Fatalf("failed to seed codex config: %v", err)
	}

	var showCalls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"qwen2.5:14b"},{"name":"llama3.2:latest"}]}`)
		case "/api/show":
			showCalls.Add(1)
			fmt.Fprint(w, `{"model_info":{"general.context_length":131072}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	state, err := BuildLauncherState(context.Background())
	if err != nil {
		t.Fatalf("BuildLauncherState returned error: %v", err)
	}
	if !state.RunModelUsable {
		t.Fatal("expected deprecated saved run model to stay usable")
	}
	if state.Integrations["codex"].CurrentModel != "llama3.2:latest" {
		t.Fatalf("expected saved integration model to remain visible, got %q", state.Integrations["codex"].CurrentModel)
	}
	if !state.Integrations["codex"].ModelUsable {
		t.Fatal("expected deprecated saved integration model to stay usable")
	}
	if showCalls.Load() != 0 {
		t.Fatalf("saved models present in tags should not require /api/show, got %d calls", showCalls.Load())
	}
}

func TestLoadSelectableModelsFiltersDeprecatedModels(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[`+
				`{"model":"qwen2.5-coder:32b","description":"old coding model"},`+
				`{"model":"qwen3.5","description":"new local model"}`+
				`]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[`+
				`{"name":"qwen2.5:14b"},`+
				`{"name":"llama3.2:latest"},`+
				`{"name":"custom-local:latest"},`+
				`{"name":"qwen3.5"}`+
				`]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	client, err := newLauncherClient(defaultLaunchPolicy(true, false))
	if err != nil {
		t.Fatalf("newLauncherClient returned error: %v", err)
	}
	items, orderedChecked, err := client.loadSelectableModels(context.Background(), []string{"qwen2.5:14b", "custom-local"}, "qwen2.5:14b", "no models available")
	if err != nil {
		t.Fatalf("loadSelectableModels returned error: %v", err)
	}

	got := names(items)
	for _, deprecated := range []string{"qwen2.5:14b", "llama3.2", "qwen2.5-coder:32b"} {
		if slices.Contains(got, deprecated) {
			t.Fatalf("deprecated model %q should be filtered from selectable models: %v", deprecated, got)
		}
	}
	if !slices.Contains(got, "qwen3.5") {
		t.Fatalf("expected newer recommendation to remain selectable, got %v", got)
	}
	if !slices.Contains(got, "custom-local") {
		t.Fatalf("expected custom local model to remain selectable, got %v", got)
	}
	if slices.Contains(orderedChecked, "qwen2.5:14b") {
		t.Fatalf("deprecated prechecked model should be filtered, got %v", orderedChecked)
	}
	if !slices.Contains(orderedChecked, "custom-local") {
		t.Fatalf("non-deprecated prechecked model should remain, got %v", orderedChecked)
	}
}

func TestBuildLauncherState_ManagedSingleIntegrationShowsSavedModelWhenLiveConfigMissing(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"gemma4"}]}`)
		case "/api/show":
			fmt.Fprint(w, `{"model_info":{"general.context_length":131072}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	if err := config.SaveIntegration("pi", []string{"gemma4"}); err != nil {
		t.Fatalf("failed to save managed integration config: %v", err)
	}

	runner := &launcherManagedRunner{}
	withIntegrationOverride(t, "pi", runner)

	state, err := BuildLauncherState(context.Background())
	if err != nil {
		t.Fatalf("BuildLauncherState returned error: %v", err)
	}

	if state.Integrations["pi"].CurrentModel != "gemma4" {
		t.Fatalf("expected saved model to remain visible, got %q", state.Integrations["pi"].CurrentModel)
	}
	if state.Integrations["pi"].ModelUsable {
		t.Fatal("expected missing live config to mark managed model unusable")
	}
}

func TestLaunchIntegration_ManagedSingleIntegrationConfiguresOnboardsAndRuns(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withInteractiveSession(t, true)
	withLauncherHooks(t)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"gemma4"}]}`)
		case "/api/show":
			fmt.Fprint(w, `{"model_info":{"general.context_length":131072}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	runner := &launcherManagedRunner{
		paths: nil,
	}
	withIntegrationOverride(t, "stubmanaged", runner)

	DefaultSingleSelector = func(title string, items []SelectionItem, current string) (string, error) {
		return "gemma4", nil
	}
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		return true, nil
	}

	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{Name: "stubmanaged"}); err != nil {
		t.Fatalf("LaunchIntegration returned error: %v", err)
	}

	if diff := compareStrings(runner.configured, []string{"gemma4"}); diff != "" {
		t.Fatalf("configured models mismatch: %s", diff)
	}
	if runner.refreshCalls != 1 {
		t.Fatalf("expected runtime refresh once after configure, got %d", runner.refreshCalls)
	}
	if runner.onboardCalls != 1 {
		t.Fatalf("expected onboarding to run once, got %d", runner.onboardCalls)
	}
	if runner.ranModel != "gemma4" {
		t.Fatalf("expected launch to run configured model, got %q", runner.ranModel)
	}

	saved, err := config.LoadIntegration("stubmanaged")
	if err != nil {
		t.Fatalf("failed to reload managed integration config: %v", err)
	}
	if diff := compareStrings(saved.Models, []string{"gemma4"}); diff != "" {
		t.Fatalf("saved models mismatch: %s", diff)
	}
}

func TestLaunchIntegration_ManagedSingleIntegrationReOnboardsWhenSavedFlagIsStale(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withInteractiveSession(t, true)
	withLauncherHooks(t)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"gemma4"}]}`)
		case "/api/show":
			fmt.Fprint(w, `{"model_info":{"general.context_length":131072}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	runner := &launcherManagedRunner{
		currentModel:       "gemma4",
		onboardingComplete: false,
	}
	withIntegrationOverride(t, "stubmanaged", runner)

	if err := config.SaveIntegration("stubmanaged", []string{"gemma4"}); err != nil {
		t.Fatalf("failed to save managed integration config: %v", err)
	}
	if err := config.MarkIntegrationOnboarded("stubmanaged"); err != nil {
		t.Fatalf("failed to mark managed integration onboarded: %v", err)
	}

	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{Name: "stubmanaged"}); err != nil {
		t.Fatalf("LaunchIntegration returned error: %v", err)
	}

	if runner.onboardCalls != 1 {
		t.Fatalf("expected stale onboarded flag to trigger onboarding, got %d calls", runner.onboardCalls)
	}
	if runner.refreshCalls != 0 {
		t.Fatalf("expected no runtime refresh when config is unchanged, got %d", runner.refreshCalls)
	}
	if runner.ranModel != "gemma4" {
		t.Fatalf("expected launch to run saved model after onboarding repair, got %q", runner.ranModel)
	}
}

func TestLaunchIntegration_ManagedSingleIntegrationConfigOnlySkipsFinalRun(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withInteractiveSession(t, true)
	withLauncherHooks(t)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			fmt.Fprint(w, `{"model_info":{"general.context_length":131072}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	runner := &launcherManagedRunner{
		paths: nil,
	}
	withIntegrationOverride(t, "stubmanaged", runner)

	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		return true, nil
	}

	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{
		Name:          "stubmanaged",
		ModelOverride: "gemma4",
		ConfigureOnly: true,
	}); err != nil {
		t.Fatalf("LaunchIntegration returned error: %v", err)
	}

	if runner.ranModel != "" {
		t.Fatalf("expected configure-only flow to skip final launch, got %q", runner.ranModel)
	}
	if runner.refreshCalls != 1 {
		t.Fatalf("expected configure-only flow to refresh runtime once, got %d", runner.refreshCalls)
	}
	if runner.onboardCalls != 1 {
		t.Fatalf("expected configure-only flow to onboard once, got %d", runner.onboardCalls)
	}
}

func TestLaunchIntegration_ManagedSingleIntegrationPrintsConfigurationSuccessAfterConfigure(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withInteractiveSession(t, true)
	withLauncherHooks(t)

	runner := &launcherManagedRunner{
		configSuccessMessage: "configured successfully\nrestore via success message",
		restoreHint:          "run restore command",
		skipModelReadiness:   true,
	}
	withIntegrationOverride(t, "stubmanaged", runner)

	stderr := captureStderr(t, func() {
		if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{
			Name:           "stubmanaged",
			ModelOverride:  "gemma4",
			ForceConfigure: true,
			ConfigureOnly:  true,
		}); err != nil {
			t.Fatalf("LaunchIntegration returned error: %v", err)
		}
	})

	if diff := compareStrings(runner.configured, []string{"gemma4"}); diff != "" {
		t.Fatalf("configured models mismatch: %s", diff)
	}
	if !strings.Contains(stderr, "configured successfully") {
		t.Fatalf("expected configuration success in stderr, got %q", stderr)
	}
	if !strings.Contains(stderr, "restore via success message") {
		t.Fatalf("expected restore guidance in configuration success, got %q", stderr)
	}
	if strings.Contains(stderr, "run restore command") {
		t.Fatalf("restore hint should not print separately after configure, got %q", stderr)
	}
}

func TestLaunchIntegration_QwenConfiguresSingleModel(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withInteractiveSession(t, true)
	withLauncherHooks(t)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"gemma4"}]}`)
		case "/api/show":
			fmt.Fprint(w, `{"model_info":{"general.context_length":131072}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	binDir := filepath.Join(tmpDir, "bin")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		t.Fatalf("failed to create bin dir: %v", err)
	}
	writeFakeBinary(t, binDir, "qwen")
	t.Setenv("PATH", binDir)

	DefaultSingleSelector = func(title string, items []SelectionItem, current string) (string, error) {
		return "gemma4", nil
	}
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		return true, nil
	}

	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{
		Name:          "qwen",
		ConfigureOnly: true,
	}); err != nil {
		t.Fatalf("LaunchIntegration returned error: %v", err)
	}

	data, err := os.ReadFile(filepath.Join(tmpDir, ".qwen", "settings.json"))
	if err != nil {
		t.Fatalf("failed to read qwen config: %v", err)
	}

	var cfg map[string]any
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("failed to parse qwen config: %v", err)
	}

	modelCfg := cfg["model"].(map[string]any)
	if modelCfg["name"] != "gemma4" {
		t.Fatalf("expected model.name gemma4, got %v", modelCfg["name"])
	}

	modelProviders := cfg["modelProviders"].(map[string]any)
	openai := modelProviders["openai"].([]any)
	if len(openai) != 1 {
		t.Fatalf("expected one provider, got %d", len(openai))
	}

	saved, err := config.LoadIntegration("qwen")
	if err != nil {
		t.Fatalf("failed to reload qwen integration config: %v", err)
	}
	if diff := compareStrings(saved.Models, []string{"gemma4"}); diff != "" {
		t.Fatalf("saved models mismatch: %s", diff)
	}
	if !saved.Onboarded {
		t.Fatal("expected qwen integration to be marked onboarded")
	}
}

func TestLaunchIntegration_ManagedSingleIntegrationDoesNotPrintRestoreHintWhenUnchanged(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withInteractiveSession(t, true)
	withLauncherHooks(t)

	runner := &launcherManagedRunner{
		currentModel:         "gemma4",
		onboardingComplete:   true,
		configSuccessMessage: "configured successfully",
		restoreHint:          "run restore command",
		skipModelReadiness:   true,
	}
	withIntegrationOverride(t, "stubmanaged", runner)

	if err := config.SaveIntegration("stubmanaged", []string{"gemma4"}); err != nil {
		t.Fatalf("failed to save managed integration config: %v", err)
	}
	if err := config.MarkIntegrationOnboarded("stubmanaged"); err != nil {
		t.Fatalf("failed to mark integration onboarded: %v", err)
	}

	stderr := captureStderr(t, func() {
		if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{Name: "stubmanaged"}); err != nil {
			t.Fatalf("LaunchIntegration returned error: %v", err)
		}
	})

	if len(runner.configured) != 0 {
		t.Fatalf("expected Configure to be skipped when saved matches, got %v", runner.configured)
	}
	if strings.Contains(stderr, "configured successfully") {
		t.Fatalf("configuration success should not print when config is unchanged, got %q", stderr)
	}
	if strings.Contains(stderr, "run restore command") {
		t.Fatalf("restore hint should not print when config is unchanged, got %q", stderr)
	}
}

func TestLaunchIntegration_ManagedSingleIntegrationForceConfigureUsesModelOverride(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withInteractiveSession(t, true)
	withLauncherHooks(t)

	runner := &launcherManagedRunner{
		paths:              nil,
		skipModelReadiness: true,
	}
	withIntegrationOverride(t, "stubmanaged", runner)

	DefaultSingleSelector = func(title string, items []SelectionItem, current string) (string, error) {
		return "", fmt.Errorf("selector should not run with an explicit model override")
	}

	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{
		Name:           "stubmanaged",
		ModelOverride:  "gemma4",
		ForceConfigure: true,
		ConfigureOnly:  true,
	}); err != nil {
		t.Fatalf("LaunchIntegration returned error: %v", err)
	}

	if diff := compareStrings(runner.configured, []string{"gemma4"}); diff != "" {
		t.Fatalf("configured models mismatch: %s", diff)
	}
	if runner.ranModel != "" {
		t.Fatalf("expected configure-only flow to skip final launch, got %q", runner.ranModel)
	}
}

func TestLaunchIntegration_ManagedSingleIntegrationSkipsRewriteWhenSavedMatches(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withInteractiveSession(t, true)
	withLauncherHooks(t)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"gemma4"}]}`)
		case "/api/show":
			fmt.Fprint(w, `{"model_info":{"general.context_length":131072}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	if err := config.SaveIntegration("stubmanaged", []string{"gemma4"}); err != nil {
		t.Fatalf("failed to save managed integration config: %v", err)
	}

	runner := &launcherManagedRunner{
		currentModel: "gemma4",
	}
	withIntegrationOverride(t, "stubmanaged", runner)

	DefaultSingleSelector = func(title string, items []SelectionItem, current string) (string, error) {
		t.Fatal("selector should not be called when saved model matches target")
		return "", nil
	}
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		t.Fatal("confirm prompt should not run when saved model matches target")
		return false, nil
	}

	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{Name: "stubmanaged"}); err != nil {
		t.Fatalf("LaunchIntegration returned error: %v", err)
	}

	if len(runner.configured) != 0 {
		t.Fatalf("expected Configure to be skipped when saved matches, got %v", runner.configured)
	}
	if runner.refreshCalls != 0 {
		t.Fatalf("expected no runtime refresh when config is unchanged, got %d", runner.refreshCalls)
	}
	if runner.ranModel != "gemma4" {
		t.Fatalf("expected launch to run saved model, got %q", runner.ranModel)
	}
}

func TestLaunchIntegration_ManagedSingleIntegrationRewritesWhenSavedMatchesButLiveConfigMissing(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withInteractiveSession(t, true)
	withLauncherHooks(t)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			fmt.Fprint(w, `{"model_info":{"general.context_length":131072}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	if err := config.SaveIntegration("stubmanaged", []string{"gemma4"}); err != nil {
		t.Fatalf("failed to save managed integration config: %v", err)
	}

	runner := &launcherManagedRunner{}
	withIntegrationOverride(t, "stubmanaged", runner)

	DefaultSingleSelector = func(title string, items []SelectionItem, current string) (string, error) {
		t.Fatal("selector should not be called when saved model is usable")
		return "", nil
	}
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		return true, nil
	}

	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{Name: "stubmanaged"}); err != nil {
		t.Fatalf("LaunchIntegration returned error: %v", err)
	}

	if diff := compareStrings(runner.configured, []string{"gemma4"}); diff != "" {
		t.Fatalf("expected Configure to rewrite missing live config: %s", diff)
	}
	if runner.refreshCalls != 1 {
		t.Fatalf("expected runtime refresh once after rewrite, got %d", runner.refreshCalls)
	}
	if runner.ranModel != "gemma4" {
		t.Fatalf("expected launch to run saved model, got %q", runner.ranModel)
	}
}

func TestLaunchIntegration_ManagedSingleIntegrationRewritesWhenSavedDiffers(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withInteractiveSession(t, true)
	withLauncherHooks(t)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"gemma4"}]}`)
		case "/api/show":
			fmt.Fprint(w, `{"model_info":{"general.context_length":131072}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	if err := config.SaveIntegration("stubmanaged", []string{"old-model"}); err != nil {
		t.Fatalf("failed to save managed integration config: %v", err)
	}

	runner := &launcherManagedRunner{}
	withIntegrationOverride(t, "stubmanaged", runner)

	DefaultSingleSelector = func(title string, items []SelectionItem, current string) (string, error) {
		t.Fatal("selector should not be called when model override is provided")
		return "", nil
	}
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		return true, nil
	}

	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{
		Name:          "stubmanaged",
		ModelOverride: "gemma4",
	}); err != nil {
		t.Fatalf("LaunchIntegration returned error: %v", err)
	}

	if diff := compareStrings(runner.configured, []string{"gemma4"}); diff != "" {
		t.Fatalf("expected Configure to run when saved differs from target: %s", diff)
	}
	if runner.refreshCalls != 1 {
		t.Fatalf("expected runtime refresh once after configure, got %d", runner.refreshCalls)
	}
	if runner.ranModel != "gemma4" {
		t.Fatalf("expected launch to run configured model, got %q", runner.ranModel)
	}
}

func TestLaunchIntegration_ManagedSingleIntegrationRewritesWhenLiveConfigDrifts(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withInteractiveSession(t, true)
	withLauncherHooks(t)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"gemma4"},{"name":"qwen3:8b"}]}`)
		case "/api/show":
			fmt.Fprint(w, `{"model_info":{"general.context_length":131072}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	if err := config.SaveIntegration("stubmanaged", []string{"gemma4"}); err != nil {
		t.Fatalf("failed to save managed integration config: %v", err)
	}

	runner := &launcherManagedRunner{
		currentModel: "qwen3:8b",
	}
	withIntegrationOverride(t, "stubmanaged", runner)

	DefaultSingleSelector = func(title string, items []SelectionItem, current string) (string, error) {
		t.Fatal("selector should not be called when live config already provides the target")
		return "", nil
	}
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		return true, nil
	}

	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{Name: "stubmanaged"}); err != nil {
		t.Fatalf("LaunchIntegration returned error: %v", err)
	}

	if diff := compareStrings(runner.configured, []string{"qwen3:8b"}); diff != "" {
		t.Fatalf("expected Configure to reconcile stale saved config to live target: %s", diff)
	}
	if runner.refreshCalls != 1 {
		t.Fatalf("expected runtime refresh once after drift reconciliation, got %d", runner.refreshCalls)
	}
	if runner.ranModel != "qwen3:8b" {
		t.Fatalf("expected launch to run live configured model, got %q", runner.ranModel)
	}

	saved, err := config.LoadIntegration("stubmanaged")
	if err != nil {
		t.Fatalf("failed to reload managed integration config: %v", err)
	}
	if diff := compareStrings(saved.Models, []string{"qwen3:8b"}); diff != "" {
		t.Fatalf("saved models mismatch after drift reconciliation: %s", diff)
	}
}

func TestLaunchIntegration_ManagedSingleIntegrationStopsWhenRuntimeRefreshFails(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withInteractiveSession(t, true)
	withLauncherHooks(t)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			fmt.Fprint(w, `{"model_info":{"general.context_length":131072}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	runner := &launcherManagedRunner{
		refreshErr: fmt.Errorf("boom"),
	}
	withIntegrationOverride(t, "stubmanaged", runner)

	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		return true, nil
	}

	err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{
		Name:          "stubmanaged",
		ModelOverride: "gemma4",
	})
	if err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("expected runtime refresh error, got %v", err)
	}
	if runner.ranModel != "" {
		t.Fatalf("expected final launch to stop on runtime refresh failure, got %q", runner.ranModel)
	}
	if runner.refreshCalls != 1 {
		t.Fatalf("expected one runtime refresh attempt, got %d", runner.refreshCalls)
	}
	if runner.onboardCalls != 0 {
		t.Fatalf("expected onboarding to stop after refresh failure, got %d", runner.onboardCalls)
	}
}

func TestLaunchIntegration_ManagedSingleIntegrationCanConfigureWithModelList(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withInteractiveSession(t, true)
	withLauncherHooks(t)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"gemma4"},{"name":"qwen3:8b"}]}`)
		case "/api/show":
			fmt.Fprint(w, `{"model_info":{"general.context_length":131072}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	runner := &launcherManagedListRunner{}
	withIntegrationOverride(t, "stubmanaged", runner)

	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		return true, nil
	}

	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{
		Name:          "stubmanaged",
		ModelOverride: "gemma4",
	}); err != nil {
		t.Fatalf("LaunchIntegration returned error: %v", err)
	}

	if diff := compareStringSlices(runner.configuredModelLists, [][]string{{"gemma4", "kimi-k2.6:cloud", "qwen3.5:cloud", "glm-5.1:cloud", "minimax-m2.7:cloud", "qwen3.5", "qwen3:8b"}}); diff != "" {
		t.Fatalf("configured model list mismatch (-want +got):\n%s", diff)
	}
	if diff := compareStrings(runner.configured, []string{"gemma4"}); diff != "" {
		t.Fatalf("configured primary mismatch: %s", diff)
	}
}

func TestLaunchIntegration_ManagedAutodiscoverySkipsModelPicker(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withInteractiveSession(t, true)
	withLauncherHooks(t)

	runner := &launcherManagedAutodiscoveryRunner{}
	withIntegrationOverride(t, "stubmanaged", runner)

	DefaultSingleSelector = func(title string, items []SelectionItem, current string) (string, error) {
		t.Fatal("model selector should not run for autodiscovery integrations")
		return "", nil
	}
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		return true, nil
	}

	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{Name: "stubmanaged"}); err != nil {
		t.Fatalf("LaunchIntegration returned error: %v", err)
	}

	if runner.autodiscoveryConfigures != 1 {
		t.Fatalf("expected one autodiscovery configure, got %d", runner.autodiscoveryConfigures)
	}
	if runner.ranModel != "Ollama Cloud" {
		t.Fatalf("expected launch to run autodiscovery label, got %q", runner.ranModel)
	}
	saved, err := config.LoadIntegration("stubmanaged")
	if err != nil {
		t.Fatalf("failed to reload managed integration config: %v", err)
	}
	if diff := compareStrings(saved.Models, []string{"Ollama Cloud"}); diff != "" {
		t.Fatalf("saved models mismatch: %s", diff)
	}
}

func TestLaunchIntegration_ManagedAutodiscoveryPrintsConfigurationSuccessAfterConfigure(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withInteractiveSession(t, true)
	withLauncherHooks(t)

	runner := &launcherManagedAutodiscoveryRunner{
		restoreHint:          "run restore command",
		configSuccessMessage: "configured successfully\nrestore via success message",
	}
	withIntegrationOverride(t, "stubmanaged", runner)

	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		return true, nil
	}

	stderr := captureStderr(t, func() {
		if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{Name: "stubmanaged"}); err != nil {
			t.Fatalf("LaunchIntegration returned error: %v", err)
		}
	})

	if runner.autodiscoveryConfigures != 1 {
		t.Fatalf("expected one autodiscovery configure, got %d", runner.autodiscoveryConfigures)
	}
	if !strings.Contains(stderr, "configured successfully") {
		t.Fatalf("expected configuration success in stderr, got %q", stderr)
	}
	if !strings.Contains(stderr, "restore via success message") {
		t.Fatalf("expected restore guidance in configuration success, got %q", stderr)
	}
	if strings.Contains(stderr, "run restore command") {
		t.Fatalf("restore hint should not print separately after configure, got %q", stderr)
	}
}

func TestLaunchIntegration_ManagedAutodiscoveryPrintsRestoreHintWhenAlreadyConfigured(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withInteractiveSession(t, true)
	withLauncherHooks(t)

	runner := &launcherManagedAutodiscoveryRunner{
		autodiscoveryConfigured: true,
		restoreHint:             "run restore command",
	}
	withIntegrationOverride(t, "stubmanaged", runner)

	if err := config.SaveIntegration("stubmanaged", []string{"Ollama Cloud"}); err != nil {
		t.Fatalf("failed to save managed integration config: %v", err)
	}
	if err := config.MarkIntegrationOnboarded("stubmanaged"); err != nil {
		t.Fatalf("failed to mark integration onboarded: %v", err)
	}

	stderr := captureStderr(t, func() {
		if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{Name: "stubmanaged"}); err != nil {
			t.Fatalf("LaunchIntegration returned error: %v", err)
		}
	})

	if runner.autodiscoveryConfigures != 0 {
		t.Fatalf("expected configured autodiscovery integration not to reconfigure, got %d configures", runner.autodiscoveryConfigures)
	}
	if runner.ranModel != "Ollama Cloud" {
		t.Fatalf("expected launch to run autodiscovery label, got %q", runner.ranModel)
	}
	if !strings.Contains(stderr, "run restore command") {
		t.Fatalf("expected restore hint in stderr, got %q", stderr)
	}
}

func TestLaunchIntegration_ManagedAutodiscoveryPrintsConfigurationSuccessWhenAlreadyConfigured(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withInteractiveSession(t, true)
	withLauncherHooks(t)

	runner := &launcherManagedAutodiscoveryRunner{
		autodiscoveryConfigured: true,
		restoreHint:             "run restore command",
		configSuccessMessage:    "configured successfully\nrestore via success message",
	}
	withIntegrationOverride(t, "stubmanaged", runner)

	if err := config.SaveIntegration("stubmanaged", []string{"Ollama Cloud"}); err != nil {
		t.Fatalf("failed to save managed integration config: %v", err)
	}
	if err := config.MarkIntegrationOnboarded("stubmanaged"); err != nil {
		t.Fatalf("failed to mark integration onboarded: %v", err)
	}

	stderr := captureStderr(t, func() {
		if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{Name: "stubmanaged"}); err != nil {
			t.Fatalf("LaunchIntegration returned error: %v", err)
		}
	})

	if runner.autodiscoveryConfigures != 0 {
		t.Fatalf("expected configured autodiscovery integration not to reconfigure, got %d configures", runner.autodiscoveryConfigures)
	}
	if !strings.Contains(stderr, "configured successfully") {
		t.Fatalf("expected configuration success in stderr, got %q", stderr)
	}
	if !strings.Contains(stderr, "restore via success message") {
		t.Fatalf("expected restore guidance in configuration success, got %q", stderr)
	}
	if strings.Contains(stderr, "run restore command") {
		t.Fatalf("restore hint should not print separately when success message exists, got %q", stderr)
	}
}

func TestLaunchIntegration_RestorePrintsSuccessMessage(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	runner := &launcherRestorableRunner{
		restoreSuccessMessage: "restored successfully",
	}
	withIntegrationOverride(t, "stubrestore", runner)

	stderr := captureStderr(t, func() {
		if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{
			Name:    "stubrestore",
			Restore: true,
		}); err != nil {
			t.Fatalf("LaunchIntegration returned error: %v", err)
		}
	})

	if !runner.restored {
		t.Fatal("expected restore to run")
	}
	if !strings.Contains(stderr, "restored successfully") {
		t.Fatalf("expected restore success in stderr, got %q", stderr)
	}
}

func TestLaunchIntegration_ManagedAutodiscoveryForceConfigureRerunsSetup(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withInteractiveSession(t, true)
	withLauncherHooks(t)

	runner := &launcherManagedAutodiscoveryRunner{
		autodiscoveryConfigured: true,
	}
	withIntegrationOverride(t, "stubmanaged", runner)

	if err := config.SaveIntegration("stubmanaged", []string{"Ollama Cloud"}); err != nil {
		t.Fatalf("failed to save managed integration config: %v", err)
	}
	if err := config.MarkIntegrationOnboarded("stubmanaged"); err != nil {
		t.Fatalf("failed to mark integration onboarded: %v", err)
	}

	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{
		Name:           "stubmanaged",
		ForceConfigure: true,
	}); err != nil {
		t.Fatalf("LaunchIntegration returned error: %v", err)
	}

	if runner.autodiscoveryConfigures != 1 {
		t.Fatalf("expected forced autodiscovery configure to rerun setup, got %d configures", runner.autodiscoveryConfigures)
	}
	if runner.ranModel != "Ollama Cloud" {
		t.Fatalf("expected launch to run autodiscovery label, got %q", runner.ranModel)
	}
}

func TestLaunchIntegration_CloudAutodiscoveryUsesSignInHook(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withInteractiveSession(t, true)
	withLauncherHooks(t)

	runner := &launcherManagedAutodiscoveryRunner{usesCloud: true}
	withIntegrationOverride(t, "stubmanaged", runner)

	signInCalled := false
	DefaultSignIn = func(modelName, signInURL string) (string, error) {
		signInCalled = true
		if modelName != "Ollama Cloud" {
			t.Fatalf("sign-in model = %q, want Ollama Cloud", modelName)
		}
		if signInURL != "https://example.com/signin" {
			t.Fatalf("sign-in URL = %q, want test URL", signInURL)
		}
		return "test-user", nil
	}
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		return true, nil
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

	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{Name: "stubmanaged"}); err != nil {
		t.Fatalf("LaunchIntegration returned error: %v", err)
	}

	if !signInCalled {
		t.Fatal("expected cloud autodiscovery launch to use the sign-in hook")
	}
	if runner.autodiscoveryConfigures != 1 {
		t.Fatalf("expected one autodiscovery configure, got %d", runner.autodiscoveryConfigures)
	}
	if runner.ranModel != "Ollama Cloud" {
		t.Fatalf("expected launch to run autodiscovery label, got %q", runner.ranModel)
	}
}

func TestBuildLauncherIntegrationState_CloudAutodiscoveryDoesNotCheckSignIn(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	runner := &launcherManagedAutodiscoveryRunner{
		autodiscoveryConfigured: true,
		usesCloud:               true,
	}
	withIntegrationOverride(t, "stubmanaged", runner)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/status":
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprint(w, `{"error":"not found"}`)
		case "/api/me":
			t.Fatal("build launcher state should not check whoami")
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	launchClient, err := newLauncherClient(defaultLaunchPolicy(true, false))
	if err != nil {
		t.Fatal(err)
	}
	state, err := launchClient.buildLauncherIntegrationState(context.Background(), IntegrationInfo{
		Name:        "stubmanaged",
		DisplayName: "Stub Managed",
	})
	if err != nil {
		t.Fatalf("buildLauncherIntegrationState returned error: %v", err)
	}

	if state.CurrentModel != "Ollama Cloud" {
		t.Fatalf("current model = %q, want Ollama Cloud", state.CurrentModel)
	}
	if !state.ModelUsable {
		t.Fatal("expected cloud autodiscovery config to stay usable until launch-time auth check")
	}
}

func TestLaunchIntegration_ManagedAutodiscoveryRejectsModelOverride(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withInteractiveSession(t, true)
	withLauncherHooks(t)

	runner := &launcherManagedAutodiscoveryRunner{}
	withIntegrationOverride(t, "stubmanaged", runner)

	err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{
		Name:          "stubmanaged",
		ModelOverride: "qwen3.5:cloud",
	})
	if err == nil || !strings.Contains(err.Error(), "discovers models automatically") {
		t.Fatalf("LaunchIntegration error = %v, want automatic discovery guidance", err)
	}
	if runner.autodiscoveryConfigures != 0 {
		t.Fatalf("expected no configure after model override rejection, got %d", runner.autodiscoveryConfigures)
	}
}

func TestLaunchIntegration_ManagedSingleIntegrationHeadlessNeedsInteractiveOnboarding(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withInteractiveSession(t, false)
	withLauncherHooks(t)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			fmt.Fprint(w, `{"model_info":{"general.context_length":131072}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	runner := &launcherManagedRunner{
		paths: nil,
	}
	withIntegrationOverride(t, "stubmanaged", runner)

	err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{
		Name:          "stubmanaged",
		ModelOverride: "gemma4",
		Policy:        &LaunchPolicy{Confirm: LaunchConfirmAutoApprove, MissingModel: LaunchMissingModelAutoPull},
	})
	if err == nil {
		t.Fatal("expected headless onboarding requirement to fail")
	}
	if !strings.Contains(err.Error(), "interactive gateway setup") {
		t.Fatalf("expected interactive onboarding guidance, got %v", err)
	}
	if runner.ranModel != "" {
		t.Fatalf("expected no final launch when onboarding is still required, got %q", runner.ranModel)
	}
	if runner.onboardCalls != 0 {
		t.Fatalf("expected no onboarding attempts in headless mode, got %d", runner.onboardCalls)
	}
}

func TestLaunchIntegration_ManagedSingleIntegrationHeadlessAllowsNonInteractiveOnboarding(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withInteractiveSession(t, false)
	withLauncherHooks(t)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			fmt.Fprint(w, `{"model_info":{"general.context_length":131072}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	runner := &launcherHeadlessManagedRunner{}
	withIntegrationOverride(t, "stubmanaged", runner)

	err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{
		Name:          "stubmanaged",
		ModelOverride: "gemma4",
		Policy:        &LaunchPolicy{Confirm: LaunchConfirmAutoApprove, MissingModel: LaunchMissingModelAutoPull},
	})
	if err != nil {
		t.Fatalf("expected non-interactive onboarding to succeed headlessly, got %v", err)
	}
	if diff := compareStrings(runner.configured, []string{"gemma4"}); diff != "" {
		t.Fatalf("configured models mismatch: %s", diff)
	}
	if runner.onboardCalls != 1 {
		t.Fatalf("expected onboarding to run once, got %d", runner.onboardCalls)
	}
	if runner.ranModel != "gemma4" {
		t.Fatalf("expected launch to run configured model, got %q", runner.ranModel)
	}
}

func TestBuildLauncherState_InstalledAndCloudDisabled(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)

	binDir := t.TempDir()
	writeFakeBinary(t, binDir, "opencode")
	t.Setenv("PATH", binDir)

	if err := config.SetLastModel("glm-5:cloud"); err != nil {
		t.Fatalf("failed to save last model: %v", err)
	}
	if err := config.SaveIntegration("claude", []string{"glm-5:cloud"}); err != nil {
		t.Fatalf("failed to save claude config: %v", err)
	}
	if err := config.SaveIntegration("opencode", []string{"glm-5:cloud", "sample-model"}); err != nil {
		t.Fatalf("failed to save opencode config: %v", err)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"sample-model"}]}`)
		case "/api/status":
			fmt.Fprint(w, `{"cloud":{"disabled":true,"source":"config"}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	state, err := BuildLauncherState(context.Background())
	if err != nil {
		t.Fatalf("BuildLauncherState returned error: %v", err)
	}

	if !state.Integrations["opencode"].Installed {
		t.Fatal("expected opencode to be marked installed")
	}
	if state.Integrations["claude"].Installed {
		t.Fatal("expected claude to be marked not installed")
	}
	if state.RunModelUsable {
		t.Fatal("expected saved cloud run model to be unusable when cloud is disabled")
	}
	if state.Integrations["claude"].ModelUsable {
		t.Fatal("expected claude cloud config to be unusable when cloud is disabled")
	}
	if !state.Integrations["opencode"].ModelUsable {
		t.Fatal("expected editor config with a remaining local model to stay usable")
	}
	if state.Integrations["opencode"].CurrentModel != "sample-model" {
		t.Fatalf("expected editor current model to fall back to remaining local model, got %q", state.Integrations["opencode"].CurrentModel)
	}
}

func TestBuildLauncherState_MigratesLegacyOpenclawAliasConfig(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)

	if err := config.SaveIntegration("clawdbot", []string{"sample-model"}); err != nil {
		t.Fatalf("failed to seed legacy alias config: %v", err)
	}
	if err := config.SaveAliases("clawdbot", map[string]string{"primary": "sample-model"}); err != nil {
		t.Fatalf("failed to seed legacy alias map: %v", err)
	}
	if err := config.MarkIntegrationOnboarded("clawdbot"); err != nil {
		t.Fatalf("failed to seed legacy onboarding state: %v", err)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"sample-model"}]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	state, err := BuildLauncherState(context.Background())
	if err != nil {
		t.Fatalf("BuildLauncherState returned error: %v", err)
	}
	if state.Integrations["openclaw"].CurrentModel != "sample-model" {
		t.Fatalf("expected openclaw state to reuse legacy alias config, got %q", state.Integrations["openclaw"].CurrentModel)
	}

	migrated, err := config.LoadIntegration("openclaw")
	if err != nil {
		t.Fatalf("expected canonical config to be migrated, got %v", err)
	}
	if !slices.Equal(migrated.Models, []string{"sample-model"}) {
		t.Fatalf("unexpected migrated models: %v", migrated.Models)
	}
	if migrated.Aliases["primary"] != "sample-model" {
		t.Fatalf("expected aliases to migrate, got %v", migrated.Aliases)
	}
	if !migrated.Onboarded {
		t.Fatal("expected onboarding state to migrate to canonical openclaw key")
	}
}

func TestBuildLauncherState_ToleratesInventoryFailure(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)

	if err := config.SetLastModel("sample-model"); err != nil {
		t.Fatalf("failed to seed last model: %v", err)
	}
	if err := config.SaveIntegration("claude", []string{"qwen3:8b"}); err != nil {
		t.Fatalf("failed to seed claude config: %v", err)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			w.WriteHeader(http.StatusInternalServerError)
			fmt.Fprint(w, `{"error":"temporary failure"}`)
		case "/api/show":
			var req apiShowRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			fmt.Fprintf(w, `{"model":%q}`, req.Model)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	state, err := BuildLauncherState(context.Background())
	if err != nil {
		t.Fatalf("BuildLauncherState should tolerate inventory failure, got %v", err)
	}
	if !state.RunModelUsable {
		t.Fatal("expected saved run model to remain usable via show fallback")
	}
	if state.Integrations["claude"].CurrentModel != "qwen3:8b" {
		t.Fatalf("expected saved integration model to remain visible, got %q", state.Integrations["claude"].CurrentModel)
	}
	if !state.Integrations["claude"].ModelUsable {
		t.Fatal("expected saved integration model to remain usable via show fallback")
	}
}

func TestBuildLauncherState_UsesTagsInventoryWithoutShow(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)

	if err := config.SetLastModel("sample-model"); err != nil {
		t.Fatalf("failed to seed last model: %v", err)
	}
	if err := config.SaveIntegration("codex", []string{"qwen3:8b"}); err != nil {
		t.Fatalf("failed to seed codex config: %v", err)
	}

	var showCalls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			fmt.Fprint(w, `{"models":[`+
				`{"name":"sample-model","capabilities":["completion","tools"],"context_length":131072,"size":3200000000},`+
				`{"name":"qwen3:8b","capabilities":["completion","tools"],"context_length":65536,"size":4500000000}`+
				`]}`)
		case "/api/show":
			showCalls.Add(1)
			fmt.Fprint(w, `{"model_info":{"general.context_length":131072}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	state, err := BuildLauncherState(context.Background())
	if err != nil {
		t.Fatalf("BuildLauncherState returned error: %v", err)
	}
	if !state.RunModelUsable {
		t.Fatal("expected saved run model to be usable from tags inventory")
	}
	if state.Integrations["codex"].CurrentModel != "qwen3:8b" {
		t.Fatalf("expected codex current model from saved config, got %q", state.Integrations["codex"].CurrentModel)
	}
	if !state.Integrations["codex"].ModelUsable {
		t.Fatal("expected saved codex model to be usable from tags inventory")
	}
	if got := showCalls.Load(); got != 0 {
		t.Fatalf("show calls = %d, want 0 for broad launcher state", got)
	}
}

func TestResolveRunModel_UsesSavedModelWithoutSelector(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	if err := config.SetLastModel("sample-model"); err != nil {
		t.Fatalf("failed to save last model: %v", err)
	}

	selectorCalled := false
	DefaultSingleSelector = func(title string, items []SelectionItem, current string) (string, error) {
		selectorCalled = true
		return "", nil
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"sample-model"}]}`)
		case "/api/show":
			fmt.Fprint(w, `{"model":"sample-model"}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	model, err := ResolveRunModel(context.Background(), RunModelRequest{})
	if err != nil {
		t.Fatalf("ResolveRunModel returned error: %v", err)
	}
	if model != "sample-model" {
		t.Fatalf("expected saved model, got %q", model)
	}
	if selectorCalled {
		t.Fatal("selector should not be called when saved model is usable")
	}
}

func TestResolveRunModel_HeadlessYesAutoPicksLastModel(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)
	withInteractiveSession(t, false)

	if err := config.SetLastModel("missing-model"); err != nil {
		t.Fatalf("failed to save last model: %v", err)
	}

	DefaultSingleSelector = func(title string, items []SelectionItem, current string) (string, error) {
		t.Fatal("selector should not be called in headless --yes mode")
		return "", nil
	}

	restoreConfirm := withLaunchConfirmPolicy(launchConfirmPolicy{yes: true})
	defer restoreConfirm()

	pullCalled := false
	modelPulled := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"sample-model"}]}`)
		case "/api/show":
			var req apiShowRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			if req.Model == "missing-model" && !modelPulled {
				w.WriteHeader(http.StatusNotFound)
				fmt.Fprint(w, `{"error":"model not found"}`)
				return
			}
			fmt.Fprintf(w, `{"model":%q}`, req.Model)
		case "/api/pull":
			pullCalled = true
			modelPulled = true
			w.WriteHeader(http.StatusOK)
			fmt.Fprint(w, `{"status":"success"}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	var model string
	stderr := captureStderr(t, func() {
		var err error
		model, err = ResolveRunModel(context.Background(), RunModelRequest{})
		if err != nil {
			t.Fatalf("ResolveRunModel returned error: %v", err)
		}
	})

	if model != "missing-model" {
		t.Fatalf("expected saved last model to be selected, got %q", model)
	}
	if !pullCalled {
		t.Fatal("expected missing saved model to be auto-pulled in headless --yes mode")
	}
	if !strings.Contains(stderr, `Headless mode: auto-selected last used model "missing-model"`) {
		t.Fatalf("expected headless auto-pick message in stderr, got %q", stderr)
	}
}

func TestResolveRunModel_UsesRequestPolicy(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)
	withInteractiveSession(t, false)

	if err := config.SetLastModel("missing-model"); err != nil {
		t.Fatalf("failed to save last model: %v", err)
	}

	DefaultSingleSelector = func(title string, items []SelectionItem, current string) (string, error) {
		t.Fatal("selector should not be called when request policy enables headless auto-pick")
		return "", nil
	}

	pullCalled := false
	modelPulled := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"sample-model"}]}`)
		case "/api/show":
			var req apiShowRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			if req.Model == "missing-model" && !modelPulled {
				w.WriteHeader(http.StatusNotFound)
				fmt.Fprint(w, `{"error":"model not found"}`)
				return
			}
			fmt.Fprintf(w, `{"model":%q}`, req.Model)
		case "/api/pull":
			pullCalled = true
			modelPulled = true
			w.WriteHeader(http.StatusOK)
			fmt.Fprint(w, `{"status":"success"}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	reqPolicy := LaunchPolicy{
		Confirm:      LaunchConfirmAutoApprove,
		MissingModel: LaunchMissingModelAutoPull,
	}
	model, err := ResolveRunModel(context.Background(), RunModelRequest{Policy: &reqPolicy})
	if err != nil {
		t.Fatalf("ResolveRunModel returned error: %v", err)
	}
	if model != "missing-model" {
		t.Fatalf("expected saved last model to be selected, got %q", model)
	}
	if !pullCalled {
		t.Fatal("expected missing saved model to be auto-pulled when request policy enables auto-pull")
	}
}

func TestResolveRunModel_ForcePickerAlwaysUsesSelector(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	if err := config.SetLastModel("sample-model"); err != nil {
		t.Fatalf("failed to save last model: %v", err)
	}

	var selectorCalls int
	DefaultSingleSelector = func(title string, items []SelectionItem, current string) (string, error) {
		selectorCalls++
		if current != "sample-model" {
			t.Fatalf("expected current selection to be last model, got %q", current)
		}
		return "qwen3:8b", nil
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"sample-model"},{"name":"qwen3:8b"}]}`)
		case "/api/show":
			fmt.Fprint(w, `{"model":"qwen3:8b"}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	model, err := ResolveRunModel(context.Background(), RunModelRequest{ForcePicker: true})
	if err != nil {
		t.Fatalf("ResolveRunModel returned error: %v", err)
	}
	if selectorCalls != 1 {
		t.Fatalf("expected selector to be called once, got %d", selectorCalls)
	}
	if model != "qwen3:8b" {
		t.Fatalf("expected forced selection to win, got %q", model)
	}
	if got := config.LastModel(); got != "qwen3:8b" {
		t.Fatalf("expected last model to be updated, got %q", got)
	}
}

func TestResolveRunModel_ForcePicker_DoesNotReorderByLastModel(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	if err := config.SetLastModel("qwen3.5"); err != nil {
		t.Fatalf("failed to save last model: %v", err)
	}

	var gotNames []string
	DefaultSingleSelector = func(title string, items []SelectionItem, current string) (string, error) {
		if current != "qwen3.5" {
			t.Fatalf("expected current selection to be last model, got %q", current)
		}

		gotNames = make([]string, 0, len(items))
		for _, item := range items {
			gotNames = append(gotNames, item.Name)
		}
		return "qwen3.5", nil
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"qwen3.5"},{"name":"gemma4"}]}`)
		case "/api/show":
			fmt.Fprint(w, `{"model":"qwen3.5"}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	_, err := ResolveRunModel(context.Background(), RunModelRequest{ForcePicker: true})
	if err != nil {
		t.Fatalf("ResolveRunModel returned error: %v", err)
	}

	if len(gotNames) == 0 {
		t.Fatal("expected selector to receive model items")
	}

	glmIdx := slices.Index(gotNames, "gemma4")
	qwenIdx := slices.Index(gotNames, "qwen3.5")
	if glmIdx == -1 || qwenIdx == -1 {
		t.Fatalf("expected recommended local models in selector items, got %v", gotNames)
	}
	if qwenIdx < glmIdx {
		t.Fatalf("expected list order to stay stable and not float last model to top, got %v", gotNames)
	}
}

func TestResolveRunModel_UsesSignInHookForCloudModel(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	DefaultSingleSelector = func(title string, items []SelectionItem, current string) (string, error) {
		return "glm-5:cloud", nil
	}

	signInCalled := false
	DefaultSignIn = func(modelName, signInURL string) (string, error) {
		signInCalled = true
		if modelName != "glm-5:cloud" {
			t.Fatalf("unexpected model passed to sign-in: %q", modelName)
		}
		return "test-user", nil
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[]}`)
		case "/api/status":
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprint(w, `{"error":"not found"}`)
		case "/api/show":
			fmt.Fprint(w, `{"remote_model":"glm-5"}`)
		case "/api/me":
			w.WriteHeader(http.StatusUnauthorized)
			fmt.Fprint(w, `{"error":"unauthorized","signin_url":"https://example.com/signin"}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	model, err := ResolveRunModel(context.Background(), RunModelRequest{ForcePicker: true})
	if err != nil {
		t.Fatalf("ResolveRunModel returned error: %v", err)
	}
	if model != "glm-5:cloud" {
		t.Fatalf("expected selected cloud model, got %q", model)
	}
	if !signInCalled {
		t.Fatal("expected sign-in hook to be used for cloud model")
	}
}

func TestResolveRunModel_MetadataSignedOutUsesSignInHook(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	DefaultSingleSelector = func(title string, items []SelectionItem, current string) (string, error) {
		return "qwen3.5:cloud", nil
	}

	signedIn := false
	signInCalled := false
	DefaultSignIn = func(modelName, signInURL string) (string, error) {
		signInCalled = true
		signedIn = true
		if modelName != "qwen3.5:cloud" {
			t.Fatalf("unexpected model passed to sign-in: %q", modelName)
		}
		if signInURL != "https://example.com/signin" {
			t.Fatalf("unexpected sign-in URL: %q", signInURL)
		}
		return "test-user", nil
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[{"model":"qwen3.5:cloud","description":"Reasoning","context_length":262144,"max_output_tokens":32768}]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[]}`)
		case "/api/status":
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprint(w, `{"error":"not found"}`)
		case "/api/show":
			fmt.Fprint(w, `{"remote_model":"qwen3.5"}`)
		case "/api/me":
			if !signedIn {
				w.WriteHeader(http.StatusUnauthorized)
				fmt.Fprint(w, `{"error":"unauthorized","signin_url":"https://example.com/signin"}`)
				return
			}
			fmt.Fprint(w, `{"name":"test-user","plan":"free"}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	model, err := ResolveRunModel(context.Background(), RunModelRequest{ForcePicker: true})
	if err != nil {
		t.Fatalf("ResolveRunModel returned error: %v", err)
	}
	if model != "qwen3.5:cloud" {
		t.Fatalf("expected selected cloud model, got %q", model)
	}
	if !signInCalled {
		t.Fatal("expected sign-in hook to be used for account-gated cloud model")
	}
}

func TestResolveRunModel_SubscriptionModelUsesUpgradeHook(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	DefaultSingleSelectorWithUpdates = func(title string, items []SelectionItem, current string, updates <-chan []SelectionItem) (string, error) {
		for _, item := range items {
			if item.Name == "kimi-k2.6:cloud" {
				if item.AvailabilityBadge != "Upgrade required" {
					t.Fatalf("availability badge = %q, want Upgrade required", item.AvailabilityBadge)
				}
				if updates != nil {
					t.Fatal("expected no selector item update after synchronous account check")
				}
				return "kimi-k2.6:cloud", nil
			}
		}
		t.Fatalf("paid cloud model missing from selector items: %#v", items)
		return "kimi-k2.6:cloud", nil
	}
	DefaultSignIn = func(modelName, signInURL string) (string, error) {
		t.Fatalf("did not expect sign-in hook for signed-in user")
		return "", nil
	}

	plan := "free"
	upgradeCalled := false
	DefaultUpgrade = func(modelName, requiredPlan string) (string, error) {
		upgradeCalled = true
		if modelName != "kimi-k2.6:cloud" {
			t.Fatalf("unexpected model passed to upgrade: %q", modelName)
		}
		if requiredPlan != "pro" {
			t.Fatalf("unexpected required plan: %q", requiredPlan)
		}
		plan = "max"
		return plan, nil
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[{"model":"kimi-k2.6:cloud","description":"Coding","context_length":262144,"max_output_tokens":262144,"required_plan":"pro"}]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[]}`)
		case "/api/status":
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprint(w, `{"error":"not found"}`)
		case "/api/show":
			fmt.Fprint(w, `{"remote_model":"kimi-k2.6"}`)
		case "/api/me":
			fmt.Fprintf(w, `{"name":"test-user","plan":%q}`, plan)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	model, err := ResolveRunModel(context.Background(), RunModelRequest{ForcePicker: true})
	if err != nil {
		t.Fatalf("ResolveRunModel returned error: %v", err)
	}
	if model != "kimi-k2.6:cloud" {
		t.Fatalf("expected selected cloud model, got %q", model)
	}
	if !upgradeCalled {
		t.Fatal("expected upgrade hook to be used for subscription-gated cloud model")
	}
}

func TestResolveRunModel_UpgradeCancelledReturnsToModelSelector(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	selectorCalls := 0
	DefaultSingleSelector = func(title string, items []SelectionItem, current string) (string, error) {
		selectorCalls++
		switch selectorCalls {
		case 1:
			return "kimi-k2.6:cloud", nil
		case 2:
			return "sample-model", nil
		default:
			t.Fatalf("selector called too many times: %d", selectorCalls)
			return "", nil
		}
	}
	DefaultUpgrade = func(modelName, requiredPlan string) (string, error) {
		return "", ErrCancelled
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[{"model":"kimi-k2.6:cloud","description":"Coding","context_length":262144,"max_output_tokens":262144,"required_plan":"pro"}]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"sample-model"}]}`)
		case "/api/status":
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprint(w, `{"error":"not found"}`)
		case "/api/show":
			var req apiShowRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			fmt.Fprintf(w, `{"model":%q}`, req.Model)
		case "/api/me":
			fmt.Fprint(w, `{"name":"test-user","plan":"free"}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	model, err := ResolveRunModel(context.Background(), RunModelRequest{ForcePicker: true})
	if err != nil {
		t.Fatalf("ResolveRunModel returned error: %v", err)
	}
	if model != "sample-model" {
		t.Fatalf("model = %q, want sample-model", model)
	}
	if selectorCalls != 2 {
		t.Fatalf("selector calls = %d, want 2", selectorCalls)
	}
}

func TestResolveRunModel_SubscriptionModelUnavailableWhoamiAllowsSelection(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	DefaultSingleSelector = func(title string, items []SelectionItem, current string) (string, error) {
		return "kimi-k2.6:cloud", nil
	}
	DefaultUpgrade = func(modelName, requiredPlan string) (string, error) {
		t.Fatalf("did not expect upgrade hook when plan could not be verified")
		return "", nil
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[{"model":"kimi-k2.6:cloud","description":"Coding","context_length":262144,"max_output_tokens":262144,"required_plan":"pro"}]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[]}`)
		case "/api/status":
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprint(w, `{"error":"not found"}`)
		case "/api/show":
			fmt.Fprint(w, `{"remote_model":"kimi-k2.6"}`)
		case "/api/me":
			w.WriteHeader(http.StatusInternalServerError)
			fmt.Fprint(w, `{"error":"temporary failure"}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	model, err := ResolveRunModel(context.Background(), RunModelRequest{ForcePicker: true})
	if err != nil {
		t.Fatalf("ResolveRunModel returned error: %v", err)
	}
	if model != "kimi-k2.6:cloud" {
		t.Fatalf("expected selected cloud model, got %q", model)
	}
}

func TestLaunchIntegration_EditorForceConfigure(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	binDir := t.TempDir()
	writeFakeBinary(t, binDir, "droid")
	t.Setenv("PATH", binDir)

	settingsPath := filepath.Join(t.TempDir(), "settings.json")
	if err := os.WriteFile(settingsPath, []byte("{}"), 0o644); err != nil {
		t.Fatalf("failed to seed editor settings: %v", err)
	}
	editor := &launcherEditorRunner{paths: []string{settingsPath}}
	withIntegrationOverride(t, "droid", editor)

	var multiCalled bool
	DefaultMultiSelector = func(title string, items []SelectionItem, preChecked []string) ([]string, error) {
		multiCalled = true
		return []string{"sample-model", "qwen3:8b"}, nil
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"sample-model"},{"name":"qwen3:8b"}]}`)
		case "/api/show":
			var req apiShowRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			fmt.Fprintf(w, `{"model":%q}`, req.Model)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{
		Name:           "droid",
		ForceConfigure: true,
	}); err != nil {
		t.Fatalf("LaunchIntegration returned error: %v", err)
	}

	if !multiCalled {
		t.Fatal("expected multi selector to be used for forced editor configure")
	}
	if diff := compareStringSlices(editor.edited, [][]string{{"sample-model", "qwen3:8b"}}); diff != "" {
		t.Fatalf("unexpected edited models (-want +got):\n%s", diff)
	}
	if editor.ranModel != "sample-model" {
		t.Fatalf("expected launch to use first selected model, got %q", editor.ranModel)
	}
	saved, err := config.LoadIntegration("droid")
	if err != nil {
		t.Fatalf("failed to reload saved config: %v", err)
	}
	if diff := compareStrings(saved.Models, []string{"sample-model", "qwen3:8b"}); diff != "" {
		t.Fatalf("unexpected saved models (-want +got):\n%s", diff)
	}
}

func TestLaunchIntegration_ClineRewritesWhenLiveProviderDrifted(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	binDir := t.TempDir()
	writeFakeBinary(t, binDir, "cline")
	t.Setenv("PATH", binDir)

	if err := config.SaveIntegration("cline", []string{"sample-model"}); err != nil {
		t.Fatalf("failed to seed saved config: %v", err)
	}

	providersPath := clineProvidersPath(tmpDir)
	if err := os.MkdirAll(filepath.Dir(providersPath), 0o755); err != nil {
		t.Fatalf("failed to create providers dir: %v", err)
	}
	existingProviders := map[string]any{
		"version":          float64(1),
		"lastUsedProvider": "openai-codex-cli",
		"providers": map[string]any{
			"openai-codex-cli": map[string]any{
				"settings": map[string]any{
					"provider":  "openai-codex-cli",
					"model":     "gpt-5.5",
					"reasoning": "medium",
				},
				"updatedAt":   "2026-06-01T12:00:00Z",
				"tokenSource": "manual",
			},
		},
	}
	data, _ := json.Marshal(existingProviders)
	if err := os.WriteFile(providersPath, data, 0o644); err != nil {
		t.Fatalf("failed to seed providers config: %v", err)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			var req apiShowRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			fmt.Fprintf(w, `{"model":%q}`, req.Model)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{
		Name: "cline",
	}); err != nil {
		t.Fatalf("LaunchIntegration returned error: %v", err)
	}

	providersConfig, err := fileutil.ReadJSON(providersPath)
	if err != nil {
		t.Fatalf("failed to read providers config: %v", err)
	}
	if providersConfig["lastUsedProvider"] != clineLaunchProvider {
		t.Fatalf("lastUsedProvider = %v, want %s", providersConfig["lastUsedProvider"], clineLaunchProvider)
	}
	providers, _ := providersConfig["providers"].(map[string]any)
	if _, ok := providers["openai-codex-cli"]; !ok {
		t.Fatal("expected existing openai-codex-cli provider to be preserved")
	}
	ollamaProvider, _ := providers[clineLaunchProvider].(map[string]any)
	settings, _ := ollamaProvider["settings"].(map[string]any)
	if settings["provider"] != clineLaunchProvider {
		t.Fatalf("ollama settings.provider = %v, want %s", settings["provider"], clineLaunchProvider)
	}
	if settings["model"] != "sample-model" {
		t.Fatalf("ollama settings.model = %v, want sample-model", settings["model"])
	}
	if settings["baseUrl"] != srv.URL+"/v1" {
		t.Fatalf("ollama settings.baseUrl = %v, want %s/v1", settings["baseUrl"], srv.URL)
	}
}

func TestLaunchIntegration_EditorForceConfigure_FloatsCheckedModelsInPicker(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	binDir := t.TempDir()
	writeFakeBinary(t, binDir, "droid")
	t.Setenv("PATH", binDir)

	editor := &launcherEditorRunner{models: []string{"sample-model", "missing-local"}}
	withIntegrationOverride(t, "droid", editor)

	if err := config.SaveIntegration("droid", []string{"qwen3.5:cloud", "qwen3.5"}); err != nil {
		t.Fatalf("failed to seed config: %v", err)
	}

	var gotItems []string
	var gotPreChecked []string
	DefaultMultiSelector = func(title string, items []SelectionItem, preChecked []string) ([]string, error) {
		for _, item := range items {
			gotItems = append(gotItems, item.Name)
		}
		gotPreChecked = append([]string(nil), preChecked...)
		return []string{"qwen3.5:cloud", "qwen3.5"}, nil
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"qwen3.5:cloud","remote_model":"qwen3.5"},{"name":"qwen3.5"}]}`)
		case "/api/show":
			var req apiShowRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			if req.Model == "qwen3.5:cloud" {
				fmt.Fprint(w, `{"remote_model":"qwen3.5"}`)
				return
			}
			fmt.Fprintf(w, `{"model":%q}`, req.Model)
		case "/api/me":
			fmt.Fprint(w, `{"name":"test-user"}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{
		Name:           "droid",
		ForceConfigure: true,
	}); err != nil {
		t.Fatalf("LaunchIntegration returned error: %v", err)
	}

	if len(gotItems) == 0 {
		t.Fatal("expected multi selector to receive items")
	}
	wantItems := recommendedNames()
	if diff := cmp.Diff(wantItems, gotItems); diff != "" {
		t.Fatalf("expected fixed recommended order in selector items (-want +got):\n%s", diff)
	}
	if len(gotPreChecked) < 2 {
		t.Fatalf("expected prechecked models to be preserved, got %v", gotPreChecked)
	}
	if gotPreChecked[0] != "qwen3.5:cloud" {
		t.Fatalf("expected saved default to remain first in prechecked, got %v", gotPreChecked)
	}
}

func TestLaunchIntegration_EditorModelOverridePreservesExtras(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	binDir := t.TempDir()
	writeFakeBinary(t, binDir, "droid")
	t.Setenv("PATH", binDir)

	editor := &launcherEditorRunner{}
	withIntegrationOverride(t, "droid", editor)

	if err := config.SaveIntegration("droid", []string{"sample-model", "mistral"}); err != nil {
		t.Fatalf("failed to seed config: %v", err)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/show" {
			var req apiShowRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			fmt.Fprintf(w, `{"model":%q}`, req.Model)
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{
		Name:          "droid",
		ModelOverride: "qwen3:8b",
	}); err != nil {
		t.Fatalf("LaunchIntegration returned error: %v", err)
	}

	want := []string{"qwen3:8b", "sample-model", "mistral"}
	saved, err := config.LoadIntegration("droid")
	if err != nil {
		t.Fatalf("failed to reload saved config: %v", err)
	}
	if diff := compareStrings(saved.Models, want); diff != "" {
		t.Fatalf("unexpected saved models (-want +got):\n%s", diff)
	}
	if diff := compareStringSlices(editor.edited, [][]string{want}); diff != "" {
		t.Fatalf("unexpected edited models (-want +got):\n%s", diff)
	}
	if editor.ranModel != "qwen3:8b" {
		t.Fatalf("expected override model to launch first, got %q", editor.ranModel)
	}
}

func TestLaunchIntegration_EditorCloudDisabledFallsBackToSelector(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	binDir := t.TempDir()
	writeFakeBinary(t, binDir, "droid")
	t.Setenv("PATH", binDir)

	editor := &launcherEditorRunner{}
	withIntegrationOverride(t, "droid", editor)

	if err := config.SaveIntegration("droid", []string{"glm-5:cloud"}); err != nil {
		t.Fatalf("failed to seed config: %v", err)
	}

	var multiCalled bool
	DefaultMultiSelector = func(title string, items []SelectionItem, preChecked []string) ([]string, error) {
		multiCalled = true
		return []string{"sample-model"}, nil
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/status":
			fmt.Fprint(w, `{"cloud":{"disabled":true,"source":"config"}}`)
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"sample-model"}]}`)
		case "/api/show":
			fmt.Fprint(w, `{"model":"sample-model"}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{Name: "droid"}); err != nil {
		t.Fatalf("LaunchIntegration returned error: %v", err)
	}
	if !multiCalled {
		t.Fatal("expected editor flow to reopen selector when cloud-only config is unusable")
	}
}

func TestLaunchIntegration_EditorConfigureMultiSkipsMissingLocalAndPersistsAccepted(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	binDir := t.TempDir()
	writeFakeBinary(t, binDir, "droid")
	t.Setenv("PATH", binDir)

	editor := &launcherEditorRunner{}
	withIntegrationOverride(t, "droid", editor)

	DefaultMultiSelector = func(title string, items []SelectionItem, preChecked []string) ([]string, error) {
		return []string{"glm-5:cloud", "missing-local"}, nil
	}
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		if prompt == "Download missing-local?" {
			return false, nil
		}
		t.Fatalf("unexpected prompt: %q", prompt)
		return false, nil
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"glm-5:cloud","remote_model":"glm-5"}]}`)
		case "/api/status":
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprint(w, `{"error":"not found"}`)
		case "/api/show":
			var req apiShowRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			switch req.Model {
			case "glm-5:cloud":
				fmt.Fprint(w, `{"remote_model":"glm-5"}`)
			case "missing-local":
				w.WriteHeader(http.StatusNotFound)
				fmt.Fprint(w, `{"error":"model not found"}`)
			default:
				http.NotFound(w, r)
			}
		case "/api/me":
			fmt.Fprint(w, `{"name":"test-user"}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	var launchErr error
	stderr := captureStderr(t, func() {
		launchErr = LaunchIntegration(context.Background(), IntegrationLaunchRequest{
			Name:           "droid",
			ForceConfigure: true,
		})
	})
	if launchErr != nil {
		t.Fatalf("LaunchIntegration returned error: %v", launchErr)
	}
	if editor.ranModel != "glm-5:cloud" {
		t.Fatalf("expected launch to use cloud primary, got %q", editor.ranModel)
	}
	saved, err := config.LoadIntegration("droid")
	if err != nil {
		t.Fatalf("failed to reload saved config: %v", err)
	}
	if diff := compareStrings(saved.Models, []string{"glm-5:cloud"}); diff != "" {
		t.Fatalf("unexpected saved models (-want +got):\n%s", diff)
	}
	if diff := compareStringSlices(editor.edited, [][]string{{"glm-5:cloud"}}); diff != "" {
		t.Fatalf("unexpected edited models (-want +got):\n%s", diff)
	}
	if !strings.Contains(stderr, "Skipped missing-local:") {
		t.Fatalf("expected skip reason in stderr, got %q", stderr)
	}
}

func TestLaunchIntegration_EditorConfigureMultiSkipsUnauthedCloudAndPersistsAccepted(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	binDir := t.TempDir()
	writeFakeBinary(t, binDir, "droid")
	t.Setenv("PATH", binDir)

	editor := &launcherEditorRunner{}
	withIntegrationOverride(t, "droid", editor)

	DefaultMultiSelector = func(title string, items []SelectionItem, preChecked []string) ([]string, error) {
		return []string{"sample-model", "glm-5:cloud"}, nil
	}
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		t.Fatalf("unexpected prompt: %q", prompt)
		return false, nil
	}
	DefaultSignIn = func(modelName, signInURL string) (string, error) {
		return "", ErrCancelled
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"sample-model"},{"name":"glm-5:cloud","remote_model":"glm-5"}]}`)
		case "/api/status":
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprint(w, `{"error":"not found"}`)
		case "/api/show":
			var req apiShowRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			switch req.Model {
			case "sample-model":
				fmt.Fprint(w, `{"model":"sample-model"}`)
			case "glm-5:cloud":
				fmt.Fprint(w, `{"remote_model":"glm-5"}`)
			default:
				http.NotFound(w, r)
			}
		case "/api/me":
			w.WriteHeader(http.StatusUnauthorized)
			fmt.Fprint(w, `{"error":"unauthorized","signin_url":"https://example.com/signin"}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	var launchErr error
	stderr := captureStderr(t, func() {
		launchErr = LaunchIntegration(context.Background(), IntegrationLaunchRequest{
			Name:           "droid",
			ForceConfigure: true,
		})
	})
	if launchErr != nil {
		t.Fatalf("LaunchIntegration returned error: %v", launchErr)
	}
	if editor.ranModel != "sample-model" {
		t.Fatalf("expected launch to use local primary, got %q", editor.ranModel)
	}
	saved, err := config.LoadIntegration("droid")
	if err != nil {
		t.Fatalf("failed to reload saved config: %v", err)
	}
	if diff := compareStrings(saved.Models, []string{"sample-model"}); diff != "" {
		t.Fatalf("unexpected saved models (-want +got):\n%s", diff)
	}
	if diff := compareStringSlices(editor.edited, [][]string{{"sample-model"}}); diff != "" {
		t.Fatalf("unexpected edited models (-want +got):\n%s", diff)
	}
	if !strings.Contains(stderr, "Skipped glm-5:cloud: sign in was cancelled") {
		t.Fatalf("expected skip reason in stderr, got %q", stderr)
	}
}

func TestLaunchIntegration_EditorConfigureUpgradeCancelledReturnsToModelSelector(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	binDir := t.TempDir()
	writeFakeBinary(t, binDir, "droid")
	t.Setenv("PATH", binDir)

	editor := &launcherEditorRunner{}
	withIntegrationOverride(t, "droid", editor)

	selectorCalls := 0
	DefaultMultiSelector = func(title string, items []SelectionItem, preChecked []string) ([]string, error) {
		selectorCalls++
		switch selectorCalls {
		case 1:
			return []string{"kimi-k2.6:cloud"}, nil
		case 2:
			if diff := compareStrings(preChecked, []string{"kimi-k2.6:cloud"}); diff != "" {
				t.Fatalf("second selector preChecked (-want +got):\n%s", diff)
			}
			return []string{"sample-model"}, nil
		default:
			t.Fatalf("selector called too many times: %d", selectorCalls)
			return nil, nil
		}
	}
	DefaultUpgrade = func(modelName, requiredPlan string) (string, error) {
		return "", ErrCancelled
	}
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		if prompt == "Proceed?" {
			return true, nil
		}
		t.Fatalf("unexpected prompt: %q", prompt)
		return false, nil
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[{"model":"kimi-k2.6:cloud","description":"Coding","context_length":262144,"max_output_tokens":262144,"required_plan":"pro"}]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"sample-model"}]}`)
		case "/api/status":
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprint(w, `{"error":"not found"}`)
		case "/api/show":
			var req apiShowRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			fmt.Fprintf(w, `{"model":%q}`, req.Model)
		case "/api/me":
			fmt.Fprint(w, `{"name":"test-user","plan":"free"}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{
		Name:           "droid",
		ForceConfigure: true,
	}); err != nil {
		t.Fatalf("LaunchIntegration returned error: %v", err)
	}
	if selectorCalls != 2 {
		t.Fatalf("selector calls = %d, want 2", selectorCalls)
	}
	if editor.ranModel != "sample-model" {
		t.Fatalf("expected launch to use local model, got %q", editor.ranModel)
	}
	if diff := compareStringSlices(editor.edited, [][]string{{"sample-model"}}); diff != "" {
		t.Fatalf("unexpected edited models (-want +got):\n%s", diff)
	}
}

func TestLaunchIntegration_EditorConfigureMultiRemovesReselectedFailingModel(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	binDir := t.TempDir()
	writeFakeBinary(t, binDir, "droid")
	t.Setenv("PATH", binDir)

	editor := &launcherEditorRunner{}
	withIntegrationOverride(t, "droid", editor)

	if err := config.SaveIntegration("droid", []string{"glm-5:cloud", "sample-model"}); err != nil {
		t.Fatalf("failed to seed config: %v", err)
	}
	DefaultMultiSelector = func(title string, items []SelectionItem, preChecked []string) ([]string, error) {
		return append([]string(nil), preChecked...), nil
	}
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		t.Fatalf("unexpected prompt: %q", prompt)
		return false, nil
	}
	DefaultSignIn = func(modelName, signInURL string) (string, error) {
		return "", ErrCancelled
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"glm-5:cloud","remote_model":"glm-5"},{"name":"sample-model"}]}`)
		case "/api/status":
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprint(w, `{"error":"not found"}`)
		case "/api/show":
			var req apiShowRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			if req.Model == "glm-5:cloud" {
				fmt.Fprint(w, `{"remote_model":"glm-5"}`)
				return
			}
			if req.Model == "sample-model" {
				fmt.Fprint(w, `{"model":"sample-model"}`)
				return
			}
			http.NotFound(w, r)
		case "/api/me":
			w.WriteHeader(http.StatusUnauthorized)
			fmt.Fprint(w, `{"error":"unauthorized","signin_url":"https://example.com/signin"}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	var launchErr error
	stderr := captureStderr(t, func() {
		launchErr = LaunchIntegration(context.Background(), IntegrationLaunchRequest{
			Name:           "droid",
			ForceConfigure: true,
		})
	})
	if launchErr != nil {
		t.Fatalf("LaunchIntegration returned error: %v", launchErr)
	}
	if editor.ranModel != "sample-model" {
		t.Fatalf("expected launch to use surviving model, got %q", editor.ranModel)
	}
	if diff := compareStringSlices(editor.edited, [][]string{{"sample-model"}}); diff != "" {
		t.Fatalf("unexpected edited models (-want +got):\n%s", diff)
	}
	saved, loadErr := config.LoadIntegration("droid")
	if loadErr != nil {
		t.Fatalf("failed to reload saved config: %v", loadErr)
	}
	if diff := compareStrings(saved.Models, []string{"sample-model"}); diff != "" {
		t.Fatalf("unexpected saved models (-want +got):\n%s", diff)
	}
	if !strings.Contains(stderr, "Skipped glm-5:cloud: sign in was cancelled") {
		t.Fatalf("expected skip reason in stderr, got %q", stderr)
	}
}

func TestLaunchIntegration_EditorConfigureMultiAllFailuresKeepsExistingAndSkipsLaunch(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	binDir := t.TempDir()
	writeFakeBinary(t, binDir, "droid")
	t.Setenv("PATH", binDir)

	editor := &launcherEditorRunner{}
	withIntegrationOverride(t, "droid", editor)

	if err := config.SaveIntegration("droid", []string{"sample-model"}); err != nil {
		t.Fatalf("failed to seed config: %v", err)
	}

	DefaultMultiSelector = func(title string, items []SelectionItem, preChecked []string) ([]string, error) {
		return []string{"missing-local-a", "missing-local-b"}, nil
	}
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		if prompt == "Download missing-local-a?" || prompt == "Download missing-local-b?" {
			return false, nil
		}
		t.Fatalf("unexpected prompt: %q", prompt)
		return false, nil
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[]}`)
		case "/api/show":
			var req apiShowRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			switch req.Model {
			case "missing-local-a", "missing-local-b":
				w.WriteHeader(http.StatusNotFound)
				fmt.Fprint(w, `{"error":"model not found"}`)
			default:
				http.NotFound(w, r)
			}
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	var launchErr error
	stderr := captureStderr(t, func() {
		launchErr = LaunchIntegration(context.Background(), IntegrationLaunchRequest{
			Name:           "droid",
			ForceConfigure: true,
		})
	})
	if launchErr != nil {
		t.Fatalf("LaunchIntegration returned error: %v", launchErr)
	}
	if editor.ranModel != "" {
		t.Fatalf("expected no launch when all selected models are skipped, got %q", editor.ranModel)
	}
	if len(editor.edited) != 0 {
		t.Fatalf("expected no editor writes when all selections fail, got %v", editor.edited)
	}
	saved, err := config.LoadIntegration("droid")
	if err != nil {
		t.Fatalf("failed to reload saved config: %v", err)
	}
	if diff := compareStrings(saved.Models, []string{"sample-model"}); diff != "" {
		t.Fatalf("unexpected saved models (-want +got):\n%s", diff)
	}
	if !strings.Contains(stderr, "Skipped missing-local-a:") {
		t.Fatalf("expected first skip reason in stderr, got %q", stderr)
	}
	if !strings.Contains(stderr, "Skipped missing-local-b:") {
		t.Fatalf("expected second skip reason in stderr, got %q", stderr)
	}
}

func TestLaunchIntegration_ConfiguredEditorLaunchValidatesPrimaryOnly(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	binDir := t.TempDir()
	writeFakeBinary(t, binDir, "droid")
	t.Setenv("PATH", binDir)

	editor := &launcherEditorRunner{models: []string{"sample-model", "missing-local"}}
	withIntegrationOverride(t, "droid", editor)

	if err := config.SaveIntegration("droid", []string{"sample-model", "missing-local"}); err != nil {
		t.Fatalf("failed to seed config: %v", err)
	}

	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		t.Fatalf("did not expect prompt during normal configured launch: %q", prompt)
		return false, nil
	}

	var missingShowCalled bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/show" {
			http.NotFound(w, r)
			return
		}
		var req apiShowRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		switch req.Model {
		case "sample-model":
			fmt.Fprint(w, `{"model":"sample-model"}`)
		case "missing-local":
			missingShowCalled = true
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprint(w, `{"error":"model not found"}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{Name: "droid"}); err != nil {
		t.Fatalf("LaunchIntegration returned error: %v", err)
	}
	if missingShowCalled {
		t.Fatal("expected configured launch to validate only the primary model")
	}
	if editor.ranModel != "sample-model" {
		t.Fatalf("expected launch to use saved primary model, got %q", editor.ranModel)
	}
	if len(editor.edited) != 0 {
		t.Fatalf("expected no editor writes during normal launch, got %v", editor.edited)
	}

	saved, err := config.LoadIntegration("droid")
	if err != nil {
		t.Fatalf("failed to reload saved config: %v", err)
	}
	if diff := compareStrings(saved.Models, []string{"sample-model", "missing-local"}); diff != "" {
		t.Fatalf("unexpected saved models (-want +got):\n%s", diff)
	}
}

func TestLaunchIntegration_ConfiguredEditorLaunchSkipsReconfigure(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	binDir := t.TempDir()
	writeFakeBinary(t, binDir, "droid")
	t.Setenv("PATH", binDir)

	settingsPath := filepath.Join(t.TempDir(), "settings.json")
	if err := os.WriteFile(settingsPath, []byte("{}"), 0o644); err != nil {
		t.Fatalf("failed to seed editor settings: %v", err)
	}
	editor := &launcherEditorRunner{paths: []string{settingsPath}, models: []string{"sample-model", "qwen3:8b"}}
	withIntegrationOverride(t, "droid", editor)

	if err := config.SaveIntegration("droid", []string{"sample-model", "qwen3:8b"}); err != nil {
		t.Fatalf("failed to seed config: %v", err)
	}

	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		t.Fatalf("did not expect prompt during a normal editor launch: %s", prompt)
		return false, nil
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/show" {
			var req apiShowRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			fmt.Fprintf(w, `{"model":%q}`, req.Model)
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{Name: "droid"}); err != nil {
		t.Fatalf("LaunchIntegration returned error: %v", err)
	}
	if len(editor.edited) != 0 {
		t.Fatalf("expected normal launch to skip editor rewrites, got %v", editor.edited)
	}
	if editor.ranModel != "sample-model" {
		t.Fatalf("expected launch to use saved primary model, got %q", editor.ranModel)
	}

	saved, err := config.LoadIntegration("droid")
	if err != nil {
		t.Fatalf("failed to reload saved config: %v", err)
	}
	if diff := compareStrings(saved.Models, []string{"sample-model", "qwen3:8b"}); diff != "" {
		t.Fatalf("unexpected saved models (-want +got):\n%s", diff)
	}
}

func TestLaunchIntegration_ConfiguredEditorLaunchRewritesDriftedLiveConfig(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	binDir := t.TempDir()
	writeFakeBinary(t, binDir, "droid")
	t.Setenv("PATH", binDir)

	editor := &launcherEditorRunner{models: []string{"qwen3:8b"}}
	withIntegrationOverride(t, "droid", editor)

	if err := config.SaveIntegration("droid", []string{"sample-model", "mistral"}); err != nil {
		t.Fatalf("failed to seed config: %v", err)
	}

	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		t.Fatalf("did not expect prompt during a normal editor launch: %s", prompt)
		return false, nil
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/show" {
			var req apiShowRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			fmt.Fprintf(w, `{"model":%q}`, req.Model)
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{Name: "droid"}); err != nil {
		t.Fatalf("LaunchIntegration returned error: %v", err)
	}
	if diff := cmp.Diff([][]string{{"sample-model", "mistral"}}, editor.edited); diff != "" {
		t.Fatalf("expected editor config rewrite when live config drifts (-want +got):\n%s", diff)
	}
	if editor.ranModel != "sample-model" {
		t.Fatalf("expected launch to use saved primary model, got %q", editor.ranModel)
	}

	saved, err := config.LoadIntegration("droid")
	if err != nil {
		t.Fatalf("failed to reload saved config: %v", err)
	}
	if diff := compareStrings(saved.Models, []string{"sample-model", "mistral"}); diff != "" {
		t.Fatalf("unexpected saved models (-want +got):\n%s", diff)
	}
}

func TestLaunchIntegration_OpenclawPreservesExistingModelList(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	binDir := t.TempDir()
	writeFakeBinary(t, binDir, "openclaw")
	t.Setenv("PATH", binDir)

	editor := &launcherEditorRunner{models: []string{"sample-model", "mistral"}}
	withIntegrationOverride(t, "openclaw", editor)

	if err := config.SaveIntegration("openclaw", []string{"sample-model", "mistral"}); err != nil {
		t.Fatalf("failed to seed config: %v", err)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/show" {
			var req apiShowRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			fmt.Fprintf(w, `{"model":%q}`, req.Model)
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{Name: "openclaw"}); err != nil {
		t.Fatalf("LaunchIntegration returned error: %v", err)
	}
	if len(editor.edited) != 0 {
		t.Fatalf("expected launch to preserve the existing OpenClaw config, got rewrites %v", editor.edited)
	}
	if editor.ranModel != "sample-model" {
		t.Fatalf("expected launch to use first saved model, got %q", editor.ranModel)
	}

	saved, err := config.LoadIntegration("openclaw")
	if err != nil {
		t.Fatalf("failed to reload saved config: %v", err)
	}
	if diff := compareStrings(saved.Models, []string{"sample-model", "mistral"}); diff != "" {
		t.Fatalf("unexpected saved models (-want +got):\n%s", diff)
	}
}

func TestLaunchIntegration_OpenclawInstallsBeforeConfigSideEffects(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	t.Setenv("PATH", t.TempDir())

	editor := &launcherEditorRunner{}
	withIntegrationOverride(t, "openclaw", editor)

	selectorCalled := false
	DefaultMultiSelector = func(title string, items []SelectionItem, preChecked []string) ([]string, error) {
		selectorCalled = true
		return []string{"sample-model"}, nil
	}

	err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{Name: "openclaw"})
	if err == nil {
		t.Fatal("expected launch to fail before configuration when OpenClaw is missing")
	}
	if !strings.Contains(err.Error(), "required dependencies are missing") {
		t.Fatalf("expected install prerequisite error, got %v", err)
	}
	if selectorCalled {
		t.Fatal("expected install check to happen before model selection")
	}
	if len(editor.edited) != 0 {
		t.Fatalf("expected no editor writes before install succeeds, got %v", editor.edited)
	}
	if _, statErr := os.Stat(filepath.Join(tmpDir, ".openclaw", "openclaw.json")); !os.IsNotExist(statErr) {
		t.Fatalf("expected no OpenClaw config file to be created, stat err = %v", statErr)
	}
}

func TestLaunchIntegration_PiInstallsBeforeConfigSideEffects(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	t.Setenv("PATH", t.TempDir())

	editor := &launcherEditorRunner{}
	withIntegrationOverride(t, "pi", editor)

	selectorCalled := false
	DefaultMultiSelector = func(title string, items []SelectionItem, preChecked []string) ([]string, error) {
		selectorCalled = true
		return []string{"sample-model"}, nil
	}

	err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{Name: "pi"})
	if err == nil {
		t.Fatal("expected launch to fail before configuration when Pi is missing")
	}
	if !strings.Contains(err.Error(), "required dependencies are missing") {
		t.Fatalf("expected install prerequisite error, got %v", err)
	}
	if selectorCalled {
		t.Fatal("expected install check to happen before model selection")
	}
	if len(editor.edited) != 0 {
		t.Fatalf("expected no editor writes before install succeeds, got %v", editor.edited)
	}
	if _, statErr := os.Stat(filepath.Join(tmpDir, ".pi", "agent", "models.json")); !os.IsNotExist(statErr) {
		t.Fatalf("expected no Pi config file to be created, stat err = %v", statErr)
	}
}

func TestLaunchIntegration_ConfigureOnlyDoesNotRequireInstalledBinary(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)
	t.Setenv("PATH", t.TempDir())

	editor := &launcherEditorRunner{paths: []string{"/tmp/settings.json"}}
	withIntegrationOverride(t, "droid", editor)

	DefaultMultiSelector = func(title string, items []SelectionItem, preChecked []string) ([]string, error) {
		return []string{"sample-model"}, nil
	}

	var prompts []string
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		prompts = append(prompts, prompt)
		if strings.Contains(prompt, "Launch LauncherEditor now?") {
			return false, nil
		}
		return true, nil
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"sample-model"}]}`)
		case "/api/show":
			fmt.Fprint(w, `{"model":"sample-model"}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{
		Name:           "droid",
		ForceConfigure: true,
		ConfigureOnly:  true,
	}); err != nil {
		t.Fatalf("LaunchIntegration returned error: %v", err)
	}
	if diff := compareStringSlices(editor.edited, [][]string{{"sample-model"}}); diff != "" {
		t.Fatalf("unexpected edited models (-want +got):\n%s", diff)
	}
	if editor.ranModel != "" {
		t.Fatalf("expected configure-only flow to skip launch, got %q", editor.ranModel)
	}
	if !slices.Contains(prompts, "Launch LauncherEditor now?") {
		t.Fatalf("expected configure-only launch prompt, got %v", prompts)
	}
}

func TestLaunchIntegration_ClaudeSavesPrimaryModel(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)

	binDir := t.TempDir()
	writeFakeBinary(t, binDir, "claude")
	t.Setenv("PATH", binDir)

	var aliasSyncCalled bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[]}`)
		case "/api/status":
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprint(w, `{"error":"not found"}`)
		case "/api/show":
			fmt.Fprint(w, `{"remote_model":"glm-5"}`)
		case "/api/me":
			fmt.Fprint(w, `{"name":"test-user"}`)
		case "/api/experimental/aliases":
			aliasSyncCalled = true
			t.Fatalf("did not expect alias sync call after removing Claude alias flow")
		default:
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{
		Name:          "claude",
		ModelOverride: "glm-5:cloud",
	}); err != nil {
		t.Fatalf("LaunchIntegration returned error: %v", err)
	}

	saved, err := config.LoadIntegration("claude")
	if err != nil {
		t.Fatalf("failed to reload saved config: %v", err)
	}
	if diff := compareStrings(saved.Models, []string{"glm-5:cloud"}); diff != "" {
		t.Fatalf("unexpected saved models (-want +got):\n%s", diff)
	}
	if aliasSyncCalled {
		t.Fatal("expected Claude launch flow not to sync aliases")
	}
}

func TestLaunchIntegration_ClaudeForceConfigureReprompts(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	binDir := t.TempDir()
	writeFakeBinary(t, binDir, "claude")
	t.Setenv("PATH", binDir)

	if err := config.SaveIntegration("claude", []string{"qwen3:8b"}); err != nil {
		t.Fatalf("failed to seed config: %v", err)
	}

	var selectorCalls int
	DefaultSingleSelector = func(title string, items []SelectionItem, current string) (string, error) {
		selectorCalls++
		return "glm-5:cloud", nil
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"qwen3:8b"}]}`)
		case "/api/show":
			fmt.Fprint(w, `{"model":"qwen3:8b"}`)
		case "/api/me":
			fmt.Fprint(w, `{"name":"test-user"}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{
		Name:           "claude",
		ForceConfigure: true,
	}); err != nil {
		t.Fatalf("LaunchIntegration returned error: %v", err)
	}
	if selectorCalls != 1 {
		t.Fatalf("expected forced configure to reprompt for model selection, got %d calls", selectorCalls)
	}
	saved, err := config.LoadIntegration("claude")
	if err != nil {
		t.Fatalf("failed to reload saved config: %v", err)
	}
	if saved.Models[0] != "glm-5:cloud" {
		t.Fatalf("expected saved primary to be replaced, got %q", saved.Models[0])
	}
}

func TestLaunchIntegration_ClaudeForceConfigureMissingSelectionDoesNotSave(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	binDir := t.TempDir()
	writeFakeBinary(t, binDir, "claude")
	t.Setenv("PATH", binDir)

	if err := config.SaveIntegration("claude", []string{"sample-model"}); err != nil {
		t.Fatalf("failed to seed config: %v", err)
	}

	DefaultSingleSelector = func(title string, items []SelectionItem, current string) (string, error) {
		return "missing-model", nil
	}
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		if prompt == "Download missing-model?" {
			return false, nil
		}
		t.Fatalf("unexpected prompt: %q", prompt)
		return false, nil
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"sample-model"}]}`)
		case "/api/show":
			var req apiShowRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			if req.Model == "missing-model" {
				w.WriteHeader(http.StatusNotFound)
				fmt.Fprint(w, `{"error":"model not found"}`)
				return
			}
			fmt.Fprintf(w, `{"model":%q}`, req.Model)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{
		Name:           "claude",
		ForceConfigure: true,
	})
	if err == nil {
		t.Fatal("expected missing selected model to abort launch")
	}

	saved, loadErr := config.LoadIntegration("claude")
	if loadErr != nil {
		t.Fatalf("failed to reload saved config: %v", loadErr)
	}
	if diff := compareStrings(saved.Models, []string{"sample-model"}); diff != "" {
		t.Fatalf("unexpected saved models (-want +got):\n%s", diff)
	}
}

func TestLaunchIntegration_ClaudeModelOverrideSkipsSelector(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)
	withInteractiveSession(t, true)

	binDir := t.TempDir()
	writeFakeBinary(t, binDir, "claude")
	t.Setenv("PATH", binDir)

	var selectorCalls int
	DefaultSingleSelector = func(title string, items []SelectionItem, current string) (string, error) {
		selectorCalls++
		return "", fmt.Errorf("selector should not run when --model override is set")
	}

	var confirmCalls int
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		confirmCalls++
		if !strings.Contains(prompt, "glm-4") {
			t.Fatalf("expected download prompt for override model, got %q", prompt)
		}
		return true, nil
	}

	var pullCalled bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprint(w, `{"error":"model not found"}`)
		case "/api/pull":
			pullCalled = true
			w.WriteHeader(http.StatusOK)
			fmt.Fprint(w, `{"status":"success"}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{
		Name:          "claude",
		ModelOverride: "glm-4",
	}); err != nil {
		t.Fatalf("LaunchIntegration returned error: %v", err)
	}

	if selectorCalls != 0 {
		t.Fatalf("expected model override to skip selector, got %d calls", selectorCalls)
	}
	if confirmCalls == 0 {
		t.Fatal("expected missing override model to prompt for download in interactive mode")
	}
	if !pullCalled {
		t.Fatal("expected missing override model to be pulled after confirmation")
	}

	saved, err := config.LoadIntegration("claude")
	if err != nil {
		t.Fatalf("failed to reload saved config: %v", err)
	}
	if saved.Models[0] != "glm-4" {
		t.Fatalf("expected saved primary to match override, got %q", saved.Models[0])
	}
}

func TestLaunchIntegration_ClaudeModelOverrideDeprecatedDeclineOpensPicker(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)
	withInteractiveSession(t, true)

	binDir := t.TempDir()
	writeFakeBinary(t, binDir, "claude")
	t.Setenv("PATH", binDir)

	var showCalls atomic.Int32
	var prompt string
	var promptOptions ConfirmOptions
	DefaultConfirmPrompt = func(p string, options ConfirmOptions) (bool, error) {
		prompt = p
		promptOptions = options
		return false, nil
	}
	var selectorCalls int
	DefaultSingleSelector = func(title string, items []SelectionItem, current string) (string, error) {
		selectorCalls++
		if title != "Select model for Claude Code:" {
			t.Fatalf("picker title = %q, want Claude Code model picker", title)
		}
		if current != "llama3.2" {
			t.Fatalf("picker current = %q, want deprecated override", current)
		}
		itemNames := selectionItemNames(items)
		if slices.Contains(itemNames, "llama3.2") {
			t.Fatalf("expected deprecated override to be hidden from picker, got %v", itemNames)
		}
		for _, model := range []string{"best-cloud:cloud", "best-local"} {
			if !slices.Contains(itemNames, model) {
				t.Fatalf("expected compatible model %q in picker, got %v", model, itemNames)
			}
		}
		return "best-local", nil
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[`+
				`{"model":"best-cloud:cloud","description":"Cloud rec","context_length":262144,"max_output_tokens":32768},`+
				`{"model":"best-local","description":"Local rec"}`+
				`]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"llama3.2"},{"name":"best-local"}]}`)
		case "/api/show":
			showCalls.Add(1)
			fmt.Fprint(w, `{"model_info":{"general.context_length":131072}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{
		Name:          "claude",
		ModelOverride: "llama3.2",
	}); err != nil {
		t.Fatalf("LaunchIntegration returned error: %v", err)
	}
	for _, want := range []string{"llama3.2 does not work well with Claude Code", "best-cloud:cloud", "best-local", "ollama launch claude --model best-cloud:cloud"} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("prompt %q does not contain %q", prompt, want)
		}
	}
	if promptOptions.YesLabel != "Launch anyway" || promptOptions.NoLabel != "Pick another model" || promptOptions.Default != ConfirmDefaultNo {
		t.Fatalf("unexpected deprecation prompt options: %+v", promptOptions)
	}
	if selectorCalls != 1 {
		t.Fatalf("expected picker to open after declining override, got %d calls", selectorCalls)
	}
	if showCalls.Load() != 1 {
		t.Fatalf("expected only replacement model readiness to call /api/show, got %d calls", showCalls.Load())
	}
	saved, err := config.LoadIntegration("claude")
	if err != nil {
		t.Fatalf("failed to reload saved config: %v", err)
	}
	if got := primaryModelFromConfig(saved); got != "best-local" {
		t.Fatalf("expected picked model to replace deprecated override, got %q", got)
	}
}

func TestLaunchIntegration_SavedDeprecatedDeclineOpensPicker(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)
	withInteractiveSession(t, true)

	if err := config.SaveIntegration("droid", []string{"llama3.2"}); err != nil {
		t.Fatalf("failed to seed saved config: %v", err)
	}

	binDir := t.TempDir()
	writeFakeBinary(t, binDir, "droid")
	t.Setenv("PATH", binDir)

	runner := &launcherSingleRunner{}
	withIntegrationOverride(t, "droid", runner)

	var prompt string
	DefaultConfirmPrompt = func(p string, options ConfirmOptions) (bool, error) {
		prompt = p
		return false, nil
	}

	var selectorCalls int
	var selectorCurrent string
	DefaultSingleSelector = func(title string, items []SelectionItem, current string) (string, error) {
		selectorCalls++
		selectorCurrent = current
		itemNames := selectionItemNames(items)
		if slices.Contains(itemNames, "llama3.2") {
			t.Fatalf("expected saved deprecated model to be hidden from picker, got %v", itemNames)
		}
		if !slices.Contains(itemNames, "best-local") {
			t.Fatalf("expected replacement model to remain selectable, got %v", itemNames)
		}
		return "best-local", nil
	}

	var showCalls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[`+
				`{"model":"best-cloud:cloud","description":"Cloud rec","context_length":262144,"max_output_tokens":32768},`+
				`{"model":"best-local","description":"Local rec"}`+
				`]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"llama3.2"},{"name":"best-local"}]}`)
		case "/api/show":
			showCalls.Add(1)
			fmt.Fprint(w, `{"model_info":{"general.context_length":131072}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{Name: "droid"}); err != nil {
		t.Fatalf("LaunchIntegration returned error: %v", err)
	}
	if !strings.Contains(prompt, "llama3.2 does not work well with StubSingle") {
		t.Fatalf("expected saved deprecated model prompt before picker, got %q", prompt)
	}
	if selectorCalls != 1 {
		t.Fatalf("expected picker to open after declining saved deprecated model, got %d calls", selectorCalls)
	}
	if selectorCurrent != "llama3.2" {
		t.Fatalf("expected saved deprecated model as picker current value, got %q", selectorCurrent)
	}
	if showCalls.Load() != 1 {
		t.Fatalf("expected only replacement model readiness to call /api/show, got %d calls", showCalls.Load())
	}
	if runner.ranModel != "best-local" {
		t.Fatalf("expected integration to run with replacement model, got %q", runner.ranModel)
	}
	saved, err := config.LoadIntegration("droid")
	if err != nil {
		t.Fatalf("failed to reload saved config: %v", err)
	}
	if got := primaryModelFromConfig(saved); got != "best-local" {
		t.Fatalf("expected replacement model to be saved, got %q", got)
	}
}

func TestLaunchIntegration_ModelOverrideDeprecatedConfirmRuns(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)
	withInteractiveSession(t, true)

	binDir := t.TempDir()
	writeFakeBinary(t, binDir, "droid")
	t.Setenv("PATH", binDir)

	runner := &launcherSingleRunner{}
	withIntegrationOverride(t, "droid", runner)

	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		if !strings.Contains(prompt, "qwen2.5-coder:32b does not work well with StubSingle") {
			t.Fatalf("unexpected deprecated model prompt: %q", prompt)
		}
		return true, nil
	}
	var showCalls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[`+
				`{"model":"best-cloud:cloud","description":"Cloud rec","context_length":262144,"max_output_tokens":32768},`+
				`{"model":"best-local","description":"Local rec"}`+
				`]}`)
		case "/api/show":
			showCalls.Add(1)
			fmt.Fprint(w, `{"model_info":{"general.context_length":131072}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{
		Name:          "droid",
		ModelOverride: "qwen2.5-coder:32b",
	})
	if err != nil {
		t.Fatalf("expected deprecated model override confirmation to continue, got %v", err)
	}
	if showCalls.Load() == 0 {
		t.Fatal("expected confirmed deprecated override to continue to /api/show")
	}
	if runner.ranModel != "qwen2.5-coder:32b" {
		t.Fatalf("expected integration to run with confirmed deprecated model, got %q", runner.ranModel)
	}
}

func TestLaunchIntegration_ModelOverrideDeprecatedSuggestsLocalWhenCloudDisabled(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)
	withInteractiveSession(t, true)

	binDir := t.TempDir()
	writeFakeBinary(t, binDir, "droid")
	t.Setenv("PATH", binDir)

	runner := &launcherSingleRunner{}
	withIntegrationOverride(t, "droid", runner)

	var prompt string
	DefaultConfirmPrompt = func(p string, options ConfirmOptions) (bool, error) {
		prompt = p
		return false, nil
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[`+
				`{"model":"best-cloud:cloud","description":"Cloud rec","context_length":262144,"max_output_tokens":32768},`+
				`{"model":"best-local","description":"Local rec"}`+
				`]}`)
		case "/api/status":
			fmt.Fprint(w, `{"cloud":{"disabled":true,"source":"config"}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{
		Name:          "droid",
		ModelOverride: "llama3.2",
	})
	if err == nil {
		t.Fatal("expected deprecated model override to fail")
	}
	if !strings.Contains(prompt, "ollama launch droid --model best-local") {
		t.Fatalf("expected local replacement command when cloud is disabled, got %q", prompt)
	}
	if strings.Contains(prompt, "ollama launch droid --model best-cloud:cloud") {
		t.Fatalf("did not expect cloud replacement command when cloud is disabled, got %q", prompt)
	}
}

func TestLaunchIntegration_ConfigureOnlyPrompt(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	runner := &launcherSingleRunner{}
	withIntegrationOverride(t, "stubsingle", runner)

	DefaultSingleSelector = func(title string, items []SelectionItem, current string) (string, error) {
		return "sample-model", nil
	}

	var prompts []string
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		prompts = append(prompts, prompt)
		if strings.Contains(prompt, "Launch StubSingle now?") {
			return false, nil
		}
		return true, nil
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"sample-model"}]}`)
		case "/api/show":
			fmt.Fprint(w, `{"model":"sample-model"}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{
		Name:           "stubsingle",
		ForceConfigure: true,
		ConfigureOnly:  true,
	}); err != nil {
		t.Fatalf("LaunchIntegration returned error: %v", err)
	}
	if runner.ranModel != "" {
		t.Fatalf("expected configure-only flow to skip launch when prompt is declined, got %q", runner.ranModel)
	}
	if !slices.Contains(prompts, "Launch StubSingle now?") {
		t.Fatalf("expected launch confirmation prompt, got %v", prompts)
	}
}

func TestLaunchIntegration_ModelOverrideHeadlessMissingFailsWithoutPrompt(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)
	withInteractiveSession(t, false)

	binDir := t.TempDir()
	writeFakeBinary(t, binDir, "droid")
	t.Setenv("PATH", binDir)

	runner := &launcherSingleRunner{}
	withIntegrationOverride(t, "droid", runner)

	confirmCalled := false
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		confirmCalled = true
		return true, nil
	}

	pullCalled := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprint(w, `{"error":"model not found"}`)
		case "/api/pull":
			pullCalled = true
			w.WriteHeader(http.StatusOK)
			fmt.Fprint(w, `{"status":"success"}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{
		Name:          "droid",
		ModelOverride: "missing-model",
	})
	if err == nil {
		t.Fatal("expected missing model to fail in headless mode")
	}
	if !strings.Contains(err.Error(), "ollama pull missing-model") {
		t.Fatalf("expected actionable missing model error, got %v", err)
	}
	if confirmCalled {
		t.Fatal("expected no confirmation prompt in headless mode")
	}
	if pullCalled {
		t.Fatal("expected pull request not to run in headless mode")
	}
	if runner.ranModel != "" {
		t.Fatalf("expected launch to abort before running integration, got %q", runner.ranModel)
	}
}

func TestLaunchIntegration_ModelOverrideHeadlessCanOverrideMissingModelPolicy(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)
	withInteractiveSession(t, false)

	binDir := t.TempDir()
	writeFakeBinary(t, binDir, "droid")
	t.Setenv("PATH", binDir)

	runner := &launcherSingleRunner{}
	withIntegrationOverride(t, "droid", runner)

	confirmCalled := false
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		confirmCalled = true
		if !strings.Contains(prompt, "missing-model") {
			t.Fatalf("expected prompt to mention missing model, got %q", prompt)
		}
		return true, nil
	}

	pullCalled := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprint(w, `{"error":"model not found"}`)
		case "/api/pull":
			pullCalled = true
			w.WriteHeader(http.StatusOK)
			fmt.Fprint(w, `{"status":"success"}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	customPolicy := LaunchPolicy{MissingModel: LaunchMissingModelPromptToPull}
	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{
		Name:          "droid",
		ModelOverride: "missing-model",
		Policy:        &customPolicy,
	}); err != nil {
		t.Fatalf("expected policy override to allow prompt/pull in headless mode, got %v", err)
	}
	if !confirmCalled {
		t.Fatal("expected confirmation prompt when missing-model policy is overridden to prompt/pull")
	}
	if !pullCalled {
		t.Fatal("expected pull request to run when missing-model policy is overridden to prompt/pull")
	}
	if runner.ranModel != "missing-model" {
		t.Fatalf("expected integration to launch after pull, got %q", runner.ranModel)
	}
}

func TestLaunchIntegration_ModelOverrideInteractiveMissingPromptsAndPulls(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)
	withInteractiveSession(t, true)

	binDir := t.TempDir()
	writeFakeBinary(t, binDir, "droid")
	t.Setenv("PATH", binDir)

	runner := &launcherSingleRunner{}
	withIntegrationOverride(t, "droid", runner)

	confirmCalled := false
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		confirmCalled = true
		if !strings.Contains(prompt, "missing-model") {
			t.Fatalf("expected prompt to mention missing model, got %q", prompt)
		}
		return true, nil
	}

	pullCalled := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprint(w, `{"error":"model not found"}`)
		case "/api/pull":
			pullCalled = true
			w.WriteHeader(http.StatusOK)
			fmt.Fprint(w, `{"status":"success"}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{
		Name:          "droid",
		ModelOverride: "missing-model",
	})
	if err != nil {
		t.Fatalf("expected interactive override to prompt/pull and succeed, got %v", err)
	}
	if !confirmCalled {
		t.Fatal("expected interactive flow to prompt before pulling missing model")
	}
	if !pullCalled {
		t.Fatal("expected pull request to run after interactive confirmation")
	}
	if runner.ranModel != "missing-model" {
		t.Fatalf("expected integration to run with pulled model, got %q", runner.ranModel)
	}
}

func TestLaunchIntegration_HeadlessSelectorFlowFailsWithoutPrompt(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)
	withInteractiveSession(t, false)

	binDir := t.TempDir()
	writeFakeBinary(t, binDir, "droid")
	t.Setenv("PATH", binDir)

	runner := &launcherSingleRunner{}
	withIntegrationOverride(t, "droid", runner)

	DefaultSingleSelector = func(title string, items []SelectionItem, current string) (string, error) {
		return "missing-model", nil
	}

	confirmCalled := false
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		confirmCalled = true
		return true, nil
	}

	pullCalled := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"sample-model"}]}`)
		case "/api/show":
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprint(w, `{"error":"model not found"}`)
		case "/api/pull":
			pullCalled = true
			w.WriteHeader(http.StatusOK)
			fmt.Fprint(w, `{"status":"success"}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)

	err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{
		Name:           "droid",
		ForceConfigure: true,
	})
	if err == nil {
		t.Fatal("expected headless selector flow to fail on missing model")
	}
	if !strings.Contains(err.Error(), "ollama pull missing-model") {
		t.Fatalf("expected actionable missing model error, got %v", err)
	}
	if confirmCalled {
		t.Fatal("expected no confirmation prompt in headless selector flow")
	}
	if pullCalled {
		t.Fatal("expected no pull request in headless selector flow")
	}
	if runner.ranModel != "" {
		t.Fatalf("expected flow to abort before launch, got %q", runner.ranModel)
	}
}

type apiShowRequest struct {
	Model string `json:"model"`
}

func compareStrings(got, want []string) string {
	if slices.Equal(got, want) {
		return ""
	}
	return fmt.Sprintf("want %v got %v", want, got)
}

func compareStringSlices(got, want [][]string) string {
	if len(got) != len(want) {
		return fmt.Sprintf("want %v got %v", want, got)
	}
	for i := range got {
		if !slices.Equal(got[i], want[i]) {
			return fmt.Sprintf("want %v got %v", want, got)
		}
	}
	return ""
}
