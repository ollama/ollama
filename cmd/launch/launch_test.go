package launch

import (
	"bytes"
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
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/config"
)

type launcherEditorRunner struct {
	paths    []string
	edited   [][]string
	ranModel string
}

func (r *launcherEditorRunner) Run(model string, args []string) error {
	r.ranModel = model
	return nil
}

func (r *launcherEditorRunner) String() string { return "LauncherEditor" }

func (r *launcherEditorRunner) Paths() []string { return r.paths }

func (r *launcherEditorRunner) Edit(models []string) error {
	r.edited = append(r.edited, append([]string(nil), models...))
	return nil
}

func (r *launcherEditorRunner) Models() []string { return nil }

type launcherSingleRunner struct {
	ranModel string
}

func (r *launcherSingleRunner) Run(model string, args []string) error {
	r.ranModel = model
	return nil
}

func (r *launcherSingleRunner) String() string { return "StubSingle" }

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
	oldMulti := DefaultMultiSelector
	oldConfirm := DefaultConfirmPrompt
	oldSignIn := DefaultSignIn
	t.Cleanup(func() {
		DefaultSingleSelector = oldSingle
		DefaultMultiSelector = oldMulti
		DefaultConfirmPrompt = oldConfirm
		DefaultSignIn = oldSignIn
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
	if err := config.SaveIntegration("opencode", []string{"glm-5:cloud", "llama3.2"}); err != nil {
		t.Fatalf("failed to save opencode config: %v", err)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"llama3.2"}]}`)
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
	if state.Integrations["opencode"].CurrentModel != "llama3.2" {
		t.Fatalf("expected editor current model to fall back to remaining local model, got %q", state.Integrations["opencode"].CurrentModel)
	}
}

func TestBuildLauncherState_MigratesLegacyOpenclawAliasConfig(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)

	if err := config.SaveIntegration("clawdbot", []string{"llama3.2"}); err != nil {
		t.Fatalf("failed to seed legacy alias config: %v", err)
	}
	if err := config.SaveAliases("clawdbot", map[string]string{"primary": "llama3.2"}); err != nil {
		t.Fatalf("failed to seed legacy alias map: %v", err)
	}
	if err := config.MarkIntegrationOnboarded("clawdbot"); err != nil {
		t.Fatalf("failed to seed legacy onboarding state: %v", err)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"llama3.2"}]}`)
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
	if state.Integrations["openclaw"].CurrentModel != "llama3.2" {
		t.Fatalf("expected openclaw state to reuse legacy alias config, got %q", state.Integrations["openclaw"].CurrentModel)
	}

	migrated, err := config.LoadIntegration("openclaw")
	if err != nil {
		t.Fatalf("expected canonical config to be migrated, got %v", err)
	}
	if !slices.Equal(migrated.Models, []string{"llama3.2"}) {
		t.Fatalf("unexpected migrated models: %v", migrated.Models)
	}
	if migrated.Aliases["primary"] != "llama3.2" {
		t.Fatalf("expected aliases to migrate, got %v", migrated.Aliases)
	}
	if !migrated.Onboarded {
		t.Fatal("expected onboarding state to migrate to canonical openclaw key")
	}
}

func TestBuildLauncherState_ToleratesInventoryFailure(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)

	if err := config.SetLastModel("llama3.2"); err != nil {
		t.Fatalf("failed to seed last model: %v", err)
	}
	if err := config.SaveIntegration("claude", []string{"qwen3:8b"}); err != nil {
		t.Fatalf("failed to seed claude config: %v", err)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
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

func TestResolveRunModel_UsesSavedModelWithoutSelector(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	if err := config.SetLastModel("llama3.2"); err != nil {
		t.Fatalf("failed to save last model: %v", err)
	}

	selectorCalled := false
	DefaultSingleSelector = func(title string, items []ModelItem, current string) (string, error) {
		selectorCalled = true
		return "", nil
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"llama3.2"}]}`)
		case "/api/show":
			fmt.Fprint(w, `{"model":"llama3.2"}`)
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
	if model != "llama3.2" {
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

	DefaultSingleSelector = func(title string, items []ModelItem, current string) (string, error) {
		t.Fatal("selector should not be called in headless --yes mode")
		return "", nil
	}

	restoreConfirm := withLaunchConfirmPolicy(launchConfirmPolicy{yes: true})
	defer restoreConfirm()

	pullCalled := false
	modelPulled := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"llama3.2"}]}`)
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

	DefaultSingleSelector = func(title string, items []ModelItem, current string) (string, error) {
		t.Fatal("selector should not be called when request policy enables headless auto-pick")
		return "", nil
	}

	pullCalled := false
	modelPulled := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"llama3.2"}]}`)
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

	if err := config.SetLastModel("llama3.2"); err != nil {
		t.Fatalf("failed to save last model: %v", err)
	}

	var selectorCalls int
	DefaultSingleSelector = func(title string, items []ModelItem, current string) (string, error) {
		selectorCalls++
		if current != "llama3.2" {
			t.Fatalf("expected current selection to be last model, got %q", current)
		}
		return "qwen3:8b", nil
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"llama3.2"},{"name":"qwen3:8b"}]}`)
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
	DefaultSingleSelector = func(title string, items []ModelItem, current string) (string, error) {
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
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"qwen3.5"},{"name":"glm-4.7-flash"}]}`)
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

	glmIdx := slices.Index(gotNames, "glm-4.7-flash")
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

	DefaultSingleSelector = func(title string, items []ModelItem, current string) (string, error) {
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

func TestLaunchIntegration_EditorForceConfigure(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	binDir := t.TempDir()
	writeFakeBinary(t, binDir, "droid")
	t.Setenv("PATH", binDir)

	editor := &launcherEditorRunner{paths: []string{"/tmp/settings.json"}}
	withIntegrationOverride(t, "droid", editor)

	var multiCalled bool
	DefaultMultiSelector = func(title string, items []ModelItem, preChecked []string) ([]string, error) {
		multiCalled = true
		return []string{"llama3.2", "qwen3:8b"}, nil
	}

	var proceedPrompt bool
	DefaultConfirmPrompt = func(prompt string) (bool, error) {
		if prompt == "Proceed?" {
			proceedPrompt = true
		}
		return true, nil
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"llama3.2"},{"name":"qwen3:8b"}]}`)
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
	if !proceedPrompt {
		t.Fatal("expected backup warning confirmation before edit")
	}
	if diff := compareStringSlices(editor.edited, [][]string{{"llama3.2", "qwen3:8b"}}); diff != "" {
		t.Fatalf("unexpected edited models (-want +got):\n%s", diff)
	}
	if editor.ranModel != "llama3.2" {
		t.Fatalf("expected launch to use first selected model, got %q", editor.ranModel)
	}
	saved, err := config.LoadIntegration("droid")
	if err != nil {
		t.Fatalf("failed to reload saved config: %v", err)
	}
	if diff := compareStrings(saved.Models, []string{"llama3.2", "qwen3:8b"}); diff != "" {
		t.Fatalf("unexpected saved models (-want +got):\n%s", diff)
	}
}

func TestLaunchIntegration_EditorForceConfigure_DoesNotFloatCheckedModelsInPicker(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	binDir := t.TempDir()
	writeFakeBinary(t, binDir, "droid")
	t.Setenv("PATH", binDir)

	editor := &launcherEditorRunner{}
	withIntegrationOverride(t, "droid", editor)

	if err := config.SaveIntegration("droid", []string{"qwen3.5:cloud", "qwen3.5"}); err != nil {
		t.Fatalf("failed to seed config: %v", err)
	}

	var gotItems []string
	var gotPreChecked []string
	DefaultMultiSelector = func(title string, items []ModelItem, preChecked []string) ([]string, error) {
		for _, item := range items {
			gotItems = append(gotItems, item.Name)
		}
		gotPreChecked = append([]string(nil), preChecked...)
		return []string{"qwen3.5:cloud", "qwen3.5"}, nil
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
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
	if gotItems[0] != "kimi-k2.5:cloud" {
		t.Fatalf("expected stable recommendation order with kimi-k2.5:cloud first, got %v", gotItems)
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

	if err := config.SaveIntegration("droid", []string{"llama3.2", "mistral"}); err != nil {
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

	want := []string{"qwen3:8b", "llama3.2", "mistral"}
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
	DefaultMultiSelector = func(title string, items []ModelItem, preChecked []string) ([]string, error) {
		multiCalled = true
		return []string{"llama3.2"}, nil
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/status":
			fmt.Fprint(w, `{"cloud":{"disabled":true,"source":"config"}}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"llama3.2"}]}`)
		case "/api/show":
			fmt.Fprint(w, `{"model":"llama3.2"}`)
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

func TestLaunchIntegration_ConfiguredEditorLaunchSkipsReconfigure(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	binDir := t.TempDir()
	writeFakeBinary(t, binDir, "droid")
	t.Setenv("PATH", binDir)

	editor := &launcherEditorRunner{paths: []string{"/tmp/settings.json"}}
	withIntegrationOverride(t, "droid", editor)

	if err := config.SaveIntegration("droid", []string{"llama3.2", "qwen3:8b"}); err != nil {
		t.Fatalf("failed to seed config: %v", err)
	}

	DefaultConfirmPrompt = func(prompt string) (bool, error) {
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
	if editor.ranModel != "llama3.2" {
		t.Fatalf("expected launch to use saved primary model, got %q", editor.ranModel)
	}

	saved, err := config.LoadIntegration("droid")
	if err != nil {
		t.Fatalf("failed to reload saved config: %v", err)
	}
	if diff := compareStrings(saved.Models, []string{"llama3.2", "qwen3:8b"}); diff != "" {
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

	editor := &launcherEditorRunner{}
	withIntegrationOverride(t, "openclaw", editor)

	if err := config.SaveIntegration("openclaw", []string{"llama3.2", "mistral"}); err != nil {
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
	if editor.ranModel != "llama3.2" {
		t.Fatalf("expected launch to use first saved model, got %q", editor.ranModel)
	}

	saved, err := config.LoadIntegration("openclaw")
	if err != nil {
		t.Fatalf("failed to reload saved config: %v", err)
	}
	if diff := compareStrings(saved.Models, []string{"llama3.2", "mistral"}); diff != "" {
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
	DefaultMultiSelector = func(title string, items []ModelItem, preChecked []string) ([]string, error) {
		selectorCalled = true
		return []string{"llama3.2"}, nil
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

func TestLaunchIntegration_ConfigureOnlyDoesNotRequireInstalledBinary(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)
	t.Setenv("PATH", t.TempDir())

	editor := &launcherEditorRunner{paths: []string{"/tmp/settings.json"}}
	withIntegrationOverride(t, "droid", editor)

	DefaultMultiSelector = func(title string, items []ModelItem, preChecked []string) ([]string, error) {
		return []string{"llama3.2"}, nil
	}

	var prompts []string
	DefaultConfirmPrompt = func(prompt string) (bool, error) {
		prompts = append(prompts, prompt)
		if strings.Contains(prompt, "Launch LauncherEditor now?") {
			return false, nil
		}
		return true, nil
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"llama3.2"}]}`)
		case "/api/show":
			fmt.Fprint(w, `{"model":"llama3.2"}`)
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
	if diff := compareStringSlices(editor.edited, [][]string{{"llama3.2"}}); diff != "" {
		t.Fatalf("unexpected edited models (-want +got):\n%s", diff)
	}
	if editor.ranModel != "" {
		t.Fatalf("expected configure-only flow to skip launch, got %q", editor.ranModel)
	}
	if !slices.Contains(prompts, "Proceed?") {
		t.Fatalf("expected editor warning prompt, got %v", prompts)
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
	DefaultSingleSelector = func(title string, items []ModelItem, current string) (string, error) {
		selectorCalls++
		return "glm-5:cloud", nil
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
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

func TestLaunchIntegration_ClaudeModelOverrideSkipsSelector(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)
	withInteractiveSession(t, true)

	binDir := t.TempDir()
	writeFakeBinary(t, binDir, "claude")
	t.Setenv("PATH", binDir)

	var selectorCalls int
	DefaultSingleSelector = func(title string, items []ModelItem, current string) (string, error) {
		selectorCalls++
		return "", fmt.Errorf("selector should not run when --model override is set")
	}

	var confirmCalls int
	DefaultConfirmPrompt = func(prompt string) (bool, error) {
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

func TestLaunchIntegration_ConfigureOnlyPrompt(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	runner := &launcherSingleRunner{}
	withIntegrationOverride(t, "stubsingle", runner)

	DefaultSingleSelector = func(title string, items []ModelItem, current string) (string, error) {
		return "llama3.2", nil
	}

	var prompts []string
	DefaultConfirmPrompt = func(prompt string) (bool, error) {
		prompts = append(prompts, prompt)
		if strings.Contains(prompt, "Launch StubSingle now?") {
			return false, nil
		}
		return true, nil
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"llama3.2"}]}`)
		case "/api/show":
			fmt.Fprint(w, `{"model":"llama3.2"}`)
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
	DefaultConfirmPrompt = func(prompt string) (bool, error) {
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
	DefaultConfirmPrompt = func(prompt string) (bool, error) {
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
	DefaultConfirmPrompt = func(prompt string) (bool, error) {
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

	DefaultSingleSelector = func(title string, items []ModelItem, current string) (string, error) {
		return "missing-model", nil
	}

	confirmCalled := false
	DefaultConfirmPrompt = func(prompt string) (bool, error) {
		confirmCalled = true
		return true, nil
	}

	pullCalled := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"llama3.2"}]}`)
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

func TestConfirmLowContextLength(t *testing.T) {
	tests := []struct {
		name          string
		models        []string
		statusBody    string
		statusCode    int
		showParams    string // Parameters field returned by /api/show
		wantWarning   bool
		wantModelfile bool // true if warning should mention Modelfile
	}{
		{
			name:       "no warning when server context meets recommended",
			models:     []string{"llama3.2"},
			statusBody: `{"cloud":{},"context_length":65536}`,
			statusCode: http.StatusOK,
		},
		{
			name:       "no warning when server context exceeds recommended",
			models:     []string{"llama3.2"},
			statusBody: `{"cloud":{},"context_length":131072}`,
			statusCode: http.StatusOK,
		},
		{
			name:        "warns when server context is below recommended",
			models:      []string{"llama3.2"},
			statusBody:  `{"cloud":{},"context_length":4096}`,
			statusCode:  http.StatusOK,
			wantWarning: true,
		},
		{
			name:       "no warning when status endpoint fails",
			models:     []string{"llama3.2"},
			statusCode: http.StatusInternalServerError,
		},
		{
			name:       "no warning for cloud-only models even with low context",
			models:     []string{"gpt-4o:cloud"},
			statusBody: `{"cloud":{},"context_length":4096}`,
			statusCode: http.StatusOK,
		},
		{
			name:       "no warning when models list is empty",
			models:     []string{},
			statusBody: `{"cloud":{},"context_length":4096}`,
			statusCode: http.StatusOK,
		},
		{
			name:       "no warning when modelfile num_ctx meets recommended",
			models:     []string{"llama3.2"},
			statusBody: `{"cloud":{},"context_length":4096}`,
			statusCode: http.StatusOK,
			showParams: "num_ctx                        65536",
		},
		{
			name:       "no warning when modelfile num_ctx exceeds recommended",
			models:     []string{"llama3.2"},
			statusBody: `{"cloud":{},"context_length":4096}`,
			statusCode: http.StatusOK,
			showParams: "num_ctx                        131072",
		},
		{
			name:          "warns with modelfile hint when modelfile num_ctx is below recommended",
			models:        []string{"llama3.2"},
			statusBody:    `{"cloud":{},"context_length":131072}`,
			statusCode:    http.StatusOK,
			showParams:    "num_ctx                        4096",
			wantWarning:   true,
			wantModelfile: true,
		},
		{
			name:          "warns with modelfile hint when both server and modelfile are low",
			models:        []string{"llama3.2"},
			statusBody:    `{"cloud":{},"context_length":2048}`,
			statusCode:    http.StatusOK,
			showParams:    "num_ctx                        4096",
			wantWarning:   true,
			wantModelfile: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/api/status" {
					w.WriteHeader(tt.statusCode)
					if tt.statusBody != "" {
						fmt.Fprint(w, tt.statusBody)
					}
					return
				}
				if r.URL.Path == "/api/show" {
					w.WriteHeader(http.StatusOK)
					fmt.Fprintf(w, `{"parameters":%q}`, tt.showParams)
					return
				}
				http.NotFound(w, r)
			}))
			defer srv.Close()
			t.Setenv("OLLAMA_HOST", srv.URL)

			client, err := newTestClient(srv.URL)
			if err != nil {
				t.Fatalf("failed to create client: %v", err)
			}

			// capture stderr
			oldStderr := os.Stderr
			r, w, _ := os.Pipe()
			os.Stderr = w

			err = confirmLowContextLength(context.Background(), client, tt.models)

			w.Close()
			var buf bytes.Buffer
			buf.ReadFrom(r)
			os.Stderr = oldStderr
			output := buf.String()

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			hasWarning := strings.Contains(output, "Warning:")
			if hasWarning != tt.wantWarning {
				t.Fatalf("expected warning=%v, got output: %q", tt.wantWarning, output)
			}
			if tt.wantWarning && tt.wantModelfile {
				if !strings.Contains(output, "official model") {
					t.Fatalf("expected official model hint in output: %q", output)
				}
			}
			if tt.wantWarning && !tt.wantModelfile {
				if strings.Contains(output, "official model") {
					t.Fatalf("expected server hint, not official model hint in output: %q", output)
				}
			}
		})
	}
}

func TestParseNumCtxFromParameters(t *testing.T) {
	tests := []struct {
		name       string
		parameters string
		want       int
	}{
		{
			name:       "extracts num_ctx",
			parameters: "num_ctx                        65536",
			want:       65536,
		},
		{
			name:       "extracts num_ctx among other parameters",
			parameters: "temperature                    0.7\nnum_ctx                        131072\nstop                           \"<|end|>\"",
			want:       131072,
		},
		{
			name:       "returns zero when no num_ctx",
			parameters: "temperature                    0.7\nstop                           \"<|end|>\"",
			want:       0,
		},
		{
			name:       "returns zero for empty string",
			parameters: "",
			want:       0,
		},
		{
			name:       "handles float representation",
			parameters: "num_ctx                        65536.0",
			want:       65536,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := parseNumCtxFromParameters(tt.parameters)
			if got != tt.want {
				t.Fatalf("parseNumCtxFromParameters(%q) = %d, want %d", tt.parameters, got, tt.want)
			}
		})
	}
}

func newTestClient(url string) (*api.Client, error) {
	return api.ClientFromEnvironment()
}
