package launch

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

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
	t.Setenv("USERPROFILE", dir)
}

func writeFakeBinary(t *testing.T, dir, name string) {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatalf("failed to write fake binary: %v", err)
	}
}

func withIntegrationOverride(t *testing.T, name string, runner config.Runner) {
	t.Helper()
	restore := config.OverrideIntegration(name, runner)
	t.Cleanup(restore)
}

func withLauncherHooks(t *testing.T) {
	t.Helper()
	oldSingle := config.DefaultSingleSelector
	oldMulti := config.DefaultMultiSelector
	oldConfirm := config.DefaultConfirmPrompt
	oldSignIn := config.DefaultSignIn
	t.Cleanup(func() {
		config.DefaultSingleSelector = oldSingle
		config.DefaultMultiSelector = oldMulti
		config.DefaultConfirmPrompt = oldConfirm
		config.DefaultSignIn = oldSignIn
	})
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

func TestResolveRunModel_UsesSavedModelWithoutSelector(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	if err := config.SetLastModel("llama3.2"); err != nil {
		t.Fatalf("failed to save last model: %v", err)
	}

	selectorCalled := false
	config.DefaultSingleSelector = func(title string, items []config.ModelItem, current string) (string, error) {
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

func TestResolveRunModel_ForcePickerAlwaysUsesSelector(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	if err := config.SetLastModel("llama3.2"); err != nil {
		t.Fatalf("failed to save last model: %v", err)
	}

	var selectorCalls int
	config.DefaultSingleSelector = func(title string, items []config.ModelItem, current string) (string, error) {
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

func TestResolveRunModel_UsesSignInHookForCloudModel(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	config.DefaultSingleSelector = func(title string, items []config.ModelItem, current string) (string, error) {
		return "glm-5:cloud", nil
	}

	signInCalled := false
	config.DefaultSignIn = func(modelName, signInURL string) (string, error) {
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
	config.DefaultMultiSelector = func(title string, items []config.ModelItem, preChecked []string) ([]string, error) {
		multiCalled = true
		return []string{"llama3.2", "qwen3:8b"}, nil
	}

	var proceedPrompt bool
	config.DefaultConfirmPrompt = func(prompt string) (bool, error) {
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
	config.DefaultMultiSelector = func(title string, items []config.ModelItem, preChecked []string) ([]string, error) {
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

func TestLaunchIntegration_ClaudeSyncsAliasesAndSavesPrimary(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)

	binDir := t.TempDir()
	writeFakeBinary(t, binDir, "claude")
	t.Setenv("PATH", binDir)

	var aliasTargets []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			fmt.Fprint(w, `{"models":[]}`)
		case "/api/status":
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprint(w, `{"error":"not found"}`)
		case "/api/show":
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprint(w, `{"error":"not found"}`)
		case "/api/me":
			fmt.Fprint(w, `{"name":"test-user"}`)
		case "/api/experimental/aliases":
			if r.Method != http.MethodPost {
				t.Fatalf("expected alias sync to use POST, got %s", r.Method)
			}
			var req map[string]any
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Fatalf("failed to decode alias request: %v", err)
			}
			aliasTargets = append(aliasTargets, fmt.Sprintf("%s=%s", req["alias"], req["target"]))
			fmt.Fprint(w, `{}`)
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
	if saved.Aliases["primary"] != "glm-5:cloud" {
		t.Fatalf("expected primary alias to match saved model, got %q", saved.Aliases["primary"])
	}
	if saved.Aliases["fast"] != "glm-5:cloud" {
		t.Fatalf("expected fast alias to match cloud primary, got %q", saved.Aliases["fast"])
	}
	slices.Sort(aliasTargets)
	wantTargets := []string{
		"claude-sonnet-=glm-5:cloud",
		"claude-haiku-=glm-5:cloud",
	}
	slices.Sort(wantTargets)
	if !slices.Equal(aliasTargets, wantTargets) {
		t.Fatalf("unexpected synced aliases: %v", aliasTargets)
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
	if err := config.SaveAliases("claude", map[string]string{"primary": "qwen3:8b"}); err != nil {
		t.Fatalf("failed to seed aliases: %v", err)
	}

	var selectorCalls int
	config.DefaultSingleSelector = func(title string, items []config.ModelItem, current string) (string, error) {
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
		case "/api/experimental/aliases":
			fmt.Fprint(w, `{}`)
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
		t.Fatalf("expected forced configure to reprompt for alias primary, got %d calls", selectorCalls)
	}
	saved, err := config.LoadIntegration("claude")
	if err != nil {
		t.Fatalf("failed to reload saved config: %v", err)
	}
	if saved.Models[0] != "glm-5:cloud" {
		t.Fatalf("expected saved primary to be replaced, got %q", saved.Models[0])
	}
}

func TestLaunchIntegration_ConfigureOnlyPrompt(t *testing.T) {
	tmpDir := t.TempDir()
	setLaunchTestHome(t, tmpDir)
	withLauncherHooks(t)

	runner := &launcherSingleRunner{}
	withIntegrationOverride(t, "stubsingle", runner)

	config.DefaultSingleSelector = func(title string, items []config.ModelItem, current string) (string, error) {
		return "llama3.2", nil
	}

	var prompts []string
	config.DefaultConfirmPrompt = func(prompt string) (bool, error) {
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
