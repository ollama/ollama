//go:build darwin

package main

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/cmd/launch"
)

type fakeCodexDesktopController struct {
	installed  bool
	running    bool
	launched   string
	models     []string
	stopped    bool
	onboarded  bool
	launchErrs []error
	launches   [][]string
}

func (f *fakeCodexDesktopController) Installed() bool { return f.installed }
func (f *fakeCodexDesktopController) OllamaProfileRunning() bool {
	return f.running
}
func (f *fakeCodexDesktopController) LaunchOllamaProfileFromDesktop(primary string, models []launch.LaunchModel) error {
	names := codexDesktopModelNames(models)
	f.launches = append(f.launches, names)
	if len(f.launchErrs) > 0 {
		err := f.launchErrs[0]
		f.launchErrs = f.launchErrs[1:]
		if err != nil {
			return err
		}
	}
	f.launched = primary
	f.models = names
	f.running = true
	return nil
}

func stubCodexDesktopModelLoader(t *testing.T) {
	originalLoadModels := codexDesktopLoadModels
	originalLoadConnectionModels := codexDesktopLoadConnectionModels
	t.Cleanup(func() {
		codexDesktopLoadModels = originalLoadModels
		codexDesktopLoadConnectionModels = originalLoadConnectionModels
	})
	loader := func(_ context.Context, selected []string) (string, []launch.LaunchModel, error) {
		if len(selected) == 0 {
			selected = []string{"qwen3:8b", "glm-5.2:cloud"}
		}
		models := make([]launch.LaunchModel, 0, len(selected))
		for _, name := range selected {
			models = append(models, launch.LaunchModel{Name: name})
		}
		return selected[0], models, nil
	}
	codexDesktopLoadModels = loader
	codexDesktopLoadConnectionModels = loader
}
func (f *fakeCodexDesktopController) StopOllamaProfileFromDesktop() error {
	f.stopped = true
	f.running = false
	return nil
}
func (f *fakeCodexDesktopController) Onboard() error {
	f.onboarded = true
	return nil
}

func TestSetCodexDesktopConnectionControlsOnlyOllamaProfile(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	originalController := codexDesktop
	t.Cleanup(func() {
		codexDesktop = originalController
	})
	stubCodexDesktopModelLoader(t)

	fake := &fakeCodexDesktopController{installed: true}
	codexDesktop = fake
	if err := setCodexDesktopConnection(true); err != nil {
		t.Fatal(err)
	}
	if fake.launched != "qwen3:8b" || !fake.running || !fake.onboarded {
		t.Fatalf("controller = %#v, want isolated profile launched and onboarded", fake)
	}
	saved, err := config.LoadIntegration(codexDesktopIntegrationName)
	if err != nil {
		t.Fatal(err)
	}
	if len(saved.Models) != 2 || saved.Models[0] != "qwen3:8b" || saved.Models[1] != "glm-5.2:cloud" {
		t.Fatalf("saved integration = %#v, want selected model", saved)
	}

	if err := setCodexDesktopConnection(false); err != nil {
		t.Fatal(err)
	}
	if !fake.stopped || fake.running {
		t.Fatalf("controller = %#v, want only Ollama profile stopped", fake)
	}
}

func TestApplyCodexDesktopModelsRestartsOnlyIsolatedProfile(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	originalController := codexDesktop
	t.Cleanup(func() { codexDesktop = originalController })
	stubCodexDesktopModelLoader(t)
	if err := config.SaveIntegration(codexDesktopIntegrationName, []string{"old-model"}); err != nil {
		t.Fatal(err)
	}

	fake := &fakeCodexDesktopController{installed: true, running: true}
	codexDesktop = fake
	if err := applyCodexDesktopModels([]string{"qwen3:8b", "glm-5.2:cloud"}); err != nil {
		t.Fatal(err)
	}
	if !fake.stopped || !fake.running {
		t.Fatalf("controller = %#v, want isolated profile restarted", fake)
	}
	if len(fake.models) != 2 || fake.models[0] != "qwen3:8b" || fake.models[1] != "glm-5.2:cloud" {
		t.Fatalf("launched models = %#v", fake.models)
	}
	saved := config.IntegrationModels(codexDesktopIntegrationName)
	if len(saved) != 2 || saved[0] != "qwen3:8b" || saved[1] != "glm-5.2:cloud" {
		t.Fatalf("saved models = %#v", saved)
	}
}

func TestApplyCodexDesktopModelsStartsStoppedIsolatedProfile(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	originalController := codexDesktop
	t.Cleanup(func() { codexDesktop = originalController })
	stubCodexDesktopModelLoader(t)

	fake := &fakeCodexDesktopController{installed: true}
	codexDesktop = fake
	if err := applyCodexDesktopModels([]string{"qwen3:8b", "glm-5.2:cloud"}); err != nil {
		t.Fatal(err)
	}
	if fake.stopped || !fake.running {
		t.Fatalf("controller = %#v, want stopped profile started without a stop", fake)
	}
	if len(fake.models) != 2 || fake.models[0] != "qwen3:8b" || fake.models[1] != "glm-5.2:cloud" {
		t.Fatalf("launched models = %#v", fake.models)
	}
}

func TestApplyCodexDesktopModelsRestoresSelectionAfterStoppedLaunchFailure(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	originalController := codexDesktop
	t.Cleanup(func() { codexDesktop = originalController })
	stubCodexDesktopModelLoader(t)
	if err := config.SaveIntegration(codexDesktopIntegrationName, []string{"old-model"}); err != nil {
		t.Fatal(err)
	}

	fake := &fakeCodexDesktopController{
		installed:  true,
		launchErrs: []error{errors.New("start failed")},
	}
	codexDesktop = fake
	if err := applyCodexDesktopModels([]string{"new-model"}); err == nil {
		t.Fatal("applyCodexDesktopModels returned nil error after start failure")
	}
	if fake.running || fake.stopped || len(fake.launches) != 1 {
		t.Fatalf("controller = %#v, want one failed start and no stop", fake)
	}
	saved := config.IntegrationModels(codexDesktopIntegrationName)
	if len(saved) != 1 || saved[0] != "old-model" {
		t.Fatalf("saved models = %#v, want previous selection", saved)
	}
}

func TestApplyCodexDesktopModelsRestoresPreviousProfileAfterLaunchFailure(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	originalController := codexDesktop
	t.Cleanup(func() { codexDesktop = originalController })
	stubCodexDesktopModelLoader(t)
	if err := config.SaveIntegration(codexDesktopIntegrationName, []string{"old-model"}); err != nil {
		t.Fatal(err)
	}

	fake := &fakeCodexDesktopController{
		installed:  true,
		running:    true,
		launchErrs: []error{errors.New("new profile failed"), nil},
	}
	codexDesktop = fake
	if err := applyCodexDesktopModels([]string{"new-model"}); err == nil {
		t.Fatal("applyCodexDesktopModels returned nil error after launch failure")
	}
	if !fake.running || fake.launched != "old-model" {
		t.Fatalf("controller = %#v, want previous profile restored", fake)
	}
	saved := config.IntegrationModels(codexDesktopIntegrationName)
	if len(saved) != 1 || saved[0] != "old-model" {
		t.Fatalf("saved models = %#v, want previous selection", saved)
	}
}

func TestBuildCodexDesktopModelsUsesAvailableRecommendations(t *testing.T) {
	recommendations := []api.ModelRecommendation{
		{Model: "missing-local", ContextLength: 4096},
		{Model: "glm-5.2:cloud", ContextLength: 131072, MaxOutputTokens: 32768},
		{Model: "qwen3:8b", ContextLength: 8192},
	}
	listed := []api.ListModelResponse{
		{Name: "qwen3:8b", Size: 42, Details: api.ModelDetails{ContextLength: 32768}},
		{Name: "llama3.2:latest"},
	}

	primary, models, err := buildCodexDesktopModels(
		[]string{"llama3.2", "glm-5.2:cloud", "qwen3:8b"},
		recommendations,
		listed,
		[]string{"glm-5.2:cloud"},
	)
	if err != nil {
		t.Fatal(err)
	}
	if primary != "llama3.2:latest" {
		t.Fatalf("primary = %q, want llama3.2:latest", primary)
	}
	if len(models) != 3 {
		t.Fatalf("models = %#v, want cloud and two installed models", models)
	}
	if models[0].Name != "llama3.2:latest" {
		t.Fatalf("first model = %q, want preferred installed model", models[0].Name)
	}
	byName := make(map[string]bool, len(models))
	for _, model := range models {
		byName[model.Name] = model.Remote
		if model.Name == "qwen3:8b" && model.ContextLength != 32768 {
			t.Fatalf("installed model = %#v, want local inventory metadata", model)
		}
	}
	if !byName["glm-5.2:cloud"] {
		t.Fatalf("models = %#v, want remote cloud recommendation", models)
	}
}

func TestBuildCodexDesktopModelsRejectsEmptyCatalog(t *testing.T) {
	if _, _, err := buildCodexDesktopModels(nil, nil, nil, nil); err == nil {
		t.Fatal("buildCodexDesktopModels returned nil error for empty catalog")
	}
}

func TestBuildCodexDesktopModelsRejectsUnavailableLocalSelection(t *testing.T) {
	if _, _, err := buildCodexDesktopModels([]string{"missing-local"}, nil, []api.ListModelResponse{{Name: "qwen3:8b"}}, nil); err == nil {
		t.Fatal("buildCodexDesktopModels returned nil error for unavailable selection")
	}
}

func TestLoadCodexDesktopAvailableModelsRetriesEmptyStartupInventory(t *testing.T) {
	tagRequests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/experimental/model-recommendations":
			_, _ = w.Write([]byte(`{"recommendations":[]}`))
		case "/api/tags":
			tagRequests++
			if tagRequests < 3 {
				_, _ = w.Write([]byte(`{"models":[]}`))
				return
			}
			_, _ = w.Write([]byte(`{"models":[{"name":"qwen3:8b"}]}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	base, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	client := api.NewClient(base, server.Client())

	originalClientFactory := codexDesktopClientFactory
	originalCloudModels := codexDesktopCloudModels
	originalAttempts := codexDesktopModelLoadAttempts
	originalRetryWait := codexDesktopModelRetryWait
	t.Cleanup(func() {
		codexDesktopClientFactory = originalClientFactory
		codexDesktopCloudModels = originalCloudModels
		codexDesktopModelLoadAttempts = originalAttempts
		codexDesktopModelRetryWait = originalRetryWait
	})
	codexDesktopClientFactory = func() (*api.Client, error) { return client, nil }
	codexDesktopCloudModels = func(context.Context) ([]string, error) { return nil, nil }
	codexDesktopModelLoadAttempts = 3
	codexDesktopModelRetryWait = time.Millisecond

	models, err := loadCodexDesktopAvailableModels(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if tagRequests != 3 {
		t.Fatalf("tag requests = %d, want 3", tagRequests)
	}
	if got := codexDesktopModelNames(models); len(got) != 1 || got[0] != "qwen3:8b" {
		t.Fatalf("models = %#v, want qwen3:8b", got)
	}
}

func TestReconcileCodexDesktopModelsDropsUnavailableSavedSelections(t *testing.T) {
	available := []launch.LaunchModel{
		{Name: "qwen3:8b"},
		{Name: "glm-5.2:cloud", Remote: true},
	}
	primary, models, err := reconcileCodexDesktopModels(
		[]string{"removed-model", "qwen3:8b", "expired:cloud", "glm-5.2:cloud"},
		available,
	)
	if err != nil {
		t.Fatal(err)
	}
	if primary != "qwen3:8b" {
		t.Fatalf("primary = %q, want qwen3:8b", primary)
	}
	if got := codexDesktopModelNames(models); len(got) != 2 || got[0] != "qwen3:8b" || got[1] != "glm-5.2:cloud" {
		t.Fatalf("models = %#v, want remaining available saved models", got)
	}
}

func TestReconcileCodexDesktopModelsFallsBackWhenAllSavedSelectionsAreUnavailable(t *testing.T) {
	available := []launch.LaunchModel{{Name: "qwen3:8b"}, {Name: "llama3.2:latest"}}
	primary, models, err := reconcileCodexDesktopModels([]string{"removed-model"}, available)
	if err != nil {
		t.Fatal(err)
	}
	if primary != "qwen3:8b" {
		t.Fatalf("primary = %q, want qwen3:8b", primary)
	}
	if got := codexDesktopModelNames(models); len(got) != 2 || got[0] != "qwen3:8b" || got[1] != "llama3.2:latest" {
		t.Fatalf("models = %#v, want current defaults", got)
	}
}

func TestBuildCodexDesktopModelsExcludesRecommendationOnlyCloudModels(t *testing.T) {
	recommendations := []api.ModelRecommendation{
		{Model: "kimi-k2.6:cloud", ContextLength: 262144},
		{Model: "qwen3:8b", ContextLength: 8192},
	}
	listed := []api.ListModelResponse{{Name: "qwen3:8b"}}

	primary, models, err := buildCodexDesktopModels(nil, recommendations, listed, nil)
	if err != nil {
		t.Fatal(err)
	}
	if primary != "qwen3:8b" || len(models) != 1 || models[0].Name != "qwen3:8b" {
		t.Fatalf("models = %q, %#v; want only eligible local model", primary, models)
	}
}

func TestBuildCodexDesktopModelsExcludesUnentitledListedCloudModels(t *testing.T) {
	listed := []api.ListModelResponse{
		{Name: "glm-5.2:cloud", RemoteModel: "glm-5.2"},
		{Name: "qwen3:8b"},
	}

	primary, models, err := buildCodexDesktopModels(nil, nil, listed, nil)
	if err != nil {
		t.Fatal(err)
	}
	if primary != "qwen3:8b" || len(models) != 1 || models[0].Name != "qwen3:8b" {
		t.Fatalf("models = %q, %#v; want cloud manifest excluded without account access", primary, models)
	}
}

func TestBuildCodexDesktopModelsLimitsCatalogToFive(t *testing.T) {
	listed := []api.ListModelResponse{
		{Name: "model-1"},
		{Name: "model-2"},
		{Name: "model-3"},
		{Name: "model-4"},
		{Name: "model-5"},
		{Name: "model-6"},
		{Name: "preferred-model"},
	}

	primary, models, err := buildCodexDesktopModels([]string{"preferred-model", "model-1", "model-2", "model-3", "model-4"}, nil, listed, nil)
	if err != nil {
		t.Fatal(err)
	}
	if primary != "preferred-model" {
		t.Fatalf("primary = %q, want preferred-model", primary)
	}
	if len(models) != codexDesktopMaxModels {
		t.Fatalf("model count = %d, want %d", len(models), codexDesktopMaxModels)
	}
	if models[0].Name != "preferred-model" {
		t.Fatalf("first model = %q, want preferred-model", models[0].Name)
	}
}

func TestBuildCodexDesktopModelsRejectsMoreThanFiveSelections(t *testing.T) {
	selected := []string{"model-1", "model-2", "model-3", "model-4", "model-5", "model-6"}
	listed := make([]api.ListModelResponse, 0, len(selected))
	for _, name := range selected {
		listed = append(listed, api.ListModelResponse{Name: name})
	}
	if _, _, err := buildCodexDesktopModels(selected, nil, listed, nil); err == nil {
		t.Fatal("buildCodexDesktopModels returned nil error for six selections")
	}
}
