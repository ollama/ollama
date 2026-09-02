//go:build darwin

package main

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/cmd/launch"
	modelpkg "github.com/ollama/ollama/types/model"
)

type fakeCodexDesktopController struct {
	installed            bool
	running              bool
	configured           bool
	launched             string
	models               []string
	stopped              bool
	onboarded            bool
	launchErrs           []error
	launches             [][]string
	requests             uint64
	configureBeforeError bool
	shutdownRestores     int
	restarts             int
}

func (f *fakeCodexDesktopController) Installed() bool { return f.installed }
func (f *fakeCodexDesktopController) OllamaConfigured() bool {
	return f.configured
}
func (f *fakeCodexDesktopController) Running() bool { return f.running }
func (f *fakeCodexDesktopController) OllamaRequestCount() uint64 {
	return f.requests
}

func (f *fakeCodexDesktopController) UseOllamaFromDesktop(primary string, models []launch.LaunchModel, restartConfirmed bool) error {
	if f.running && !restartConfirmed {
		return errCodexDesktopRestartConfirmationRequired
	}
	names := codexDesktopModelNames(models)
	f.launches = append(f.launches, names)
	if len(f.launchErrs) > 0 {
		err := f.launchErrs[0]
		f.launchErrs = f.launchErrs[1:]
		if err != nil {
			if f.configureBeforeError {
				f.configured = true
			}
			return err
		}
	}
	f.launched = primary
	f.models = names
	f.running = true
	f.configured = true
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
func (f *fakeCodexDesktopController) RestoreFromDesktop(restartConfirmed bool) error {
	if f.running && !restartConfirmed {
		return errCodexDesktopRestartConfirmationRequired
	}
	f.stopped = true
	f.running = false
	f.configured = false
	return nil
}
func (f *fakeCodexDesktopController) RestartFromDesktop(restartConfirmed bool) error {
	if f.running && !restartConfirmed {
		return errCodexDesktopRestartConfirmationRequired
	}
	f.restarts++
	f.running = true
	return nil
}
func (f *fakeCodexDesktopController) RestoreForShutdown(context.Context) error {
	f.shutdownRestores++
	f.stopped = true
	f.running = false
	f.configured = false
	return nil
}
func (f *fakeCodexDesktopController) Onboard() error {
	f.onboarded = true
	return nil
}

func TestSetCodexDesktopConnectionSwitchesRegularProfile(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	originalController := codexDesktop
	t.Cleanup(func() {
		codexDesktop = originalController
	})
	stubCodexDesktopModelLoader(t)

	fake := &fakeCodexDesktopController{installed: true}
	codexDesktop = fake
	if err := setCodexDesktopConnection(true, false); err != nil {
		t.Fatal(err)
	}
	if fake.launched != "qwen3:8b" || !fake.running || !fake.onboarded {
		t.Fatalf("controller = %#v, want regular profile switched and onboarded", fake)
	}
	saved, err := config.LoadIntegration(codexDesktopIntegrationName)
	if err != nil {
		t.Fatal(err)
	}
	if len(saved.Models) != 2 || saved.Models[0] != "qwen3:8b" || saved.Models[1] != "glm-5.2:cloud" {
		t.Fatalf("saved integration = %#v, want selected model", saved)
	}

	if err := setCodexDesktopConnection(false, true); err != nil {
		t.Fatal(err)
	}
	if !fake.stopped || fake.configured {
		t.Fatalf("controller = %#v, want regular profile restored", fake)
	}
}

func TestSetCodexDesktopConnectionRequiresRestartConfirmation(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	originalController := codexDesktop
	t.Cleanup(func() { codexDesktop = originalController })
	stubCodexDesktopModelLoader(t)

	fake := &fakeCodexDesktopController{installed: true, running: true}
	codexDesktop = fake
	err := setCodexDesktopConnection(true, false)
	if !errors.Is(err, errCodexDesktopRestartConfirmationRequired) {
		t.Fatalf("setCodexDesktopConnection error = %v, want restart confirmation", err)
	}
	if fake.launched != "" || fake.onboarded || fake.configured {
		t.Fatalf("controller changed before restart confirmation: %#v", fake)
	}

	if err := setCodexDesktopConnection(true, true); err != nil {
		t.Fatal(err)
	}
	if fake.launched != "qwen3:8b" || !fake.onboarded || !fake.configured {
		t.Fatalf("controller = %#v, want confirmed profile switch", fake)
	}
}

func TestSetCodexDesktopConnectionIsIdempotent(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	originalController := codexDesktop
	t.Cleanup(func() { codexDesktop = originalController })
	stubCodexDesktopModelLoader(t)

	t.Run("already connected", func(t *testing.T) {
		fake := &fakeCodexDesktopController{installed: true, running: true, configured: true}
		codexDesktop = fake
		if err := setCodexDesktopConnection(true, false); err != nil {
			t.Fatal(err)
		}
		if len(fake.launches) != 0 || fake.onboarded || fake.stopped {
			t.Fatalf("idempotent connect changed controller: %#v", fake)
		}
	})

	t.Run("already disconnected", func(t *testing.T) {
		fake := &fakeCodexDesktopController{installed: true, running: true}
		codexDesktop = fake
		if err := setCodexDesktopConnection(false, false); err != nil {
			t.Fatal(err)
		}
		if len(fake.launches) != 0 || fake.onboarded || fake.stopped {
			t.Fatalf("idempotent disconnect changed controller: %#v", fake)
		}
	})
}

func TestSetCodexDesktopConnectionRestoresProfileAfterPartialSwitchFailure(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	originalController := codexDesktop
	t.Cleanup(func() { codexDesktop = originalController })
	stubCodexDesktopModelLoader(t)
	if err := config.SaveIntegration(codexDesktopIntegrationName, []string{"old-model"}); err != nil {
		t.Fatal(err)
	}

	fake := &fakeCodexDesktopController{
		installed:            true,
		launchErrs:           []error{errors.New("reopen failed")},
		configureBeforeError: true,
	}
	codexDesktop = fake
	if err := setCodexDesktopConnection(true, false); err == nil {
		t.Fatal("setCodexDesktopConnection returned nil after a partial switch failure")
	}
	if !fake.stopped || fake.configured {
		t.Fatalf("controller = %#v, want the regular profile restored", fake)
	}
	if got := config.IntegrationModels(codexDesktopIntegrationName); len(got) != 1 || got[0] != "old-model" {
		t.Fatalf("saved models = %v, want previous selection", got)
	}
}

func TestApplyCodexDesktopModelsRestartsConfiguredProfile(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	originalController := codexDesktop
	t.Cleanup(func() { codexDesktop = originalController })
	stubCodexDesktopModelLoader(t)
	if err := config.SaveIntegration(codexDesktopIntegrationName, []string{"old-model"}); err != nil {
		t.Fatal(err)
	}

	fake := &fakeCodexDesktopController{installed: true, running: true, configured: true}
	codexDesktop = fake
	if err := applyCodexDesktopModels([]string{"qwen3:8b", "glm-5.2:cloud"}, true); err != nil {
		t.Fatal(err)
	}
	if !fake.running || !fake.configured {
		t.Fatalf("controller = %#v, want configured profile restarted", fake)
	}
	if len(fake.models) != 2 || fake.models[0] != "qwen3:8b" || fake.models[1] != "glm-5.2:cloud" {
		t.Fatalf("launched models = %#v", fake.models)
	}
	saved := config.IntegrationModels(codexDesktopIntegrationName)
	if len(saved) != 2 || saved[0] != "qwen3:8b" || saved[1] != "glm-5.2:cloud" {
		t.Fatalf("saved models = %#v", saved)
	}
}

func TestApplyCodexDesktopModelsRequiresLiveRestartConfirmation(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	originalController := codexDesktop
	t.Cleanup(func() { codexDesktop = originalController })
	stubCodexDesktopModelLoader(t)
	if err := config.SaveIntegration(codexDesktopIntegrationName, []string{"old-model"}); err != nil {
		t.Fatal(err)
	}

	fake := &fakeCodexDesktopController{installed: true, running: true, configured: true}
	codexDesktop = fake
	err := applyCodexDesktopModels([]string{"new-model"}, false)
	if !errors.Is(err, errCodexDesktopRestartConfirmationRequired) {
		t.Fatalf("applyCodexDesktopModels error = %v, want restart confirmation", err)
	}
	if len(fake.launches) != 0 {
		t.Fatalf("launches before confirmation = %v, want none", fake.launches)
	}
	if got := config.IntegrationModels(codexDesktopIntegrationName); !slices.Equal(got, []string{"old-model"}) {
		t.Fatalf("saved models before confirmation = %v, want old selection", got)
	}

	if err := applyCodexDesktopModels([]string{"new-model"}, true); err != nil {
		t.Fatal(err)
	}
	if got := config.IntegrationModels(codexDesktopIntegrationName); !slices.Equal(got, []string{"new-model"}) {
		t.Fatalf("saved models after confirmation = %v, want new selection", got)
	}
}

func TestApplyCodexDesktopModelsRestartsUnchangedLiveProfileWhenRequested(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	originalController := codexDesktop
	t.Cleanup(func() { codexDesktop = originalController })
	stubCodexDesktopModelLoader(t)
	if err := config.SaveIntegration(codexDesktopIntegrationName, []string{"same-model"}); err != nil {
		t.Fatal(err)
	}

	fake := &fakeCodexDesktopController{installed: true, running: true, configured: true}
	codexDesktop = fake
	err := applyCodexDesktopModels([]string{"same-model"}, false)
	if !errors.Is(err, errCodexDesktopRestartConfirmationRequired) {
		t.Fatalf("applyCodexDesktopModels error = %v, want restart confirmation", err)
	}
	if len(fake.launches) != 0 || fake.stopped {
		t.Fatalf("unchanged apply restarted ChatGPT before confirmation: %#v", fake)
	}
	if err := applyCodexDesktopModels([]string{"same-model"}, true); err != nil {
		t.Fatal(err)
	}
	if len(fake.launches) != 1 || fake.launched != "same-model" {
		t.Fatalf("confirmed restart did not relaunch ChatGPT: %#v", fake)
	}
}

func TestApplyCodexDesktopModelsCanRestartWhenInventoryIsUnavailable(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	originalController := codexDesktop
	t.Cleanup(func() { codexDesktop = originalController })
	stubCodexDesktopModelLoader(t)
	if err := config.SaveIntegration(codexDesktopIntegrationName, []string{"same-model"}); err != nil {
		t.Fatal(err)
	}
	codexDesktopLoadModels = func(context.Context, []string) (string, []launch.LaunchModel, error) {
		return "", nil, errors.New("inventory unavailable")
	}

	fake := &fakeCodexDesktopController{installed: true, running: true, configured: true}
	codexDesktop = fake
	err := applyCodexDesktopModels([]string{"same-model"}, false)
	if !errors.Is(err, errCodexDesktopRestartConfirmationRequired) {
		t.Fatalf("applyCodexDesktopModels error = %v, want restart confirmation", err)
	}
	if fake.restarts != 0 {
		t.Fatalf("restarts before confirmation = %d, want none", fake.restarts)
	}
	if err := applyCodexDesktopModels([]string{"same-model"}, true); err != nil {
		t.Fatal(err)
	}
	if fake.restarts != 1 || len(fake.launches) != 0 {
		t.Fatalf("controller = %#v, want one catalog-independent restart", fake)
	}
}

func TestRestoreCodexAppForTermination(t *testing.T) {
	originalController := codexDesktop
	t.Cleanup(func() { codexDesktop = originalController })

	tests := []struct {
		name       string
		configured bool
		handoff    bool
		want       int
	}{
		{name: "normal shutdown restores configured profile", configured: true, want: 1},
		{name: "handoff preserves configured profile", configured: true, handoff: true},
		{name: "normal shutdown skips regular profile", configured: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fake := &fakeCodexDesktopController{installed: true, configured: tt.configured}
			codexDesktop = fake
			if err := restoreCodexAppForTermination(context.Background(), tt.handoff); err != nil {
				t.Fatal(err)
			}
			if fake.shutdownRestores != tt.want {
				t.Fatalf("shutdown restores = %d, want %d", fake.shutdownRestores, tt.want)
			}
		})
	}
}

func TestApplyCodexDesktopModelsStartsStoppedRegularProfile(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	originalController := codexDesktop
	t.Cleanup(func() { codexDesktop = originalController })
	stubCodexDesktopModelLoader(t)

	fake := &fakeCodexDesktopController{installed: true}
	codexDesktop = fake
	if err := applyCodexDesktopModels([]string{"qwen3:8b", "glm-5.2:cloud"}, false); err != nil {
		t.Fatal(err)
	}
	if fake.stopped || !fake.running || !fake.configured {
		t.Fatalf("controller = %#v, want stopped regular profile configured and started", fake)
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
	if err := applyCodexDesktopModels([]string{"new-model"}, false); err == nil {
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
		configured: true,
		running:    true,
		launchErrs: []error{errors.New("new profile failed"), nil},
	}
	codexDesktop = fake
	if err := applyCodexDesktopModels([]string{"new-model"}, true); err == nil {
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

func TestApplyCodexDesktopModelsFallsBackToNormalProfileWhenRollbackFails(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	originalController := codexDesktop
	t.Cleanup(func() { codexDesktop = originalController })
	stubCodexDesktopModelLoader(t)
	if err := config.SaveIntegration(codexDesktopIntegrationName, []string{"old-model"}); err != nil {
		t.Fatal(err)
	}

	fake := &fakeCodexDesktopController{
		installed:  true,
		configured: true,
		running:    true,
		launchErrs: []error{errors.New("new profile failed"), errors.New("old profile failed")},
	}
	codexDesktop = fake
	err := applyCodexDesktopModels([]string{"new-model"}, true)
	if err == nil || !strings.Contains(err.Error(), "restored the normal ChatGPT profile") {
		t.Fatalf("applyCodexDesktopModels error = %v, want safe fallback detail", err)
	}
	if !fake.stopped || fake.running || fake.configured {
		t.Fatalf("controller = %#v, want normal stopped ChatGPT profile", fake)
	}
	if got := config.IntegrationModels(codexDesktopIntegrationName); !slices.Equal(got, []string{"old-model"}) {
		t.Fatalf("saved models = %v, want previous selection", got)
	}
}

func TestBuildCodexDesktopModelsUsesListedAndAccountModels(t *testing.T) {
	listed := []api.ListModelResponse{
		{Name: "qwen3:8b", Size: 42, Details: api.ModelDetails{ContextLength: 32768}},
		{Name: "llama3.2:latest"},
	}

	primary, models, err := buildCodexDesktopModels(
		[]string{"llama3.2", "glm-5.2:cloud", "qwen3:8b"},
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
		t.Fatalf("models = %#v, want account-accessible cloud model", models)
	}
}

func TestCodexDesktopDefaultModelsUsesAvailableOrder(t *testing.T) {
	available := codexDesktopAvailableModels(
		[]api.ListModelResponse{{Name: "llama3.1:latest"}, {Name: "extra-local"}},
		[]string{"glm-5.3-flash:cloud", "kimi-k2.7-code:cloud", "extra-cloud:cloud"},
	)

	got := codexDesktopModelNames(codexDesktopDefaultModels(codexDesktopModelInventory{Available: available}))
	want := []string{"llama3.1:latest", "extra-local", "glm-5.3-flash:cloud", "kimi-k2.7-code:cloud", "extra-cloud:cloud"}
	if !slices.Equal(got, want) {
		t.Fatalf("default models = %v, want first available models %v", got, want)
	}
}

func TestGetCodexDesktopModelsSettingsKeepsSelectionWhenInventoryFails(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	originalController := codexDesktop
	originalClientFactory := codexDesktopClientFactory
	t.Cleanup(func() {
		codexDesktop = originalController
		codexDesktopClientFactory = originalClientFactory
	})
	if err := config.SaveIntegration(codexDesktopIntegrationName, []string{"same-model"}); err != nil {
		t.Fatal(err)
	}
	codexDesktop = &fakeCodexDesktopController{installed: true, running: true, configured: true}
	codexDesktopClientFactory = func() (*api.Client, error) {
		return nil, errors.New("inventory unavailable")
	}

	settings, err := getCodexDesktopModelsSettings()
	if err == nil {
		t.Fatal("getCodexDesktopModelsSettings returned nil inventory error")
	}
	if !slices.Equal(settings.Selected, []string{"same-model"}) {
		t.Fatalf("selected models = %v, want saved selection", settings.Selected)
	}
	if !settings.Connected || !settings.Running {
		t.Fatalf("settings = %#v, want live connected state", settings)
	}
}

func TestCodexDesktopDefaultModelsLimitsSelectionToFive(t *testing.T) {
	available := []launch.LaunchModel{
		{Name: "model-1"},
		{Name: "model-2"},
		{Name: "model-3"},
		{Name: "model-4"},
		{Name: "model-5"},
		{Name: "model-6"},
	}
	got := codexDesktopModelNames(codexDesktopDefaultModels(codexDesktopModelInventory{Available: available}))
	if len(got) != codexDesktopMaxModels || got[0] != "model-1" || got[4] != "model-5" {
		t.Fatalf("default models = %v, want first five available models", got)
	}
}

func TestBuildCodexDesktopModelsRejectsEmptyCatalog(t *testing.T) {
	if _, _, err := buildCodexDesktopModels(nil, nil, nil); err == nil {
		t.Fatal("buildCodexDesktopModels returned nil error for empty catalog")
	}
}

func TestBuildCodexDesktopModelsRejectsUnavailableLocalSelection(t *testing.T) {
	if _, _, err := buildCodexDesktopModels([]string{"missing-local"}, []api.ListModelResponse{{Name: "qwen3:8b"}}, nil); err == nil {
		t.Fatal("buildCodexDesktopModels returned nil error for unavailable selection")
	}
}

func TestLoadCodexDesktopAvailableModelsRetriesEmptyStartupInventory(t *testing.T) {
	tagRequests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
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

func TestLoadCodexDesktopModelsHydratesAccountOnlyCloudCapabilities(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			_, _ = w.Write([]byte(`{"models":[]}`))
		case "/api/show":
			_, _ = w.Write([]byte(`{"capabilities":["completion","thinking","tools","vision"],"details":{"family":"glm5_next"}}`))
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
	t.Cleanup(func() {
		codexDesktopClientFactory = originalClientFactory
		codexDesktopCloudModels = originalCloudModels
	})
	codexDesktopClientFactory = func() (*api.Client, error) { return client, nil }
	codexDesktopCloudModels = func(context.Context) ([]string, error) {
		return []string{"glm-5.3-flash:cloud"}, nil
	}

	primary, models, err := loadCodexDesktopModels(context.Background(), []string{"glm-5.3-flash:cloud"})
	if err != nil {
		t.Fatal(err)
	}
	if primary != "glm-5.3-flash:cloud" || len(models) != 1 {
		t.Fatalf("models = %q, %#v; want selected cloud model", primary, models)
	}
	if !models[0].HasCapability(modelpkg.CapabilityThinking) {
		t.Fatalf("capabilities = %v, want thinking from /api/show", models[0].Capabilities)
	}
	if models[0].Details.Family != "glm5_next" {
		t.Fatalf("family = %q, want model metadata from /api/show", models[0].Details.Family)
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
		nil,
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
	primary, models, err := reconcileCodexDesktopModels([]string{"removed-model"}, available, nil)
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

func TestReconcileCodexDesktopModelsUsesProvidedDefaultsWhenSavedSelectionsAreUnavailable(t *testing.T) {
	available := []launch.LaunchModel{
		{Name: "default:cloud", Remote: true},
		{Name: "extra-local"},
	}
	defaults := []launch.LaunchModel{available[0]}
	primary, models, err := reconcileCodexDesktopModels([]string{"removed-model"}, available, defaults)
	if err != nil {
		t.Fatal(err)
	}
	if primary != "default:cloud" {
		t.Fatalf("primary = %q, want provided default", primary)
	}
	if got := codexDesktopModelNames(models); len(got) != 1 || got[0] != "default:cloud" {
		t.Fatalf("models = %v, want only provided defaults", got)
	}
}

func TestBuildCodexDesktopModelsExcludesUnentitledListedCloudModels(t *testing.T) {
	listed := []api.ListModelResponse{
		{Name: "glm-5.2:cloud", RemoteModel: "glm-5.2"},
		{Name: "qwen3:8b"},
	}

	primary, models, err := buildCodexDesktopModels(nil, listed, nil)
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

	primary, models, err := buildCodexDesktopModels([]string{"preferred-model", "model-1", "model-2", "model-3", "model-4"}, listed, nil)
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
	if _, _, err := buildCodexDesktopModels(selected, listed, nil); err == nil {
		t.Fatal("buildCodexDesktopModels returned nil error for six selections")
	}
}

func TestCodexDesktopModelRefreshErrorUsesUserFacingCopy(t *testing.T) {
	withSavedModels := codexDesktopModelRefreshError(codexDesktopModelsSettings{
		Selected: []string{"qwen3:8b"},
	})
	if withSavedModels != "Couldn’t refresh available models. Your saved models are unchanged." {
		t.Fatalf("saved-model message = %q", withSavedModels)
	}

	withoutSavedModels := codexDesktopModelRefreshError(codexDesktopModelsSettings{})
	if withoutSavedModels != "Couldn’t refresh available models. Try again." {
		t.Fatalf("empty-selection message = %q", withoutSavedModels)
	}
}
