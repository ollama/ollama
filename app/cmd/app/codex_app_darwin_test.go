//go:build darwin

package main

import (
	"context"
	"errors"
	"fmt"
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
	"github.com/ollama/ollama/internal/proxy"
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
	updates              [][]string
	requests             uint64
	configureBeforeError bool
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

func (f *fakeCodexDesktopController) UpdateOllamaModelsFromDesktop(primary string, models []launch.LaunchModel, restartConfirmed bool) error {
	if f.running && !restartConfirmed {
		return errCodexDesktopRestartConfirmationRequired
	}
	names := codexDesktopModelNames(models)
	f.updates = append(f.updates, names)
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

func stubCodexDesktopCatalogSources(t *testing.T, recommendations []api.ModelRecommendation, access proxy.ClaudeDesktopAccessState) {
	originalRecommendations := codexDesktopRecommendations
	originalAccessState := codexDesktopAccessState
	t.Cleanup(func() {
		codexDesktopRecommendations = originalRecommendations
		codexDesktopAccessState = originalAccessState
	})
	codexDesktopRecommendations = func(context.Context) ([]api.ModelRecommendation, error) {
		return recommendations, nil
	}
	codexDesktopAccessState = func(context.Context) (proxy.ClaudeDesktopAccessState, error) {
		return access, nil
	}
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

func TestResetCodexDesktopModelsDoesNothingBeforeIntegrationIsUsed(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	originalController := codexDesktop
	originalLoadModels := codexDesktopLoadModels
	t.Cleanup(func() {
		codexDesktop = originalController
		codexDesktopLoadModels = originalLoadModels
	})
	codexDesktopLoadModels = func(context.Context, []string) (string, []launch.LaunchModel, error) {
		t.Fatal("reset loaded models for an unused integration")
		return "", nil, nil
	}
	fake := &fakeCodexDesktopController{installed: true}
	codexDesktop = fake

	if err := resetCodexDesktopModels(false); err != nil {
		t.Fatal(err)
	}
	if len(fake.launches) != 0 || len(fake.updates) != 0 || fake.running || fake.configured {
		t.Fatalf("controller changed during unused reset: %#v", fake)
	}
}

func TestResetCodexDesktopModelsSavesDefaultsWithoutConnecting(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	originalController := codexDesktop
	t.Cleanup(func() { codexDesktop = originalController })
	stubCodexDesktopModelLoader(t)
	if err := config.SaveIntegration(codexDesktopIntegrationName, []string{"old-model"}); err != nil {
		t.Fatal(err)
	}
	fake := &fakeCodexDesktopController{installed: true, running: true}
	codexDesktop = fake

	if err := resetCodexDesktopModels(false); err != nil {
		t.Fatal(err)
	}
	if len(fake.launches) != 0 || len(fake.updates) != 0 || !fake.running || fake.configured {
		t.Fatalf("disconnected ChatGPT changed during reset: %#v", fake)
	}
	if got := config.IntegrationModels(codexDesktopIntegrationName); !slices.Equal(got, []string{"qwen3:8b", "glm-5.2:cloud"}) {
		t.Fatalf("saved models = %v, want recommendation defaults", got)
	}
}

func TestResetCodexDesktopModelsUpdatesStoppedProfileWithoutOpening(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	originalController := codexDesktop
	t.Cleanup(func() { codexDesktop = originalController })
	stubCodexDesktopModelLoader(t)
	if err := config.SaveIntegration(codexDesktopIntegrationName, []string{"old-model"}); err != nil {
		t.Fatal(err)
	}
	fake := &fakeCodexDesktopController{installed: true, configured: true}
	codexDesktop = fake

	if err := resetCodexDesktopModels(false); err != nil {
		t.Fatal(err)
	}
	if len(fake.launches) != 0 || len(fake.updates) != 1 || fake.running || !fake.configured {
		t.Fatalf("controller = %#v, want one stopped profile update", fake)
	}
	if !slices.Equal(fake.models, []string{"qwen3:8b", "glm-5.2:cloud"}) {
		t.Fatalf("updated models = %v, want recommendation defaults", fake.models)
	}
}

func TestResetCodexDesktopModelsRequiresConfirmationForRunningProfile(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	originalController := codexDesktop
	t.Cleanup(func() { codexDesktop = originalController })
	stubCodexDesktopModelLoader(t)
	if err := config.SaveIntegration(codexDesktopIntegrationName, []string{"old-model"}); err != nil {
		t.Fatal(err)
	}
	fake := &fakeCodexDesktopController{installed: true, configured: true, running: true}
	codexDesktop = fake

	err := resetCodexDesktopModels(false)
	if !errors.Is(err, errCodexDesktopRestartConfirmationRequired) {
		t.Fatalf("reset error = %v, want restart confirmation", err)
	}
	if len(fake.updates) != 0 {
		t.Fatalf("updates before confirmation = %v, want none", fake.updates)
	}
	if got := config.IntegrationModels(codexDesktopIntegrationName); !slices.Equal(got, []string{"old-model"}) {
		t.Fatalf("saved models before confirmation = %v, want old selection", got)
	}

	if err := resetCodexDesktopModels(true); err != nil {
		t.Fatal(err)
	}
	if len(fake.updates) != 1 || !fake.running {
		t.Fatalf("controller = %#v, want one confirmed running update", fake)
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

func TestBuildCodexDesktopModelInventoryByAccount(t *testing.T) {
	recommendations := []api.ModelRecommendation{
		{Model: "glm-5.3-flash:cloud", RequiredPlan: "pro", Thinking: &api.ModelRecommendationThinking{Values: []any{"low", "high", "max"}, Default: "max"}},
		{Model: "gemma4:31b-cloud", RequiredPlan: "free", Thinking: &api.ModelRecommendationThinking{Values: []any{false, true}, Default: false}},
	}
	local := []api.ListModelResponse{{Name: "qwen3:8b"}}

	tests := []struct {
		name         string
		access       proxy.ClaudeDesktopAccessState
		accountCloud []string
		wantCatalog  []string
		wantDefaults []string
		wantReasons  map[string]proxy.ClaudeDesktopAccessReason
	}{
		{
			name:         "signed out keeps cloud recommendations visible",
			access:       proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedOut},
			wantCatalog:  []string{"qwen3:8b", "glm-5.3-flash:cloud", "gemma4:31b-cloud"},
			wantDefaults: []string{"qwen3:8b"},
			wantReasons: map[string]proxy.ClaudeDesktopAccessReason{
				"glm-5.3-flash:cloud": proxy.ClaudeDesktopAccessSignInRequired,
				"gemma4:31b-cloud":    proxy.ClaudeDesktopAccessSignInRequired,
			},
		},
		{
			name:         "free puts Gemma before paid recommendations",
			access:       proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "free"},
			wantCatalog:  []string{"gemma4:31b-cloud", "qwen3:8b", "glm-5.3-flash:cloud"},
			wantDefaults: []string{"gemma4:31b-cloud"},
			wantReasons: map[string]proxy.ClaudeDesktopAccessReason{
				"glm-5.3-flash:cloud": proxy.ClaudeDesktopAccessUpgradeRequired,
			},
		},
		{
			name:         "team uses recommendation order without account inventory entries",
			access:       proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "team"},
			wantCatalog:  []string{"glm-5.3-flash:cloud", "gemma4:31b-cloud", "qwen3:8b"},
			wantDefaults: []string{"glm-5.3-flash:cloud", "gemma4:31b-cloud"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inventory := buildCodexDesktopModelInventory(recommendations, local, tt.accountCloud, tt.access, true, true, true)
			gotCatalog := make([]string, 0, len(inventory.Catalog))
			for _, model := range inventory.Catalog {
				gotCatalog = append(gotCatalog, model.Model.Name)
				if wantReason, ok := tt.wantReasons[model.Model.Name]; ok && model.Reason != wantReason {
					t.Errorf("%s reason = %q, want %q", model.Model.Name, model.Reason, wantReason)
				}
			}
			if !slices.Equal(gotCatalog, tt.wantCatalog) {
				t.Errorf("catalog = %v, want %v", gotCatalog, tt.wantCatalog)
			}
			if got := codexDesktopModelNames(inventory.Defaults); !slices.Equal(got, tt.wantDefaults) {
				t.Errorf("defaults = %v, want %v", got, tt.wantDefaults)
			}
		})
	}
}

func TestBuildCodexDesktopModelInventorySignedOutWithoutLocalModel(t *testing.T) {
	recommendations := []api.ModelRecommendation{
		{Model: "glm-5.3-flash:cloud", RequiredPlan: "pro"},
		{Model: "gemma4:31b-cloud", RequiredPlan: "free"},
	}
	inventory := buildCodexDesktopModelInventory(
		recommendations,
		nil,
		nil,
		proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedOut},
		true,
		true,
		false,
	)
	if len(inventory.Catalog) != 2 {
		t.Fatalf("catalog = %#v, want both cloud recommendations visible", inventory.Catalog)
	}
	if len(inventory.Available) != 0 || len(inventory.Defaults) != 0 {
		t.Fatalf("available/defaults = %#v/%#v, want no signed-out cloud selection", inventory.Available, inventory.Defaults)
	}
	if _, _, err := selectCodexDesktopModels([]string{"glm-5.3-flash:cloud"}, inventory.Available); err == nil {
		t.Fatal("signed-out cloud recommendation was selectable")
	}
}

func TestCodexDesktopDefaultsForTeamAccountUsesFirstFiveRecommendations(t *testing.T) {
	recommendations := make([]api.ModelRecommendation, 6)
	want := make([]string, 6)
	for i := range recommendations {
		name := fmt.Sprintf("model-%d:cloud", i+1)
		recommendations[i] = api.ModelRecommendation{Model: name, RequiredPlan: "pro"}
		want[i] = name
	}
	inventory := buildCodexDesktopModelInventory(
		recommendations,
		nil,
		nil,
		proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "team"},
		true,
		true,
		true,
	)

	got := codexDesktopModelNames(inventory.Defaults)
	want = want[:codexDesktopMaxModels]
	if !slices.Equal(got, want) {
		t.Fatalf("defaults = %v, want first five recommendations %v", got, want)
	}
}

func TestCodexDesktopDefaultsForTeamAccountMatchDeployedRecommendations(t *testing.T) {
	recommendations := []api.ModelRecommendation{
		{Model: "glm-5.3-flash:cloud", RequiredPlan: "pro"},
		{Model: "glm-5.2:cloud", RequiredPlan: "pro"},
		{Model: "kimi-k3:cloud", RequiredPlan: "pro"},
		{Model: "deepseek-v4-pro", RequiredPlan: "pro"},
		{Model: "gemma4:31b-cloud", RequiredPlan: "free"},
	}
	inventory := buildCodexDesktopModelInventory(
		recommendations,
		[]api.ListModelResponse{{Name: "gemma4:12b"}},
		[]string{"gemma4:31b:cloud"},
		proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "team"},
		true,
		true,
		true,
	)

	got := codexDesktopModelNames(inventory.Defaults)
	want := []string{
		"glm-5.3-flash:cloud",
		"glm-5.2:cloud",
		"kimi-k3:cloud",
		"deepseek-v4-pro:cloud",
		"gemma4:31b:cloud",
	}
	if !slices.Equal(got, want) {
		t.Fatalf("defaults = %v, want deployed recommendation order %v", got, want)
	}
	if len(inventory.Catalog) != len(recommendations)+1 {
		t.Fatalf("catalog = %#v, want five recommendations and one local model", inventory.Catalog)
	}
	for i, entry := range inventory.Catalog[:len(recommendations)] {
		if !entry.Recommended || entry.Availability != proxy.ClaudeDesktopAvailabilityAvailable {
			t.Fatalf("recommendation %d = %#v, want available recommendation", i, entry)
		}
	}
}

func TestCodexDesktopDefaultsForFreeAccountFollowRecommendationMetadata(t *testing.T) {
	recommendations := []api.ModelRecommendation{
		{Model: "paid-model:cloud", RequiredPlan: "pro"},
		{Model: "new-free-model:cloud", RequiredPlan: "free"},
	}
	inventory := buildCodexDesktopModelInventory(
		recommendations,
		nil,
		nil,
		proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "free"},
		true,
		true,
		true,
	)

	if got, want := codexDesktopModelNames(inventory.Defaults), []string{"new-free-model:cloud"}; !slices.Equal(got, want) {
		t.Fatalf("defaults = %v, want endpoint-provided Free recommendation %v", got, want)
	}
}

func TestBuildCodexDesktopModelInventoryMergesEquivalentCloudAliases(t *testing.T) {
	inventory := buildCodexDesktopModelInventory(
		[]api.ModelRecommendation{{Model: "gemma4:31b-cloud", RequiredPlan: "free"}},
		nil,
		[]string{"gemma4:31b:cloud"},
		proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "team"},
		true,
		true,
		true,
	)

	if len(inventory.Catalog) != 1 {
		t.Fatalf("catalog = %#v, want one merged model", inventory.Catalog)
	}
	model := inventory.Catalog[0]
	if model.Model.Name != "gemma4:31b:cloud" || model.DisplayName != "gemma4:31b-cloud" {
		t.Fatalf("merged model = %#v, want inventory route with recommendation display name", model)
	}
	if got := codexDesktopModelNames(inventory.Defaults); !slices.Equal(got, []string{"gemma4:31b:cloud"}) {
		t.Fatalf("defaults = %v, want the merged cloud route", got)
	}
}

func TestCodexDesktopModelStatusesKeepsSavedUnavailableModel(t *testing.T) {
	statuses := codexDesktopModelStatuses(codexDesktopModelInventory{}, []string{"saved-model:cloud"})
	if len(statuses) != 1 {
		t.Fatalf("statuses = %#v, want saved model", statuses)
	}
	status := statuses[0]
	if status.Name != "saved-model:cloud" || !status.Selected ||
		status.Availability != string(proxy.ClaudeDesktopAvailabilityUnknown) ||
		status.Reason != string(proxy.ClaudeDesktopAccessVerificationUnavailable) {
		t.Fatalf("status = %#v, want selected model with unverified access", status)
	}
}

func TestCodexDesktopInventoryModelAccessReportsCloudOff(t *testing.T) {
	availability, reason := codexDesktopInventoryModelAccess(
		launch.LaunchModel{Name: "cloud-model:cloud", Remote: true},
		proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOff},
		true,
		false,
	)
	if availability != proxy.ClaudeDesktopAvailabilityUnavailable || reason != proxy.ClaudeDesktopAccessCloudOff {
		t.Fatalf("access = %q/%q, want unavailable/cloud_off", availability, reason)
	}
}

func TestLoadCodexDesktopRecommendationsUsesCodexQualifier(t *testing.T) {
	useTestOllamaRequestSigner(t)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/experimental/model-recommendations" || r.URL.Query().Get("app") != "codex-desktop" {
			t.Fatalf("request URL = %q", r.URL.String())
		}
		if r.Header.Get("Authorization") != "test-public-key:test-signature" {
			t.Fatal("recommendation request is missing public-key identity")
		}
		_, _ = w.Write([]byte(`{"recommendations":[{"model":"glm-5.3-flash:cloud","required_plan":"pro","thinking":{"values":["low","high","max"],"default":"max"}}],"mappings":{"ignored":{"model":"other"}}}`))
	}))
	defer server.Close()

	previousClient := codexDesktopRecommendationsClient
	previousEndpoint := codexDesktopRecommendationsEndpoint
	codexDesktopRecommendationsClient = server.Client()
	codexDesktopRecommendationsEndpoint = func() string {
		return server.URL + "/api/experimental/model-recommendations?app=codex-desktop"
	}
	t.Cleanup(func() {
		codexDesktopRecommendationsClient = previousClient
		codexDesktopRecommendationsEndpoint = previousEndpoint
	})

	recommendations, err := loadCodexDesktopRecommendations(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(recommendations) != 1 || recommendations[0].Model != "glm-5.3-flash:cloud" || recommendations[0].Thinking == nil {
		t.Fatalf("recommendations = %#v, want Codex catalog with thinking metadata", recommendations)
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
	stubCodexDesktopCatalogSources(t, nil, proxy.ClaudeDesktopAccessState{
		Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedOut,
	})
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

func TestLoadCodexDesktopModelInventoryRetriesAccountAccessDuringServerRestart(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			_, _ = w.Write([]byte(`{"models":[]}`))
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()
	base, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	client := api.NewClient(base, server.Client())

	originalClientFactory := codexDesktopClientFactory
	originalCloudModels := codexDesktopCloudModels
	originalRecommendations := codexDesktopRecommendations
	originalAccessState := codexDesktopAccessState
	originalAttempts := codexDesktopModelLoadAttempts
	originalRetryWait := codexDesktopModelRetryWait
	t.Cleanup(func() {
		codexDesktopClientFactory = originalClientFactory
		codexDesktopCloudModels = originalCloudModels
		codexDesktopRecommendations = originalRecommendations
		codexDesktopAccessState = originalAccessState
		codexDesktopModelLoadAttempts = originalAttempts
		codexDesktopModelRetryWait = originalRetryWait
	})
	codexDesktopClientFactory = func() (*api.Client, error) { return client, nil }
	codexDesktopCloudModels = func(context.Context) ([]string, error) { return nil, nil }
	codexDesktopRecommendations = func(context.Context) ([]api.ModelRecommendation, error) {
		return []api.ModelRecommendation{{Model: "glm-5.3-flash:cloud", RequiredPlan: "pro"}}, nil
	}
	accessRequests := 0
	codexDesktopAccessState = func(context.Context) (proxy.ClaudeDesktopAccessState, error) {
		accessRequests++
		if accessRequests < 3 {
			return proxy.ClaudeDesktopAccessState{}, errors.New("server is restarting")
		}
		return proxy.ClaudeDesktopAccessState{
			Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "team",
		}, nil
	}
	codexDesktopModelLoadAttempts = 3
	codexDesktopModelRetryWait = time.Millisecond

	inventory, err := loadCodexDesktopModelInventory(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if accessRequests != 3 {
		t.Fatalf("account access requests = %d, want 3", accessRequests)
	}
	if got := codexDesktopModelNames(inventory.Defaults); !slices.Equal(got, []string{"glm-5.3-flash:cloud"}) {
		t.Fatalf("defaults = %v, want recommendation after server restart", got)
	}
}

func TestLoadCodexDesktopModelsHydratesAccountOnlyCloudCapabilities(t *testing.T) {
	stubCodexDesktopCatalogSources(t, []api.ModelRecommendation{{
		Model: "glm-5.3-flash:cloud",
		Thinking: &api.ModelRecommendationThinking{
			Values: []any{"low", "high", "max"}, Default: "max",
		},
	}}, proxy.ClaudeDesktopAccessState{
		Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "pro",
	})
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
	if models[0].Thinking == nil || !slices.Equal(models[0].Thinking.Values, []any{"low", "high", "max"}) || models[0].Thinking.Default != "max" {
		t.Fatalf("thinking = %#v, want recommendation endpoint metadata", models[0].Thinking)
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
