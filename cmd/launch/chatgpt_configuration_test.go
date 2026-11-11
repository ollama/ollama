package launch

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"testing/synctest"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/internal/proxy"
)

type chatGPTFixtureTransport struct{ t *testing.T }

func (transport chatGPTFixtureTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	body := `{}`
	switch r.URL.Path {
	case "/api/status":
		body = `{"cloud":{"disabled":false}}`
	case "/api/tags":
		body = `{"models":[{"name":"custom-primary"},{"name":"custom-secondary"},{"name":"unselected"}]}`
	case "/api/show":
		body = `{"capabilities":["completion","tools","vision"],"thinking":{"values":[false,true],"default":true},"details":{"family":"fixture"}}`
	default:
		transport.t.Errorf("unexpected fixture request %s", r.URL.Path)
		return nil, fmt.Errorf("unexpected fixture request %s", r.URL.Path)
	}
	return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body)), Request: r}, nil
}

type chatGPTFixtureController struct {
	launcherManagedListRunner
	running  bool
	restores int
	restarts int
	opens    int
	updates  int
	apply    func(string, []LaunchModel) error
}

func (controller *chatGPTFixtureController) Installed() bool { return true }
func (controller *chatGPTFixtureController) OllamaConfigured() bool {
	return controller.currentModel != ""
}
func (controller *chatGPTFixtureController) Running() bool { return controller.running }
func (controller *chatGPTFixtureController) UseOllamaFromDesktop(primary string, models []LaunchModel, confirmed bool) error {
	controller.opens++
	return controller.applyModels(primary, models, confirmed)
}
func (controller *chatGPTFixtureController) UpdateOllamaModelsFromDesktop(primary string, models []LaunchModel, confirmed bool) error {
	controller.updates++
	return controller.applyModels(primary, models, confirmed)
}
func (controller *chatGPTFixtureController) applyModels(primary string, models []LaunchModel, confirmed bool) error {
	if controller.running && !confirmed {
		return ErrCodexAppRestartConfirmationRequired
	}
	if controller.apply != nil {
		if err := controller.apply(primary, models); err != nil {
			return err
		}
	}
	return controller.ConfigureWithModels(primary, models)
}
func (controller *chatGPTFixtureController) RestoreFromDesktop(bool) error {
	controller.restores++
	controller.currentModel = ""
	return nil
}
func (controller *chatGPTFixtureController) RestartFromDesktop(confirmed bool) error {
	if controller.running && !confirmed {
		return ErrCodexAppRestartConfirmationRequired
	}
	controller.restarts++
	return nil
}

func chatGPTFixture(t *testing.T) (*launcherClient, *chatGPTFixtureController, *ChatGPTModelSource) {
	t.Helper()
	setLaunchTestHome(t, t.TempDir())
	client, err := newLauncherClient(defaultLaunchPolicy(false, false))
	if err != nil {
		t.Fatal(err)
	}
	client.apiClient = api.NewClient(&url.URL{Scheme: "http", Host: "fixture.invalid"}, &http.Client{Transport: chatGPTFixtureTransport{t: t}})
	source := &ChatGPTModelSource{
		Client: client.apiClient, Attempts: 1,
		Recommendations: func(context.Context) ([]api.ModelRecommendation, error) { return nil, nil },
		CloudModels:     func(context.Context) ([]string, error) { return nil, nil },
		AccessState: func(context.Context) (proxy.ClaudeDesktopAccessState, error) {
			return proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedOut}, nil
		},
	}
	original := chatGPTModelSourceFactory
	chatGPTModelSourceFactory = func(*api.Client) *ChatGPTModelSource { return source }
	t.Cleanup(func() { chatGPTModelSourceFactory = original })
	controller := &chatGPTFixtureController{}
	controller.currentModel = "custom-primary"
	controller.onboardingComplete = true
	return client, controller, source
}

func TestChatGPTLaunchPreservesSavedSelection(t *testing.T) {
	client, runner, _ := chatGPTFixture(t)
	want := []string{"custom-primary", "custom-secondary"}
	if err := config.SaveIntegration(chatGPTIntegrationName, want); err != nil {
		t.Fatal(err)
	}
	saved, err := config.LoadIntegration(chatGPTIntegrationName)
	if err != nil {
		t.Fatal(err)
	}
	saved.Onboarded = true
	if err := client.launchManagedSingleIntegration(t.Context(), chatGPTIntegrationName, runner, runner, saved, IntegrationLaunchRequest{}); err != nil {
		t.Fatal(err)
	}
	if got := config.IntegrationModels(chatGPTIntegrationName); !slices.Equal(got, want) {
		t.Fatalf("plain CLI relaunch saved %v, want exact app selection %v", got, want)
	}
	for _, got := range runner.configuredModelLists {
		if !slices.Equal(got, want) {
			t.Fatalf("CLI applied catalog %v, want exact selected subset %v", got, want)
		}
	}
}

func TestChatGPTLauncherSummaryUsesSavedPrimaryWhenDisconnected(t *testing.T) {
	client, controller, _ := chatGPTFixture(t)
	controller.currentModel = ""
	saved := &config.IntegrationConfig{Models: []string{"custom-primary", "custom-secondary"}, Primary: "custom-secondary"}
	if err := config.SwapPrimaryModelSelection(chatGPTIntegrationName, nil, saved); err != nil {
		t.Fatal(err)
	}
	current, usable, err := client.launcherManagedModelState(t.Context(), chatGPTIntegrationName, controller)
	if err != nil || usable || current != saved.Primary {
		t.Fatalf("disconnected summary model=%s usable=%t error=%v, want saved primary", current, usable, err)
	}
}

func TestChatGPTCLIPrimaryDoesNotEditSelectedList(t *testing.T) {
	for _, override := range []string{"", "custom-secondary", "unselected"} {
		t.Run(fmt.Sprintf("override=%s", override), func(t *testing.T) {
			client, controller, _ := chatGPTFixture(t)
			want := []string{"custom-primary", "custom-secondary"}
			before := &config.IntegrationConfig{Models: want, Primary: "custom-secondary"}
			if err := config.SwapPrimaryModelSelection(chatGPTIntegrationName, nil, before); err != nil {
				t.Fatal(err)
			}
			if err := client.launchManagedSingleIntegration(t.Context(), chatGPTIntegrationName, controller, controller, before, IntegrationLaunchRequest{ModelOverride: override, ConfigureOnly: true}); err != nil {
				t.Fatal(err)
			}
			if override == "unselected" {
				want = append(want, override)
			}
			got, err := LoadChatGPTConfiguration()
			if err != nil {
				t.Fatal(err)
			}
			primary := "custom-secondary"
			if override != "" {
				primary = override
			}
			if !slices.Equal(got.Models, want) || got.Primary != primary || controller.currentModel != primary {
				t.Fatalf("saved=%+v current=%s, want list=%v primary=%s", got, controller.currentModel, want, primary)
			}
			if controller.opens != 0 || controller.updates != 1 {
				t.Fatalf("configure-only opens=%d updates=%d", controller.opens, controller.updates)
			}
			if names := controller.configuredModelLists[0]; !slices.Equal(names, want) {
				t.Fatalf("catalog=%v want=%v", names, want)
			}
			model := controller.configuredModels[0][0]
			if !model.ToolCapable || model.Thinking == nil || model.Details.Family != "fixture" {
				t.Fatalf("catalog lost capability metadata: %+v", model)
			}
		})
	}
}

func TestChatGPTAppAndCLIPreserveImplicitDefaults(t *testing.T) {
	client, controller, source := chatGPTFixture(t)
	load := func(ctx context.Context, selected []string) (string, []LaunchModel, error) {
		return source.Resolve(ctx, selected, false)
	}
	if err := ApplyChatGPTModels(t.Context(), controller, load, ChatGPTApplyRequest{OpenWhenStopped: true}); err != nil {
		t.Fatal(err)
	}
	if err := client.launchChatGPTIntegration(t.Context(), controller, IntegrationLaunchRequest{ConfigureOnly: true}); err != nil {
		t.Fatal(err)
	}
	saved, err := LoadChatGPTConfiguration()
	if err != nil {
		t.Fatal(err)
	}
	if len(saved.Models) != 0 || saved.Primary != "" {
		t.Fatalf("defaults became explicit: %+v", saved)
	}
	if len(controller.configuredModelLists) != 2 || !slices.Equal(controller.configuredModelLists[0], controller.configuredModelLists[1]) {
		t.Fatalf("entry points disagree: %v", controller.configuredModelLists)
	}
}

func TestChatGPTFailedApplyPreservesConcurrentSelection(t *testing.T) {
	_, controller, source := chatGPTFixture(t)
	before := []string{"custom-primary"}
	if err := config.SaveIntegration(chatGPTIntegrationName, before); err != nil {
		t.Fatal(err)
	}
	controller.apply = func(string, []LaunchModel) error {
		current, err := LoadChatGPTConfiguration()
		if err != nil {
			t.Fatal(err)
		}
		later := &config.IntegrationConfig{Models: []string{"unselected"}, Primary: "unselected"}
		if err := config.SwapPrimaryModelSelection(chatGPTIntegrationName, current, later); err != nil {
			t.Fatal(err)
		}
		controller.currentModel = "unselected"
		return errors.New("fixture apply failure")
	}
	load := func(ctx context.Context, selected []string) (string, []LaunchModel, error) {
		return source.Resolve(ctx, selected, false)
	}
	err := ApplyChatGPTModels(t.Context(), controller, load, ChatGPTApplyRequest{Models: []string{"custom-secondary"}})
	if err == nil || !strings.Contains(err.Error(), "preserving the newer selection") {
		t.Fatalf("error=%v", err)
	}
	saved, err := LoadChatGPTConfiguration()
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(saved.Models, []string{"unselected"}) || saved.Primary != "unselected" || controller.currentModel != "unselected" || controller.restores != 0 || controller.updates != 1 {
		t.Fatalf("overwrote newer selection: saved=%+v current=%s updates=%d restores=%d", saved, controller.currentModel, controller.updates, controller.restores)
	}
}

func TestChatGPTCLIRestartKeepsStaleSavedCatalog(t *testing.T) {
	client, controller, _ := chatGPTFixture(t)
	before := &config.IntegrationConfig{Models: []string{"removed", "custom-primary"}, Primary: "removed"}
	if err := config.SwapPrimaryModelSelection(chatGPTIntegrationName, nil, before); err != nil {
		t.Fatal(err)
	}
	if err := client.launchChatGPTIntegration(t.Context(), controller, IntegrationLaunchRequest{}); err != nil {
		t.Fatal(err)
	}
	saved, err := LoadChatGPTConfiguration()
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(saved.Models, before.Models) || saved.Primary != before.Primary || controller.restarts != 1 || controller.opens != 0 || controller.updates != 0 {
		t.Fatalf("stale relaunch changed saved catalog: saved=%+v restarts=%d opens=%d updates=%d", saved, controller.restarts, controller.opens, controller.updates)
	}
}

func TestChatGPTValidatesBeforeRestartConsentAndWrites(t *testing.T) {
	for _, selection := range [][]string{{"unavailable"}, {"a", "b", "c", "d", "e", "f"}} {
		t.Run(strings.Join(selection, ","), func(t *testing.T) {
			_, controller, source := chatGPTFixture(t)
			controller.running = true
			before := []string{"custom-primary"}
			if err := config.SaveIntegration(chatGPTIntegrationName, before); err != nil {
				t.Fatal(err)
			}
			load := func(ctx context.Context, selected []string) (string, []LaunchModel, error) {
				return source.Resolve(ctx, selected, false)
			}
			err := ApplyChatGPTModels(t.Context(), controller, load, ChatGPTApplyRequest{
				Models:         selection,
				ConfirmRestart: func() (bool, error) { t.Fatal("requested restart for invalid selection"); return false, nil },
			})
			if err == nil {
				t.Fatal("invalid selection accepted")
			}
			if got := config.IntegrationModels(chatGPTIntegrationName); !slices.Equal(got, before) || controller.opens != 0 || controller.updates != 0 {
				t.Fatalf("invalid selection changed state: models=%v opens=%d updates=%d", got, controller.opens, controller.updates)
			}
		})
	}
}

func TestChatGPTFailedApplyRestoresPreviousSelectionAndPrimary(t *testing.T) {
	_, controller, source := chatGPTFixture(t)
	before := &config.IntegrationConfig{Models: []string{"custom-primary", "custom-secondary"}, Primary: "custom-secondary"}
	if err := config.SwapPrimaryModelSelection(chatGPTIntegrationName, nil, before); err != nil {
		t.Fatal(err)
	}
	controller.currentModel = "custom-secondary"
	calls := 0
	controller.apply = func(string, []LaunchModel) error {
		calls++
		if calls == 1 {
			return errors.New("fixture failure")
		}
		return nil
	}
	load := func(ctx context.Context, selected []string) (string, []LaunchModel, error) {
		return source.Resolve(ctx, selected, false)
	}
	if err := ApplyChatGPTModels(t.Context(), controller, load, ChatGPTApplyRequest{Models: []string{"unselected"}}); err == nil {
		t.Fatal("failed apply returned success")
	}
	saved, err := LoadChatGPTConfiguration()
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(saved.Models, before.Models) || saved.Primary != before.Primary || controller.currentModel != before.Primary || controller.restores != 0 || controller.updates != 2 {
		t.Fatalf("rollback saved=%+v current=%s updates=%d restores=%d", saved, controller.currentModel, controller.updates, controller.restores)
	}
}

func TestChatGPTCLIListEditorIsTheOnlyPathThatRemovesChoices(t *testing.T) {
	client, controller, _ := chatGPTFixture(t)
	withLauncherHooks(t)
	before := []string{"custom-primary", "custom-secondary"}
	if err := config.SaveIntegration(chatGPTIntegrationName, before); err != nil {
		t.Fatal(err)
	}
	DefaultMultiSelectorWithUpdates = nil
	DefaultMultiSelector = func(_ string, _ []SelectionItem, checked []string) ([]string, error) {
		if !slices.Equal(checked, before) {
			t.Fatalf("checked=%v want=%v", checked, before)
		}
		return []string{"custom-secondary"}, nil
	}
	if err := client.launchChatGPTIntegration(t.Context(), controller, IntegrationLaunchRequest{ForceConfigure: true, ConfigureOnly: true}); err != nil {
		t.Fatal(err)
	}
	if got := config.IntegrationModels(chatGPTIntegrationName); !slices.Equal(got, []string{"custom-secondary"}) {
		t.Fatalf("explicit list edit saved=%v", got)
	}
}

func TestChatGPTCancelledConsentCannotWritePreferencesOrProfile(t *testing.T) {
	_, controller, source := chatGPTFixture(t)
	controller.running = true
	before := []string{"custom-primary"}
	if err := config.SaveIntegration(chatGPTIntegrationName, before); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	load := func(ctx context.Context, selected []string) (string, []LaunchModel, error) {
		return source.Resolve(ctx, selected, false)
	}
	err := ApplyChatGPTModels(ctx, controller, load, ChatGPTApplyRequest{
		Models:         []string{"custom-secondary"},
		ConfirmRestart: func() (bool, error) { cancel(); return true, nil },
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error=%v, want cancellation", err)
	}
	if got := config.IntegrationModels(chatGPTIntegrationName); !slices.Equal(got, before) || controller.opens != 0 || controller.updates != 0 {
		t.Fatalf("cancelled attempt changed state: models=%v opens=%d updates=%d", got, controller.opens, controller.updates)
	}
}

func TestChatGPTUnreadablePreferencesCannotBecomeDefaults(t *testing.T) {
	_, controller, _ := chatGPTFixture(t)
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(home, ".ollama", "config.json")
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	before := `{"integrations":{"chatgpt":null},"unrelated":"keep"}`
	if err := os.WriteFile(path, []byte(before), 0o600); err != nil {
		t.Fatal(err)
	}
	load := func(context.Context, []string) (string, []LaunchModel, error) {
		t.Fatal("invalid preferences triggered model loading")
		return "", nil, nil
	}
	if err := ApplyChatGPTModels(t.Context(), controller, load, ChatGPTApplyRequest{UseSavedSelection: true}); err == nil {
		t.Fatal("invalid preferences accepted")
	}
	got, err := os.ReadFile(path)
	if err != nil || string(got) != before || controller.opens != 0 || controller.updates != 0 {
		t.Fatalf("invalid preferences were overwritten: error=%v opens=%d updates=%d", err, controller.opens, controller.updates)
	}
}

func TestChatGPTReviewListEditsKeepSelectedPrimary(t *testing.T) {
	for _, selected := range [][]string{{"custom-primary", "custom-secondary", "unselected"}, {"custom-secondary", "custom-primary"}} {
		t.Run(strings.Join(selected, ","), func(t *testing.T) {
			_, controller, source := chatGPTFixture(t)
			before := &config.IntegrationConfig{Models: []string{"custom-primary", "custom-secondary"}, Primary: "custom-secondary"}
			if err := config.SwapPrimaryModelSelection(chatGPTIntegrationName, nil, before); err != nil {
				t.Fatal(err)
			}
			load := func(ctx context.Context, names []string) (string, []LaunchModel, error) {
				return source.Resolve(ctx, names, false)
			}
			if err := ApplyChatGPTModels(t.Context(), controller, load, ChatGPTApplyRequest{Models: selected}); err != nil {
				t.Fatal(err)
			}
			saved, err := LoadChatGPTConfiguration()
			if err != nil {
				t.Fatal(err)
			}
			if saved.Primary != "custom-secondary" || controller.currentModel != "custom-secondary" || !slices.Equal(saved.Models, selected) {
				t.Fatalf("list edit unexpectedly changed primary: saved=%+v live=%s", saved, controller.currentModel)
			}
		})
	}
}

func TestChatGPTReviewPrimaryOverrideKeepsImplicitDefaults(t *testing.T) {
	client, controller, source := chatGPTFixture(t)
	if err := client.launchChatGPTIntegration(t.Context(), controller, IntegrationLaunchRequest{ModelOverride: "custom-secondary", ConfigureOnly: true}); err != nil {
		t.Fatal(err)
	}
	if err := client.launchChatGPTIntegration(t.Context(), controller, IntegrationLaunchRequest{ConfigureOnly: true}); err != nil {
		t.Fatal(err)
	}
	load := func(ctx context.Context, names []string) (string, []LaunchModel, error) {
		return source.Resolve(ctx, names, false)
	}
	if err := ApplyChatGPTModels(t.Context(), controller, load, ChatGPTApplyRequest{}); err != nil {
		t.Fatal(err)
	}
	saved, err := LoadChatGPTConfiguration()
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"custom-primary", "custom-secondary", "unselected"}
	if len(saved.Models) != 0 || saved.Primary != "custom-secondary" || controller.currentModel != "custom-secondary" {
		t.Fatalf("primary override materialized/replaced defaults: saved=%+v live=%s", saved, controller.currentModel)
	}
	for _, models := range controller.configuredModelLists {
		if !slices.Equal(models, want) {
			t.Fatalf("primary override replaced implicit default catalog: got=%v want=%v", models, want)
		}
	}
	updates := controller.updates
	if err := ResetChatGPTModels(); err != nil {
		t.Fatal(err)
	}
	if controller.updates != updates || controller.currentModel != "custom-secondary" {
		t.Fatal("reset preferences changed the live app")
	}
	if err := client.launchChatGPTIntegration(t.Context(), controller, IntegrationLaunchRequest{ConfigureOnly: true}); err != nil {
		t.Fatal(err)
	}
	saved, err = LoadChatGPTConfiguration()
	if err != nil {
		t.Fatal(err)
	}
	if len(saved.Models) != 0 || saved.Primary != "" || controller.currentModel != "custom-primary" {
		t.Fatalf("reset did not restore account defaults on next apply: saved=%+v live=%s", saved, controller.currentModel)
	}
}

func TestChatGPTReviewOverrideOutsideDefaultsKeepsChoicesWithinLimit(t *testing.T) {
	for _, full := range []bool{false, true} {
		t.Run(fmt.Sprintf("full=%t", full), func(t *testing.T) {
			client, controller, source := chatGPTFixture(t)
			recommended := []api.ModelRecommendation{{Model: "custom-primary"}}
			target := "custom-secondary"
			if full {
				recommended = []api.ModelRecommendation{{Model: "custom-primary"}, {Model: "custom-secondary"}, {Model: "unselected"}, {Model: "fourth"}, {Model: "fifth"}, {Model: "sixth"}}
				target = "sixth"
			}
			source.Recommendations = func(context.Context) ([]api.ModelRecommendation, error) { return recommended, nil }
			err := client.launchChatGPTIntegration(t.Context(), controller, IntegrationLaunchRequest{ModelOverride: target, ConfigureOnly: true})
			if full {
				if err == nil || !strings.Contains(err.Error(), "--config") {
					t.Fatalf("full list override error=%v, want explicit-list guidance", err)
				}
				saved, loadErr := LoadChatGPTConfiguration()
				if loadErr != nil || saved != nil || controller.updates != 0 {
					t.Fatalf("full default list was changed: saved=%+v error=%v updates=%d", saved, loadErr, controller.updates)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			saved, err := LoadChatGPTConfiguration()
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(saved.Models, []string{"custom-primary", "custom-secondary"}) || saved.Primary != target {
				t.Fatalf("new model discarded existing default choices: %+v", saved)
			}
		})
	}
}

func TestChatGPTReviewPrimaryAliasKeepsOneEnrichedCatalogEntry(t *testing.T) {
	client, controller, source := chatGPTFixture(t)
	source.Recommendations = func(context.Context) ([]api.ModelRecommendation, error) {
		return []api.ModelRecommendation{{Model: "fixture:8b:cloud", RequiredPlan: "free", ContextLength: 262144, MaxOutputTokens: 8192}}, nil
	}
	before := &config.IntegrationConfig{Models: []string{"fixture:8b-cloud"}, Primary: "fixture:8b-cloud"}
	if err := config.SwapPrimaryModelSelection(chatGPTIntegrationName, nil, before); err != nil {
		t.Fatal(err)
	}
	if err := client.launchChatGPTIntegration(t.Context(), controller, IntegrationLaunchRequest{ConfigureOnly: true}); err != nil {
		t.Fatal(err)
	}
	catalog := codexAppCatalogModels(controller.currentModel, controller.configuredModels[0])
	if len(catalog) != 1 || catalog[0].Name != before.Primary || catalog[0].ContextLength != 262144 || catalog[0].MaxOutputTokens != 8192 || catalog[0].Thinking == nil {
		t.Fatalf("alias produced duplicate or fallback catalog metadata: %+v", catalog)
	}
}

func TestChatGPTFlowEditorKeepsSavedAliasesChecked(t *testing.T) {
	client, _, source := chatGPTFixture(t)
	withLauncherHooks(t)
	source.Recommendations = func(context.Context) ([]api.ModelRecommendation, error) {
		return []api.ModelRecommendation{{Model: "fixture:8b:cloud", RequiredPlan: "free"}}, nil
	}
	saved := &config.IntegrationConfig{Models: []string{"fixture:8b-cloud", "custom-primary"}, Primary: "fixture:8b-cloud"}
	if err := config.SwapPrimaryModelSelection(chatGPTIntegrationName, nil, saved); err != nil {
		t.Fatal(err)
	}
	DefaultModelListSelector = func(_ string, items []SelectionItem, checked []string, limit int) ([]string, error) {
		if limit != ChatGPTMaxModels {
			t.Fatalf("ChatGPT list limit=%d, want %d", limit, ChatGPTMaxModels)
		}
		known := map[string]bool{}
		for _, item := range items {
			known[item.Name] = true
		}
		for _, name := range checked {
			if !known[name] {
				t.Errorf("saved choice %q has no matching checkbox", name)
			}
		}
		return checked, nil
	}
	controller := &chatGPTFixtureController{}
	controller.currentModel = saved.Primary
	if err := client.launchChatGPTIntegration(t.Context(), controller, IntegrationLaunchRequest{ForceConfigure: true, ConfigureOnly: true}); err != nil {
		t.Fatal(err)
	}
	after, err := LoadChatGPTConfiguration()
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(after.Models, saved.Models) || after.Primary != saved.Primary {
		t.Fatalf("opening and confirming the editor changed the saved setup: %+v", after)
	}
}

func TestChatGPTFlowDefaultChangesCannotSilentlyReplacePrimary(t *testing.T) {
	client, controller, source := chatGPTFixture(t)
	saved := &config.IntegrationConfig{Primary: "custom-secondary"}
	if err := config.SwapPrimaryModelSelection(chatGPTIntegrationName, nil, saved); err != nil {
		t.Fatal(err)
	}
	controller.currentModel = saved.Primary
	source.Recommendations = func(context.Context) ([]api.ModelRecommendation, error) {
		return []api.ModelRecommendation{{Model: "custom-primary"}}, nil
	}
	err := client.launchChatGPTIntegration(t.Context(), controller, IntegrationLaunchRequest{ConfigureOnly: true})
	if err == nil || !strings.Contains(err.Error(), "starting model") {
		t.Fatalf("recommendation changes silently changed the starting model: error=%v live=%s", err, controller.currentModel)
	}
	after, loadErr := LoadChatGPTConfiguration()
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	if len(after.Models) != 0 || after.Primary != saved.Primary || controller.currentModel != saved.Primary || controller.updates != 0 {
		t.Fatalf("failed launch changed the saved setup: saved=%+v live=%s", after, controller.currentModel)
	}
}

func TestChatGPTFlowRollbackRestoresExactPrimaryAlias(t *testing.T) {
	_, controller, source := chatGPTFixture(t)
	source.Recommendations = func(context.Context) ([]api.ModelRecommendation, error) {
		return []api.ModelRecommendation{{Model: "fixture:8b:cloud", RequiredPlan: "free", ContextLength: 262144}}, nil
	}
	saved := &config.IntegrationConfig{Models: []string{"fixture:8b-cloud"}, Primary: "fixture:8b-cloud"}
	if err := config.SwapPrimaryModelSelection(chatGPTIntegrationName, nil, saved); err != nil {
		t.Fatal(err)
	}
	controller.currentModel = saved.Primary
	controller.apply = func(string, []LaunchModel) error {
		if controller.updates == 1 {
			return errors.New("fixture apply failure")
		}
		return nil
	}
	load := func(ctx context.Context, names []string) (string, []LaunchModel, error) {
		return source.Resolve(ctx, names, false)
	}
	if err := ApplyChatGPTModels(t.Context(), controller, load, ChatGPTApplyRequest{Models: []string{"custom-primary"}}); err == nil {
		t.Fatal("failed apply succeeded")
	}
	if controller.currentModel != saved.Primary {
		t.Fatalf("rollback changed exact primary: got=%s want=%s", controller.currentModel, saved.Primary)
	}
	catalog := codexAppCatalogModels(controller.currentModel, controller.configuredModels[0])
	if len(catalog) != 1 || catalog[0].ContextLength != 262144 {
		t.Fatalf("rollback lost enriched alias catalog: %+v", catalog)
	}
}

func TestChatGPTReviewInteractiveWaitDoesNotExpireLaunch(t *testing.T) {
	for _, phase := range []string{"list editor", "restart consent"} {
		t.Run(phase, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				client, controller, _ := chatGPTFixture(t)
				withLauncherHooks(t)
				client.policy = defaultLaunchPolicy(true, false)
				if err := config.SaveIntegration(chatGPTIntegrationName, []string{"custom-primary", "custom-secondary"}); err != nil {
					t.Fatal(err)
				}
				request := IntegrationLaunchRequest{ConfigureOnly: true}
				if phase == "list editor" {
					request.ForceConfigure = true
					DefaultMultiSelectorWithUpdates = nil
					DefaultMultiSelector = func(_ string, _ []SelectionItem, checked []string) ([]string, error) {
						time.Sleep(11 * time.Second)
						return checked, nil
					}
				} else {
					controller.running = true
					DefaultConfirmPrompt = func(string, ConfirmOptions) (bool, error) { time.Sleep(11 * time.Second); return true, nil }
				}
				if err := client.launchChatGPTIntegration(t.Context(), controller, request); err != nil {
					t.Fatalf("time spent in %s aborted launch: %v", phase, err)
				}
				if controller.updates != 1 {
					t.Fatalf("updates=%d, want one successful configured update", controller.updates)
				}
			})
		})
	}
}

func TestChatGPTReviewRollbackRechecksChangesDuringDiscovery(t *testing.T) {
	_, controller, source := chatGPTFixture(t)
	before := &config.IntegrationConfig{Models: []string{"custom-primary"}, Primary: "custom-primary"}
	if err := config.SwapPrimaryModelSelection(chatGPTIntegrationName, nil, before); err != nil {
		t.Fatal(err)
	}
	controller.apply = func(string, []LaunchModel) error {
		if controller.updates == 1 {
			return errors.New("fixture apply failure")
		}
		return nil
	}
	loads := 0
	load := func(ctx context.Context, names []string) (string, []LaunchModel, error) {
		loads++
		if loads == 2 {
			current, err := LoadChatGPTConfiguration()
			if err != nil {
				t.Fatal(err)
			}
			later := &config.IntegrationConfig{Models: []string{"unselected"}, Primary: "unselected"}
			if err := config.SwapPrimaryModelSelection(chatGPTIntegrationName, current, later); err != nil {
				t.Fatal(err)
			}
			controller.currentModel = "unselected"
		}
		return source.Resolve(ctx, names, false)
	}
	err := ApplyChatGPTModels(t.Context(), controller, load, ChatGPTApplyRequest{Models: []string{"custom-secondary"}})
	if err == nil || !strings.Contains(err.Error(), "preserving the newer selection") {
		t.Fatalf("rollback error=%v, want a protected later change", err)
	}
	saved, err := LoadChatGPTConfiguration()
	if err != nil {
		t.Fatal(err)
	}
	if saved.Primary != "unselected" || controller.currentModel != "unselected" || controller.updates != 1 || controller.restores != 0 {
		t.Fatalf("rollback overwrote later state: saved=%+v live=%s updates=%d restores=%d", saved, controller.currentModel, controller.updates, controller.restores)
	}
}

func TestChatGPTReviewRestartPromptDoesNotBlockSettings(t *testing.T) {
	for _, disconnect := range []bool{false, true} {
		t.Run(fmt.Sprintf("disconnect=%t", disconnect), func(t *testing.T) {
			_, controller, source := chatGPTFixture(t)
			controller.running = true
			before := &config.IntegrationConfig{Models: []string{"custom-primary"}, Primary: "custom-primary"}
			if err := config.SwapPrimaryModelSelection(chatGPTIntegrationName, nil, before); err != nil {
				t.Fatal(err)
			}
			finished := make(chan error, 1)
			consumed := false
			t.Cleanup(func() {
				if !consumed {
					<-finished
				}
			})
			load := func(ctx context.Context, names []string) (string, []LaunchModel, error) {
				return source.Resolve(ctx, names, false)
			}
			err := ApplyChatGPTModels(t.Context(), controller, load, ChatGPTApplyRequest{
				UseSavedSelection: true,
				ConfirmRestart: func() (bool, error) {
					go func() {
						if disconnect {
							finished <- RestoreChatGPTModels(controller, true)
						} else {
							finished <- ResetChatGPTModels()
						}
					}()
					select {
					case err := <-finished:
						consumed = true
						return true, err
					case <-time.After(time.Second):
						return false, errors.New("Settings was blocked by the unanswered CLI prompt")
					}
				},
			})
			expected := "preserving the newer selection"
			if disconnect {
				expected = "connection changed"
			}
			if err == nil || !strings.Contains(err.Error(), expected) {
				t.Fatalf("pending prompt did not preserve the completed Settings change: %v", err)
			}
			saved, err := LoadChatGPTConfiguration()
			if err != nil {
				t.Fatal(err)
			}
			if controller.updates != 0 {
				t.Fatalf("pending launch applied over a later Settings change: updates=%d", controller.updates)
			}
			if disconnect {
				if !slices.Equal(saved.Models, before.Models) || saved.Primary != before.Primary || controller.OllamaConfigured() {
					t.Fatalf("pending launch undid disconnect: saved=%+v configured=%t", saved, controller.OllamaConfigured())
				}
			} else if len(saved.Models) != 0 || saved.Primary != "" {
				t.Fatalf("pending launch overwrote Settings reset: saved=%+v", saved)
			}
		})
	}
}
