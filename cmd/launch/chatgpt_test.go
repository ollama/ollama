package launch

import (
	"context"
	"encoding/json"
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
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/config"
)

type chatGPTFixtureTransport struct {
	t             *testing.T
	reject        bool
	beforeRequest func(*http.Request)
}

func (f chatGPTFixtureTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	if f.beforeRequest != nil {
		f.beforeRequest(r)
	}
	if f.reject {
		f.t.Fatalf("ordinary relaunch rediscovered models: %s", r.URL.Path)
	}
	body := `{}`
	switch r.URL.Path {
	case "/api/status":
		body = `{"cloud":{"disabled":false}}`
	case "/api/tags":
		body = `{"models":[{"name":"first"},{"name":"second"},{"name":"other"}]}`
	case "/api/show":
		body = `{"capabilities":["completion","tools"],"model_info":{"general.context_length":32768}}`
	default:
		f.t.Fatalf("unexpected fixture request %s", r.URL.Path)
	}
	return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body)), Request: r}, nil
}

func chatGPTFixture(t *testing.T) (string, *CodexApp, *launcherClient, *int) {
	t.Helper()
	home := t.TempDir()
	setTestHome(t, home)
	withInteractiveSession(t, true)
	withCodexAppPlatform(t, "darwin")
	withCodexAppRouterHealth(t, func() error { return nil })
	withLauncherHooks(t)
	// Only a configuration value. API transport and native actions are mocked.
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:1")
	oldStat, oldProfile := codexAppStat, codexAppProfileIsRunning
	codexAppStat = func(string) (os.FileInfo, error) { return nil, os.ErrNotExist }
	codexAppProfileIsRunning = func(context.Context) (bool, error) { return false, nil }
	t.Cleanup(func() { codexAppStat, codexAppProfileIsRunning = oldStat, oldProfile })
	opens := 0
	withCodexAppProcessHooks(t, func() bool { return false }, func() error { t.Fatal("unexpected quit"); return nil }, func() error { opens++; return nil })
	codexAppCanOpenID = func(context.Context) bool { return true }
	client, err := newLauncherClient(defaultLaunchPolicy(true, false))
	if err != nil {
		t.Fatal(err)
	}
	client.apiClient = api.NewClient(&url.URL{Scheme: "http", Host: "fixture.invalid"}, &http.Client{Transport: chatGPTFixtureTransport{t: t}})
	client.inventory, client.recommendationsLoaded = nil, true
	return home, &CodexApp{}, client, &opens
}

func TestChatGPTSharedConfigurationLifecycle(t *testing.T) {
	home, app, client, opens := chatGPTFixture(t)
	path := filepath.Join(home, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	original := "model = \"native-model\"\nmodel_provider = \"openai\"\n\n[profiles.terminal]\nmodel = \"terminal-model\"\n"
	if err := os.WriteFile(path, []byte(original), 0o600); err != nil {
		t.Fatal(err)
	}
	for name, content := range map[string]string{"auth.json": `{"OPENAI_API_KEY":"fixture-key"}`, "history.jsonl": "fixture-history"} {
		if err := os.WriteFile(filepath.Join(filepath.Dir(path), name), []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if err := config.SaveIntegration("codex", []string{"terminal-model"}); err != nil {
		t.Fatal(err)
	}
	// Existing app connections predate the separate starting-model field.
	saved := &config.IntegrationConfig{Models: []string{"second", "first"}}
	if err := config.SwapPrimaryModelSelection(chatGPTIntegrationName, nil, saved); err != nil {
		t.Fatal(err)
	}
	if err := app.ConfigureWithModels("first", testLaunchModels(saved.Models...)); err != nil {
		t.Fatal(err)
	}
	for i, request := range []IntegrationLaunchRequest{{}, {ModelOverride: "second", ConfigureOnly: true}, {}} {
		before, _ := chatGPTSavedSelection()
		if err := client.launchChatGPT(t.Context(), app, before, request); err != nil {
			t.Fatal(err)
		}
		if i == 0 && app.CurrentModel() != "first" {
			t.Fatal("upgrade changed the app's existing starting model")
		}
	}
	saved, _ = chatGPTSavedSelection()
	if !slices.Equal(saved.Models, []string{"second", "first"}) || saved.Primary != "second" || app.CurrentModel() != "second" || *opens != 2 {
		t.Fatalf("shared selection=%+v native=%s opens=%d", saved, app.CurrentModel(), *opens)
	}
	if err := app.Restore(); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil || !strings.Contains(string(data), `model = "native-model"`) || !strings.Contains(string(data), "[profiles.terminal]") || *opens != 2 {
		t.Fatalf("disconnect changed unrelated state or opened the stopped app: %s, %v", data, err)
	}
	if err := client.launchChatGPT(t.Context(), app, saved, IntegrationLaunchRequest{}); err != nil {
		t.Fatal(err)
	}
	models, _ := codexAppRoutingModels()
	if app.CurrentModel() != "second" || !slices.Equal(models, saved.Models) || !slices.Equal(config.IntegrationModels("codex"), []string{"terminal-model"}) {
		t.Fatalf("reconnect lost the shared selection: %s %v", app.CurrentModel(), models)
	}
	for name, want := range map[string]string{"auth.json": `{"OPENAI_API_KEY":"fixture-key"}`, "history.jsonl": "fixture-history"} {
		got, err := os.ReadFile(filepath.Join(filepath.Dir(path), name))
		if err != nil || string(got) != want {
			t.Fatalf("fixture %s changed", name)
		}
	}
}

func TestChatGPTSharedConfigurationRestart(t *testing.T) {
	for _, changeModel := range []bool{false, true} {
		for _, cancelRestart := range []bool{false, true} {
			t.Run(fmt.Sprintf("change=%t/cancel=%t", changeModel, cancelRestart), func(t *testing.T) {
				home, app, client, opens := chatGPTFixture(t)
				saved := &config.IntegrationConfig{Models: []string{"first", "second"}, Primary: "first"}
				if err := config.SwapPrimaryModelSelection(chatGPTIntegrationName, nil, saved); err != nil {
					t.Fatal(err)
				}
				if err := app.ConfigureWithModels(saved.Primary, testLaunchModels(saved.Models...)); err != nil {
					t.Fatal(err)
				}
				path := filepath.Join(home, ".codex", "config.toml")
				profile, _ := os.ReadFile(path)
				cancel := make(chan struct{})
				oldSpinner, oldSleep := DefaultSpinner, codexAppSleep
				stopped := false
				DefaultSpinner = func(string) *Spinner { return NewSpinner(func() { stopped = true }, cancel) }
				t.Cleanup(func() { DefaultSpinner, codexAppSleep = oldSpinner, oldSleep })
				running, quits := true, 0
				withCodexAppProcessHooks(t, func() bool { return running }, func() error { return nil }, func() error { *opens++; return nil })
				codexAppQuitApp = func(ctx context.Context) error {
					if _, bounded := ctx.Deadline(); bounded {
						t.Fatal("interactive CLI inherited the desktop quit deadline")
					}
					quits++
					return nil
				}
				codexAppSleep = func(time.Duration) {
					if cancelRestart {
						close(cancel)
					} else {
						running = false
					}
				}
				DefaultConfirmPrompt = func(string, ConfirmOptions) (bool, error) { return true, nil }
				request := IntegrationLaunchRequest{}
				if changeModel {
					request.ModelOverride = "second"
				} else {
					client.apiClient = api.NewClient(&url.URL{Scheme: "http", Host: "fixture.invalid"}, &http.Client{Transport: chatGPTFixtureTransport{t: t, reject: true}})
				}
				err := client.launchChatGPT(t.Context(), app, saved, request)
				if cancelRestart {
					after, _ := os.ReadFile(path)
					latest, _ := chatGPTSavedSelection()
					if !errors.Is(err, ErrCancelled) || !chatGPTSelectionEqual(saved, latest) || !slices.Equal(profile, after) || *opens != 0 {
						t.Fatalf("cancel changed state: err=%v saved=%+v opens=%d", err, latest, *opens)
					}
				} else if err != nil || *opens != 1 {
					t.Fatalf("restart err=%v opens=%d", err, *opens)
				}
				if !stopped || quits != 1 {
					t.Fatalf("spinner stopped=%t quits=%d", stopped, quits)
				}
			})
		}
	}
}

func TestChatGPTSharedConfigurationPreservesNewerChoice(t *testing.T) {
	_, app, client, opens := chatGPTFixture(t)
	saved := &config.IntegrationConfig{Models: []string{"first", "second"}, Primary: "first"}
	if err := config.SwapPrimaryModelSelection(chatGPTIntegrationName, nil, saved); err != nil {
		t.Fatal(err)
	}
	if err := app.ConfigureWithModels(saved.Primary, testLaunchModels(saved.Models...)); err != nil {
		t.Fatal(err)
	}
	codexAppIsRunning = func(context.Context) bool { return true }
	later := &config.IntegrationConfig{Models: saved.Models, Primary: "second"}
	DefaultConfirmPrompt = func(string, ConfirmOptions) (bool, error) {
		return true, config.SwapPrimaryModelSelection(chatGPTIntegrationName, saved, later)
	}
	err := client.launchChatGPT(t.Context(), app, saved, IntegrationLaunchRequest{})
	latest, _ := chatGPTSavedSelection()
	if err == nil || !strings.Contains(err.Error(), "changed") || !chatGPTSelectionEqual(latest, later) || *opens != 0 {
		t.Fatalf("stale confirmation overwrote later choice: %v %+v", err, latest)
	}
}

func TestChatGPTSharedConfigurationModelOverride(t *testing.T) {
	for _, tc := range []struct {
		name      string
		models    []string
		primary   string
		override  string
		wantError bool
	}{
		{"add alias", []string{"first", "second"}, "first", "other:latest", false},
		{"app defaults", nil, "", "second", false},
		{"full list", []string{"first", "second", "third", "fourth", "fifth"}, "first", "other", true},
		{"invalid saved primary", []string{"first", "second"}, "missing", "", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, app, client, opens := chatGPTFixture(t)
			saved := &config.IntegrationConfig{Models: tc.models, Primary: tc.primary}
			if err := config.SwapPrimaryModelSelection(chatGPTIntegrationName, nil, saved); err != nil {
				t.Fatal(err)
			}
			if tc.models == nil {
				if err := app.ConfigureWithModels("first", testLaunchModels("first", "second")); err != nil {
					t.Fatal(err)
				}
			}
			err := client.launchChatGPT(t.Context(), app, saved, IntegrationLaunchRequest{ModelOverride: tc.override})
			latest, _ := chatGPTSavedSelection()
			if tc.wantError {
				if err == nil || !chatGPTSelectionEqual(saved, latest) || *opens != 0 {
					t.Fatalf("invalid override changed state: %v %+v", err, latest)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			wantModels := append(slices.Clone(tc.models), tc.override)
			if tc.models == nil {
				wantModels = []string{"first", "second"}
			}
			if !slices.Equal(latest.Models, wantModels) || latest.Primary != tc.override {
				t.Fatalf("override replaced app selection: %+v", latest)
			}
			data, _ := os.ReadFile(mustCodexAppModelCatalogPath(t))
			var catalog struct {
				Models []struct {
					Slug    string `json:"slug"`
					Context int    `json:"context_window"`
				}
			}
			if err := json.Unmarshal(data, &catalog); err != nil {
				t.Fatal(err)
			}
			matches := 0
			for _, model := range catalog.Models {
				if codexAppCatalogModelKey(model.Slug) == codexAppCatalogModelKey(tc.override) {
					matches++
					if model.Context != 32768 {
						t.Fatalf("lost model metadata: %+v", model)
					}
				}
			}
			if matches != 1 {
				t.Fatalf("primary alias has %d catalog entries", matches)
			}
		})
	}
}

func chatGPTReviewConnectedFixture(t *testing.T) (string, *CodexApp, *launcherClient, *int, *config.IntegrationConfig) {
	t.Helper()
	home, app, client, opens := chatGPTFixture(t)
	saved := &config.IntegrationConfig{Models: []string{"first", "second"}, Primary: "first"}
	if err := config.SwapPrimaryModelSelection(chatGPTIntegrationName, nil, saved); err != nil {
		t.Fatal(err)
	}
	if err := app.ConfigureWithModels(saved.Primary, testLaunchModels(saved.Models...)); err != nil {
		t.Fatal(err)
	}
	return home, app, client, opens, saved
}

func TestChatGPTReviewRequestCancellationAbortsRestart(t *testing.T) {
	for _, change := range []bool{false, true} {
		t.Run(fmt.Sprintf("model_change=%t", change), func(t *testing.T) {
			home, app, client, opens, saved := chatGPTReviewConnectedFixture(t)
			path := filepath.Join(home, ".codex", "config.toml")
			before, _ := os.ReadFile(path)
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			running := true
			withCodexAppProcessHooks(t, func() bool { return running }, func() error {
				cancel()
				running = false
				return nil
			}, func() error { *opens++; return nil })
			DefaultConfirmPrompt = func(string, ConfirmOptions) (bool, error) { return true, nil }
			request := IntegrationLaunchRequest{}
			if change {
				request.ModelOverride = "second"
			}
			err := client.launchChatGPT(ctx, app, saved, request)
			after, _ := os.ReadFile(path)
			latest, _ := chatGPTSavedSelection()
			if !errors.Is(err, ErrCancelled) || *opens != 0 || !slices.Equal(before, after) || !chatGPTSelectionEqual(saved, latest) {
				t.Fatalf("request cancellation changed or reopened the profile: err=%v opens=%d saved=%+v", err, *opens, latest)
			}
		})
	}
}

func TestChatGPTReviewLegacyRestartConsent(t *testing.T) {
	for _, change := range []bool{false, true} {
		for _, mode := range []string{"accept", "cancel", "headless-yes"} {
			t.Run(fmt.Sprintf("model_change=%t/%s", change, mode), func(t *testing.T) {
				_, app, client, opens, saved := chatGPTReviewConnectedFixture(t)
				legacyRunning, stops, prompts := true, 0, 0
				previousStop := codexAppStopProfile
				t.Cleanup(func() { codexAppStopProfile = previousStop })
				codexAppProfileIsRunning = func(context.Context) (bool, error) { return legacyRunning, nil }
				codexAppStopProfile = func(context.Context) error { stops++; legacyRunning = false; return nil }
				DefaultConfirmPrompt = func(string, ConfirmOptions) (bool, error) { prompts++; return mode != "cancel", nil }
				if mode == "headless-yes" {
					withInteractiveSession(t, false)
					client.policy = defaultLaunchPolicy(false, true)
				}
				wantPrimary := saved.Primary
				request := IntegrationLaunchRequest{}
				if change {
					wantPrimary = "second"
				}
				if change || mode == "headless-yes" {
					request.ModelOverride = wantPrimary
				}
				err := client.launchChatGPT(t.Context(), app, saved, request)
				latest, _ := chatGPTSavedSelection()
				if mode == "cancel" {
					if !errors.Is(err, ErrCancelled) || stops != 0 || *opens != 0 || !chatGPTSelectionEqual(saved, latest) {
						t.Fatalf("cancel err=%v stops=%d opens=%d saved=%+v", err, stops, *opens, latest)
					}
				} else if err != nil || stops != 1 || *opens != 1 || app.CurrentModel() != wantPrimary || latest.Primary != wantPrimary || !slices.Equal(latest.Models, saved.Models) {
					t.Fatalf("legacy reconnect err=%v stops=%d opens=%d saved=%+v", err, stops, *opens, latest)
				}
				wantPrompts := 1
				if mode == "headless-yes" {
					wantPrompts = 0
				}
				if prompts != wantPrompts {
					t.Fatalf("prompts=%d, want %d", prompts, wantPrompts)
				}
			})
		}
	}
}

func TestChatGPTReviewFailedOpenRestoresRunningProfile(t *testing.T) {
	for _, connected := range []bool{false, true} {
		for _, configureOnly := range []bool{false, true} {
			t.Run(fmt.Sprintf("connected=%t/configure_only=%t", connected, configureOnly), func(t *testing.T) {
				home, app, client, opens, saved := chatGPTReviewConnectedFixture(t)
				wantCurrent := saved.Primary
				if !connected {
					if err := app.RestoreFromDesktop(false); err != nil {
						t.Fatal(err)
					}
					if err := os.WriteFile(filepath.Join(home, ".codex", "config.toml"), []byte("model = \"native-model\"\n"), 0o600); err != nil {
						t.Fatal(err)
					}
					wantCurrent = ""
				}
				running := true
				openFailure := errors.New("fixture open failed")
				withCodexAppProcessHooks(t, func() bool { return running }, func() error { running = false; return nil }, func() error {
					*opens++
					if *opens == 1 {
						return openFailure
					}
					if app.CurrentModel() != wantCurrent {
						t.Fatalf("reopened the failed selection instead of the previous starting model: %s", app.CurrentModel())
					}
					running = true
					return nil
				})
				DefaultConfirmPrompt = func(string, ConfirmOptions) (bool, error) { return true, nil }
				err := client.launchChatGPT(t.Context(), app, saved, IntegrationLaunchRequest{ModelOverride: "second", ConfigureOnly: configureOnly})
				latest, _ := chatGPTSavedSelection()
				if !errors.Is(err, openFailure) || !running || *opens != 2 || app.CurrentModel() != wantCurrent || app.OllamaConfigured() != connected || !chatGPTSelectionEqual(latest, saved) {
					t.Fatalf("failed switch did not restore the running profile: err=%v running=%t opens=%d saved=%+v", err, running, *opens, latest)
				}
				if !connected {
					data, err := os.ReadFile(filepath.Join(home, ".codex", "config.toml"))
					if err != nil || !strings.Contains(string(data), `model = "native-model"`) {
						t.Fatalf("recovery lost the original native model: %s/%v", data, err)
					}
				}
			})
		}
	}
}

func TestChatGPTReviewFullListRejectsBeforeModelPreparation(t *testing.T) {
	for _, picker := range []bool{false, true} {
		t.Run(fmt.Sprintf("picker=%t", picker), func(t *testing.T) {
			_, app, client, opens, before := chatGPTReviewConnectedFixture(t)
			full := &config.IntegrationConfig{Models: []string{"first", "second", "third", "fourth", "fifth"}, Primary: "first"}
			if err := config.SwapPrimaryModelSelection(chatGPTIntegrationName, before, full); err != nil {
				t.Fatal(err)
			}
			if err := app.ConfigureWithModels(full.Primary, testLaunchModels(full.Models...)); err != nil {
				t.Fatal(err)
			}
			chosen := !picker
			client.apiClient = api.NewClient(&url.URL{Scheme: "http", Host: "fixture.invalid"}, &http.Client{Transport: chatGPTFixtureTransport{t: t, beforeRequest: func(r *http.Request) {
				if chosen {
					t.Fatalf("prepared an unselected sixth model before rejecting it: %s", r.URL.Path)
				}
			}}})
			DefaultSingleSelectorWithUpdates = nil
			DefaultSingleSelector = func(string, []SelectionItem, string) (string, error) { chosen = true; return "other", nil }
			request := IntegrationLaunchRequest{ModelOverride: "other"}
			if picker {
				request = IntegrationLaunchRequest{ForceConfigure: true, ConfigureOnly: true}
			}
			err := client.launchChatGPT(t.Context(), app, full, request)
			latest, _ := chatGPTSavedSelection()
			if err == nil || !strings.Contains(err.Error(), "five") || *opens != 0 || !chatGPTSelectionEqual(full, latest) {
				t.Fatalf("invalid sixth model changed the connection: err=%v opens=%d saved=%+v", err, *opens, latest)
			}
		})
	}
}
